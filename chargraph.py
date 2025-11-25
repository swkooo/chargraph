import json
import time
import argparse
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import google.genai as genai
from dotenv import load_dotenv

class FileHandler:
    @staticmethod
    def read_file(filename: str) -> str:
        """Read content from a file."""
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    
    @staticmethod
    def read_json(filename: str) -> Optional[Dict[str, Any]]:
        """Read JSON from a file if it exists."""
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    @staticmethod
    def write_json(filename: str, content: Dict[str, Any]) -> None:
        """Write JSON content to a file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=4)

class APIClient:
    def __init__(self, api_keys: List[str], model: Optional[str] = None):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.model = model
        self.max_retries = 100
        self.retry_delay = 60  # seconds
        # Initialize genai client with first key
        self.client = genai.Client(api_key=self.get_current_key())
    
    def get_current_key(self) -> str:
        """Get the current API key."""
        return self.api_keys[self.current_key_index]
    
    def switch_to_next_key(self) -> bool:
        """Switch to the next API key. Returns True if switched, False if no more keys."""
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self.client = genai.Client(api_key=self.get_current_key())
            print(f"Switched to API key {self.current_key_index + 1}/{len(self.api_keys)}")
            return True
        return False
    
    def is_quota_exceeded_error(self, error: Exception) -> bool:
        """Check if the error is a quota/rate limit error."""
        error_str = str(error).lower()
        quota_indicators = [
            '429',
            'resource_exhausted',
            'quota',
            'rate limit',
            'rate_limit',
            'too many requests'
        ]
        return any(indicator in error_str for indicator in quota_indicators)
    
    def create_messages(self, text: str, previous_json: Optional[Dict[str, Any]] = None, 
                       desc_sentences: Optional[int] = None, generate_portraits: bool = False,
                       copies: int = 1, max_main_characters: Optional[int] = None) -> list:
        """Create messages for the API request."""
        system_prompt = """You are a literary analyst specializing in character extraction and relationship mapping. Your task is to:

1. Character Identification:"""
        
        if max_main_characters is not None:
            system_prompt += f"""
   - FIRST, identify the PROTAGONIST (main character) of the story - the central character around whom the plot revolves
   - THEN, extract the protagonist plus other characters who have relationship weights of 3.0 or higher with the protagonist
   - Character selection criteria:
     * The protagonist (1 character) - the main character of the story
     * Characters with relationship weight >= 3.0 with the protagonist, ranked by weight (highest first)
     * Maximum of {max_main_characters - 1} additional characters (excluding protagonist)
     * If fewer than {max_main_characters - 1} characters have weight >= 3.0, include only those that meet the criteria (do NOT force to reach {max_main_characters} total)
     * Do NOT filter by main_character status - select based ONLY on relationship weight with the protagonist
     * Include characters regardless of whether they are main or supporting characters
     * Focus on characters who have meaningful interactions/relationships with the protagonist (weight >= 3.0)
   - Assign unique ID numbers to each character (ensure no duplicates)
   - Determine their common name (most frequently used in text)
   - List ALL references to them (nicknames, titles, etc.)
   - IMPORTANT: Extract the protagonist plus all characters with relationship weight >= 3.0 to the protagonist (up to {max_main_characters} total characters). If fewer characters meet the weight >= 3.0 criteria, extract only those that qualify."""
        else:
            system_prompt += """
   - Extract EVERY character mentioned in the text:
     * Include all characters regardless of their role or significance
     * Do not skip minor or briefly mentioned characters
     * If a character is named or described, they must be included
   - Assign unique ID numbers to each character (ensure no duplicates)
   - Determine their common name (most frequently used in text)
   - List ALL references to them (nicknames, titles, etc.)
   - Identify main characters based on:
     * Frequency of appearance
     * Plot significance
     * Number of interactions with others"""
        
        system_prompt += """

2. Relationship Analysis:
   - Document ALL character interactions, even brief ones
   - Ensure no duplicate relationships (check both directions: A→B and B→A)
   - For each relationship, provide:
     * Weight (1-10) based on:
       - Frequency of interaction
       - Significance of interactions
       - Impact on plot
     * Positivity scale (-1 to +1):
       - Negative values (-1 to -0.1) for hostile/antagonistic relationships
       - Zero (0) for neutral/professional relationships
       - Positive values (0.1 to 1) for friendly/supportive relationships
       Examples:
       - -1.0: Mortal enemies, intense hatred
       - -0.5: Rivals, strong dislike
       - 0.0: Neutral acquaintances
       - 0.5: Friends, positive relationship
       - 1.0: Best friends, family, deep love
   - Include relationship descriptors (family, friends, enemies, brief encounter, lovers, met in the elevator, etc.)

3. Special Instructions:"""
        
        if max_main_characters is None:
            system_prompt += """
   - Include ALL characters, no matter how minor their role
   - Be thorough in relationship mapping
   - Consider indirect interactions
   - Note character development and changing relationships
   - Ensure every character has at least one connection
   - Check for and eliminate any duplicate characters or relationships
   - Never omit a character just because they:
     * Appear only briefly
     * Have few or weak relationships
     * Seem insignificant to the plot
     * Are only mentioned in passing"""
        else:
            system_prompt += f"""
   - Focus ONLY on the protagonist and characters with relationship weight >= 3.0 to the protagonist (up to {max_main_characters} total characters)
   - Map relationships ONLY between these selected characters
   - Be thorough in relationship mapping between all selected characters
   - Ensure every character has at least one connection with another character
   - Only include relationships with weight >= 3.0 when selecting characters
   - If fewer than {max_main_characters} characters have weight >= 3.0 with the protagonist, include only those that qualify
   - Check for and eliminate any duplicate characters or relationships"""

        if desc_sentences is not None:
            system_prompt += f"""

4. Character Descriptions:
   - For each character, provide:
     * A concise description limited to {desc_sentences} sentences
     * Focus on their role, personality traits, and narrative significance
     * Include key story contributions and character development"""

        if generate_portraits:
            system_prompt += """
   
5. Portrait Generation:
   - For each character, create a detailed prompt for AI image generation that captures:
     * Physical appearance and distinguishing features
     * Clothing and style
     * Facial expressions and emotional state
     * Setting or background elements that reflect their role
     * Artistic style suggestions for consistent character representation"""

        if previous_json:
            if max_main_characters is not None:
                system_prompt += f"\n\nPreliminary character and relationship data: {json.dumps(previous_json)}\nCarefully update this data: ensure you have the protagonist plus characters with relationship weight >= 3.0 to the protagonist (up to {max_main_characters} total characters). If fewer characters have weight >= 3.0, include only those that qualify. Prioritize characters with highest relationship weights to the protagonist regardless of main_character status, add missing relationships, verify weights and positivity (ensure relationship weights are >= 3.0), ensure all characters have connections, and check for any duplicate characters or relationships."
            else:
                system_prompt += f"\n\nPreliminary character and relationship data: {json.dumps(previous_json)}\nCarefully update this data: add any missing characters (no matter how minor or briefly mentioned), add missing relationships, verify weights and positivity, ensure all characters have connections, and check for any duplicate characters or relationships. Every character in the text must be included, even those with minimal roles or single appearances."

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": "\n\n".join([text] * copies)}]}
        ]
    
    def get_schema(self, desc_sentences: Optional[int] = None, generate_portraits: bool = False) -> Dict[str, Any]:
        """Get the JSON schema for the API response."""
        character_properties = {
            "id": {
                "type": "NUMBER",
                "description": "Unique identifier for the character that remains consistent across iterations"
            },
            "common_name": {
                "type": "STRING",
                "description": "The most frequently used name for this character in the text"
            },
            "main_character": {
                "type": "BOOLEAN",
                "description": "True if this is a major character based on frequency of appearance, plot significance, and number of interactions"
            },
            "names": {
                "type": "ARRAY",
                "description": "All variations of the character's name, including nicknames, titles, and other references used in the text",
                "items": {"type": "STRING"}
            }
        }

        if desc_sentences is not None:
            character_properties["description"] = {
                "type": "STRING",
                "description": "Character's role in the story, personality traits, and narrative significance"
            }

        if generate_portraits:
            character_properties["portrait_prompt"] = {
                "type": "STRING",
                "description": "Detailed prompt for AI image generation of the character"
            }

        return {
            "type": "OBJECT",
            "properties": {
                "characters": {
                    "type": "ARRAY",
                    "description": "Characters and connections.",
                    "items": {
                        "type": "OBJECT",
                        "properties": character_properties,
                        "required": ["id", "names", "common_name", "main_character"]
                    }
                },
                "relations": {
                    "type": "ARRAY",
                    "description": "List of each pair of characters who met",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "id1": {
                                "type": "NUMBER",
                                "description": "Unique identifier of the first character in the relationship pair",
                            },
                            "id2": {
                                "type": "NUMBER",
                                "description": "Unique identifier of the second character in the relationship pair",
                            },
                            "relation": {
                                "type": "ARRAY",
                                "description": "Types of relationships between the characters (e.g., family, friends, enemies, colleagues)",
                                "items": {"type": "STRING"}
                            },
                            "weight": {
                                "type": "NUMBER",
                                "description": "Strength of the relationship from 1 (minimal) to 10 (strongest) based on frequency and significance of interactions"
                            },
                            "positivity": {
                                "type": "NUMBER",
                                "description": "Emotional quality of the relationship from -1 (hostile) through 0 (neutral) to 1 (positive)"
                            }
                        },
                        "required": ["id1", "id2", "relation", "weight", "positivity"]
                    }
                },
            },
            "required": ["characters", "relations"]
        }

    def make_request(self, messages: list, desc_sentences: Optional[int] = None, generate_portraits: bool = False, temperature: float = 1) -> dict:
        """Make API request with retry mechanism and automatic key rotation."""
        # Gemini implementation using new google.genai API
        model_name = "gemini-2.5-flash" if self.model is None else self.model
        combined_prompt = f"{messages[0]['content']}\n\nInput text:\n{messages[1]['content'][0]['text']}"
        
        for attempt in range(self.max_retries):
            try:
                # Use new google.genai API with structured output
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=combined_prompt,
                    config={
                        "temperature": temperature,
                        "response_mime_type": "application/json",
                        "response_schema": self.get_schema(desc_sentences, generate_portraits)
                    }
                )
                return response
                
            except Exception as e:
                error_str = str(e)
                print(f"Error during attempt {attempt + 1} with key {self.current_key_index + 1}: {error_str}")
                
                # Check if it's a quota/rate limit error
                if self.is_quota_exceeded_error(e):
                    print("Quota/Rate limit exceeded. Attempting to switch to next API key...")
                    if self.switch_to_next_key():
                        # Reset attempt counter when switching keys
                        continue
                    else:
                        print("No more API keys available. Waiting before retry...")
                
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
        
        raise Exception("Max retries exceeded with all API keys")

class CharacterExtractor:
    def __init__(self, api_keys: List[str], model: Optional[str] = None):
        self.file_handler = FileHandler()
        self.api_client = APIClient(api_keys, model)
    
    def get_output_filename(self, base_filename: str, iteration: int) -> str:
        """Generate output filename with iteration number."""
        path = Path(base_filename)
        return str(path.parent / f"{path.stem}_{iteration}{path.suffix}")
    
    def create_social_network(self, data: Dict[str, Any]) -> nx.Graph:
        """Create a NetworkX graph from character data."""
        G = nx.Graph()
        
        characters = data['characters']
        relations = data['relations']
        
        # Add nodes (characters)
        for character in characters:
            G.add_node(
                character['id'],
                common_name=character['common_name'],
                main_character=character['main_character']
            )
        
        # Add edges (relations)
        for relation in relations:
            # If edge exists, append new relations, otherwise create new edge
            if G.has_edge(relation['id1'], relation['id2']):
                G[relation['id1']][relation['id2']]['weight'] += relation['weight']
            else:
                G.add_edge(
                    relation['id1'],
                    relation['id2'],
                    weight=relation['weight']+1,
                    positivity=relation['positivity']
                )
        
        return G

    def plot_network(self, G: nx.Graph, image_file: str) -> None:
        """Plot and save the character network graph."""
        plt.figure(figsize=(15, 15))
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare node colors and sizes
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if G.nodes[node]['main_character']:
                node_colors.append('#FF6B6B')  # Coral red for main characters
                node_sizes.append(2000)
            else:
                node_colors.append('#4ECDC4')  # Turquoise for other characters
                node_sizes.append(1000)
        
        # Draw edges with varying thickness and color based on weight and positivity
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [20 * (w / max_weight) for w in edge_weights]
        
        edge_colors = []
        for u, v in G.edges():
            positivity = G[u][v]['positivity']
            if positivity < -0.1:
                edge_colors.append('red')
            elif positivity > 0.1:
                edge_colors.append('green')
            else:
                edge_colors.append('grey')
        
        # Draw the network
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=edge_widths, edge_color=edge_colors)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
        
        # Add labels with white background for better visibility
        labels = nx.get_node_attributes(G, 'common_name')
        for node, (x, y) in pos.items():
            plt.text(x, y, labels[node],
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                    horizontalalignment='center',
                    verticalalignment='center')
        
        plt.title("Character Relationship Network", fontsize=16, pad=20)
        plt.axis('off')
        
        # Add legend for nodes
        plt.plot([], [], 'o', color='#FF6B6B', label='Main Characters', markersize=15)
        plt.plot([], [], 'o', color='#4ECDC4', label='Supporting Characters', markersize=15)
        
        # Add legend for edges
        plt.plot([], [], color='red', label='Negative Relations', linewidth=3)
        plt.plot([], [], color='grey', label='Neutral Relations', linewidth=3)
        plt.plot([], [], color='green', label='Positive Relations', linewidth=3)
        
        plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(image_file, dpi=300, bbox_inches='tight')
        plt.close()

    def process_text(self, input_file: str, output_file: str, previous_json_file: Optional[str] = None,
                    iterations: int = 1, delay: int = 300, plot_graph: bool = False,
                    desc_sentences: Optional[int] = None, generate_portraits: bool = False,
                    copies: int = 1, temperature: float = 1.0, max_main_characters: Optional[int] = None) -> None:
        """Process text file and extract character information."""
        text = self.file_handler.read_file(input_file)
        previous_json = None
        
        for i in range(iterations):
            if i > 0:
                # Use previous iteration's output as input
                previous_json = self.file_handler.read_json(
                    self.get_output_filename(output_file, i - 1)
                )
                # Wait between iterations
                print(f"Waiting {delay} seconds before next iteration...")
                time.sleep(delay)
            elif previous_json_file:
                previous_json = self.file_handler.read_json(previous_json_file)
            
            print(f"Starting iteration {i + 1}/{iterations}")
            
            # Add retry mechanism for the entire API request and JSON parsing process
            max_retries = 10
            retry_delay = 10
            for attempt in range(max_retries):
                try:
                    messages = self.api_client.create_messages(text, previous_json, desc_sentences, generate_portraits, copies, max_main_characters)
                    result = self.api_client.make_request(messages, desc_sentences, generate_portraits, temperature)
                    
                    output_filename = self.get_output_filename(output_file, i)
                                        
                    # Log response content for debugging
                    debug_filename = str(Path(output_filename).with_suffix('.debug.txt'))
                    with open(debug_filename, 'w', encoding='utf-8') as f:
                        f.write(f"Attempt {attempt + 1} Result Content:\n{result}")

                    # New google.genai API returns JSON directly
                    if hasattr(result, 'text'):
                        content = json.loads(result.text)
                    elif hasattr(result, 'content'):
                        content = json.loads(result.content)
                    else:
                        # Fallback: try to parse as JSON string
                        content = json.loads(str(result))
                    
                    # Try to parse and validate the JSON structure
                    #content = json.loads(response_content)
                    
                    # Basic validation of required fields
                    if not isinstance(content, dict):
                        raise ValueError("Response content is not a JSON object")
                    if "characters" not in content or "relations" not in content:
                        raise ValueError("Missing required top-level fields")
                    if not isinstance(content["characters"], list) or not isinstance(content["relations"], list):
                        raise ValueError("characters and relations must be arrays")
                    
                    # Save the validated content
                    self.file_handler.write_json(output_filename, content)
                    print(f"Results saved to {output_filename}")
                    
                    if plot_graph:
                        # Set random seed for consistent layouts
                        import random
                        random.seed(42)
                        
                        # Create and save graph
                        image_filename = str(Path(output_filename).with_suffix('.png'))
                        G = self.create_social_network(content)
                        self.plot_network(G, image_filename)
                        print(f"Graph saved to {image_filename}")
                    
                    # If we get here, the iteration was successful
                    break
                    
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Error on attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise Exception(f"Failed to process iteration {i + 1} after {max_retries} attempts")
        
        # After all iterations, keep only the last result and delete intermediate files
        if iterations > 1:
            print(f"\nCleaning up intermediate files, keeping only the last iteration...")
            last_iteration = iterations - 1
            final_output = self.get_output_filename(output_file, last_iteration)
            final_debug = str(Path(final_output).with_suffix('.debug.txt'))
            final_image = str(Path(final_output).with_suffix('.png'))
            
            # Rename last iteration files to final names (without iteration number)
            final_base = str(Path(output_file).parent / Path(output_file).stem)
            final_json = f"{final_base}.json"
            final_debug_renamed = f"{final_base}.debug.txt"
            final_image_renamed = f"{final_base}.png"
            
            # Move last iteration files to final names
            if Path(final_output).exists():
                Path(final_output).rename(final_json)
                print(f"Renamed {final_output} to {final_json}")
            
            if Path(final_debug).exists():
                Path(final_debug).rename(final_debug_renamed)
                print(f"Renamed {final_debug} to {final_debug_renamed}")
            
            if plot_graph and Path(final_image).exists():
                Path(final_image).rename(final_image_renamed)
                print(f"Renamed {final_image} to {final_image_renamed}")
            
            # Delete intermediate iteration files
            for i in range(iterations - 1):
                intermediate_json = self.get_output_filename(output_file, i)
                intermediate_debug = str(Path(intermediate_json).with_suffix('.debug.txt'))
                intermediate_image = str(Path(intermediate_json).with_suffix('.png'))
                
                if Path(intermediate_json).exists():
                    Path(intermediate_json).unlink()
                    print(f"Deleted {intermediate_json}")
                
                if Path(intermediate_debug).exists():
                    Path(intermediate_debug).unlink()
                    print(f"Deleted {intermediate_debug}")
                
                if plot_graph and Path(intermediate_image).exists():
                    Path(intermediate_image).unlink()
                    print(f"Deleted {intermediate_image}")
            
            print(f"Cleanup complete. Final result saved as: {final_json}")

def cleanup_genai():
    """Clean up Gemini API resources."""
    # New google.genai API doesn't require explicit cleanup
    # Client resources are automatically managed
    pass

def load_api_keys_from_env(env_path: Path) -> List[str]:
    """Load API keys from .env file. Supports comma-separated format:
    GEMINI_API_KEY=key1,key2,key3
    """
    if not env_path.exists():
        raise ValueError(f".env file not found at {env_path}")
    
    # Load .env file
    load_dotenv(env_path)
    
    api_keys = []
    
    # Get GEMINI_API_KEYS (supports comma-separated format)
    api_key_value = os.getenv("GEMINI_API_KEYS")
    if api_key_value:
        # Check if it contains commas (multiple keys in one line)
        if ',' in api_key_value:
            # Split by comma and strip whitespace
            keys = [key.strip() for key in api_key_value.split(',') if key.strip()]
            api_keys.extend(keys)
        else:
            # Single key
            api_keys.append(api_key_value)
    
    if not api_keys:
        raise ValueError(f"No GEMINI_API_KEYS found in {env_path}. Please add GEMINI_API_KEYS (comma-separated format: key1,key2,key3)")
    
    print(f"Loaded {len(api_keys)} API key(s) from .env file")
    return api_keys

def main():
    parser = argparse.ArgumentParser(description='Extract characters and their relationships from text.')
    parser.add_argument('input_file', nargs='?', help='Input text file to analyze (optional, auto-detects from origin_txt)')
    parser.add_argument('output_file', nargs='?', help='Base name for output JSON files (optional, auto-generates)')
    parser.add_argument('-iter', '--iterations', type=int, default=1,
                      help='Number of iterations to run (default: 1)')
    parser.add_argument('-delay', '--delay', type=int, default=60,
                      help='Delay between iterations in seconds (default: 60)')
    parser.add_argument('-prev', '--previous', 
                      help='Previous JSON file to use as initial data')
    parser.add_argument('-plot', '--plot_graph', action='store_true',
                      help='Generate character relationship graph for each iteration')
    parser.add_argument('-desc', '--desc_sentences', type=int,
                      help='Number of sentences to use for character descriptions')
    parser.add_argument('-portrait', '--generate_portraits', action='store_true',
                      help='Generate AI portrait prompts for each character')
    parser.add_argument('-cp', '--copies', type=int, default=1,
                      help='Number of copies of text to send as prompt (default: 1)')
    parser.add_argument('-t', '--temperature', type=float, default=1.0,
                      help='Temperature (default: 1.0)')
    parser.add_argument('-m', '--model', type=str, default=None,
                      help='Model to use (default: gemini-2.5-flash)')
    parser.add_argument('-max-main', '--max_main_characters', type=int, default=20,
                      help='Extract protagonist + N-1 characters with highest relationship weights to protagonist (default: 20)')
    
    args = parser.parse_args()
    
    # Load API keys from .env file
    env_path = Path(__file__).parent / ".env"
    api_keys = load_api_keys_from_env(env_path)
    
    extractor = CharacterExtractor(api_keys, args.model)
    
    # Auto-detect input files from origin_txt if not provided
    if args.input_file is None:
        # Use script location to find project root (parent of chargraph folder)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        
        # Try multiple possible paths
        possible_paths = [
            project_root / "gajiAI/rag-chatbot_test/data/origin_txt",  # From script location (most reliable)
            Path("gajiAI/rag-chatbot_test/data/origin_txt"),  # From current working directory
            Path("../gajiAI/rag-chatbot_test/data/origin_txt"),  # From chargraph folder
        ]
        
        origin_txt_path = None
        for path in possible_paths:
            resolved_path = path.resolve() if path.is_absolute() or path.exists() else path
            if resolved_path.exists():
                origin_txt_path = resolved_path
                break
        
        if origin_txt_path is None:
            raise ValueError(f"origin_txt directory not found. Tried: {[str(p) for p in possible_paths]}. Current working directory: {os.getcwd()}, Script location: {script_dir}")
        
        # Find all .txt files (excluding .metadata.json)
        txt_files = sorted(list(origin_txt_path.glob("*.txt")))
        if not txt_files:
            raise ValueError(f"No .txt files found in {origin_txt_path}")
        
        print(f"Found {len(txt_files)} text file(s) in origin_txt. Processing all files...")
        
        # Process all txt files sequentially
        for txt_file in txt_files:
            input_file = str(txt_file)
            print(f"\n{'='*60}")
            print(f"Processing: {txt_file.name}")
            print(f"{'='*60}")
            
            # Auto-generate output file name from input file
            input_path = Path(input_file)
            output_base = input_path.stem
            # Use script location to find project root
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            
            # Try multiple possible paths for char_graph
            possible_char_graph_paths = [
                project_root / "gajiAI/rag-chatbot_test/data/char_graph",  # From script location (most reliable)
                Path("gajiAI/rag-chatbot_test/data/char_graph"),  # From current working directory
                Path("../gajiAI/rag-chatbot_test/data/char_graph"),  # From chargraph folder
            ]
            
            char_graph_path = None
            for path in possible_char_graph_paths:
                resolved_path = path.resolve() if path.is_absolute() or path.exists() else path
                # Check if parent directory exists or can be created
                if resolved_path.parent.exists() or resolved_path.parent.parent.exists():
                    char_graph_path = resolved_path
                    break
            
            if char_graph_path is None:
                # Use first path and create it
                char_graph_path = possible_char_graph_paths[0].resolve()
            
            char_graph_path.mkdir(parents=True, exist_ok=True)
            output_file = str(char_graph_path / output_base)
            
            print(f"Input: {input_file}")
            print(f"Output: {output_file}")
            
            extractor.process_text(
                input_file=input_file,
                output_file=output_file,
                previous_json_file=args.previous,
                iterations=args.iterations,
                delay=args.delay,
                plot_graph=args.plot_graph,
                desc_sentences=args.desc_sentences,
                generate_portraits=args.generate_portraits,
                copies=args.copies,
                temperature=args.temperature,
                max_main_characters=args.max_main_characters
            )
            
            print(f"Completed: {txt_file.name}\n")
    else:
        # Single file mode (manual input)
        input_file = args.input_file
        
        # Auto-generate output file name from input file if not provided
        if args.output_file is None:
            input_path = Path(input_file)
            output_base = input_path.stem
            # Use script location to find project root
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            
            # Try multiple possible paths for char_graph
            possible_char_graph_paths = [
                project_root / "gajiAI/rag-chatbot_test/data/char_graph",  # From script location (most reliable)
                Path("gajiAI/rag-chatbot_test/data/char_graph"),  # From current working directory
                Path("../gajiAI/rag-chatbot_test/data/char_graph"),  # From chargraph folder
            ]
            
            char_graph_path = None
            for path in possible_char_graph_paths:
                resolved_path = path.resolve() if path.is_absolute() or path.exists() else path
                # Check if parent directory exists or can be created
                if resolved_path.parent.exists() or resolved_path.parent.parent.exists():
                    char_graph_path = resolved_path
                    break
            
            if char_graph_path is None:
                # Use first path and create it
                char_graph_path = possible_char_graph_paths[0].resolve()
            
            char_graph_path.mkdir(parents=True, exist_ok=True)
            output_file = str(char_graph_path / output_base)
            print(f"Auto-generated output file: {output_file}")
        else:
            output_file = args.output_file
        
        extractor.process_text(
            input_file=input_file,
            output_file=output_file,
            previous_json_file=args.previous,
            iterations=args.iterations,
            delay=args.delay,
            plot_graph=args.plot_graph,
            desc_sentences=args.desc_sentences,
            generate_portraits=args.generate_portraits,
            copies=args.copies,
            temperature=args.temperature,
            max_main_characters=args.max_main_characters
        )

if __name__ == "__main__":
        main()
