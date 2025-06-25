import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_rag_agent import LocalRAGAgent
from rag_agent import RAGAgent
from store import VectorStore
from dotenv import load_dotenv
import yaml
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def load_config():
    """Load configuration from config.yaml and .env"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        load_dotenv()
        return {
            'hf_token': config.get('HUGGING_FACE_HUB_TOKEN'),
            'openai_key': os.getenv('OPENAI_API_KEY')
        }
    except Exception as e:
        console.print(f"[red]Error loading configuration: {str(e)}")
        sys.exit(1)

def compare_responses(agent, query: str, description: str):
    """Compare standard vs CoT responses for the same query"""
    console.print(f"\n[bold cyan]Test Case: {description}")
    console.print(Panel(f"Query: {query}", style="yellow"))
    
    # Standard response
    agent.use_cot = False
    standard_response = agent.process_query(query)
    console.print(Panel(
        "[bold]Standard Response:[/bold]\n" + standard_response["answer"],
        title="Without Chain of Thought",
        style="blue"
    ))
    
    # CoT response
    agent.use_cot = True
    cot_response = agent.process_query(query)
    console.print(Panel(
        "[bold]Chain of Thought Response:[/bold]\n" + cot_response["answer"],
        title="With Chain of Thought",
        style="green"
    ))

def main():
    parser = argparse.ArgumentParser(description="Compare standard vs Chain of Thought prompting")
    parser.add_argument("--model", choices=['local', 'openai'], default='local',
                       help="Choose between local Mistral model or OpenAI")
    args = parser.parse_args()
    
    config = load_config()
    store = VectorStore(persist_directory="chroma_db")
    
    # Initialize appropriate agent
    if args.model == 'local':
        if not config['hf_token']:
            console.print("[red]Error: HuggingFace token not found in config.yaml")
            sys.exit(1)
        agent = LocalRAGAgent(store)
        model_name = "Mistral-7B"
    else:
        if not config['openai_key']:
            console.print("[red]Error: OpenAI API key not found in .env")
            sys.exit(1)
        agent = RAGAgent(store, openai_api_key=config['openai_key'])
        model_name = "GPT-4"
    
    console.print(f"\n[bold]Testing {model_name} Responses[/bold]")
    console.print("=" * 80)
    
    # Test cases that highlight CoT benefits
    test_cases = [
        {
            "query": "A train travels at 60 mph for 2.5 hours, then at 45 mph for 1.5 hours. What's the total distance covered?",
            "description": "Multi-step math problem"
        },
        {
            "query": "Compare and contrast REST and GraphQL APIs, considering their strengths and use cases.",
            "description": "Complex comparison requiring structured analysis"
        },
        {
            "query": "If a tree falls in a forest and no one is around to hear it, does it make a sound? Explain your reasoning.",
            "description": "Philosophical question requiring detailed reasoning"
        }
    ]
    
    for test_case in test_cases:
        try:
            compare_responses(agent, test_case["query"], test_case["description"])
        except Exception as e:
            console.print(f"[red]Error in test case '{test_case['description']}': {str(e)}")
    
    console.print("\n[bold green]Testing complete!")

if __name__ == "__main__":
    main() 