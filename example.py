"""
Example: Build a Twitter Scraper Agent

This example demonstrates how to use the builder_agent to create 
a Twitter scraper agent from a natural language description.
"""

from builder_agent import BuilderAgent
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    # Initialize the builder agent
    builder = BuilderAgent(model_name="gpt-4o")
    
    # Define the natural language request for our agent
    request = """
    Create an agent that scrapes Twitter for tweets related to Elon Musk 
    and filters for only the positive ones using an LLM.
    
    The agent should:
    1. Take a search query (default to "Elon Musk")
    2. Retrieve recent tweets using the Twitter API
    3. Analyze each tweet with an LLM to determine sentiment
    4. Filter and return only the positive tweets
    5. Format the results nicely
    """
    
    # Create the output directory
    output_dir = "generated_agents"
    
    # Build the agent
    print("Building Twitter Scraper Agent...")
    json_path, py_path = builder.build_agent_from_request(request, output_dir)
    
    print("\nTwitter Scraper Agent built successfully!")
    print(f"JSON definition: {json_path}")
    print(f"Python code: {py_path}")
    print("\nTo run your agent:")
    print(f"python {py_path}")
    
    print("\nNote: You will need to set up Twitter API credentials to use this agent.")

if __name__ == "__main__":
    main() 