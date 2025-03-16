#!/usr/bin/env python
"""
Example script to convert an agent definition JSON to LangGraph code.
"""

import os
import sys
from pathlib import Path

# Make sure the script can find the json_to_langgraph module
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from json_to_langgraph import convert_json_to_langgraph

def main():
    """Convert the example agent to LangGraph code."""
    # Get paths relative to the script
    current_dir = Path(__file__).parent
    input_path = current_dir / "example_agent.json"
    output_path = current_dir / "example_agent.py"
    
    # Convert the agent
    try:
        convert_json_to_langgraph(input_path, output_path)
        print(f"Successfully converted {input_path} to {output_path}")
    except Exception as e:
        print(f"Error converting agent: {e}")

if __name__ == "__main__":
    main() 