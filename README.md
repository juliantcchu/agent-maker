# Agent Maker

A tool for generating LangGraph agents from natural language descriptions.

## Overview

Agent Maker allows you to create fully functional LangGraph agents by simply describing what you want in natural language. The tool:

1. Takes your natural language request
2. Analyzes the requirements and agent architecture
3. Generates a JSON representation of the agent
4. Compiles it into executable Python code using LangGraph

No need to manually define node structures, edges, or agent workflows - just describe what you want, and Agent Maker builds it for you.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/agent-maker.git
cd agent-maker

# Install requirements
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- LangGraph
- LangChain
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)

## Usage

### Command Line Interface

```bash
python builder_agent.py "Create an agent that scrapes websites and summarizes the content"
```

Options:
- `--output-dir` or `-o`: Directory to save the generated agent files (default: current directory)
- `--model` or `-m`: LLM model to use (default: gpt-4)

### Python API

```python
from builder_agent import BuilderAgent

# Initialize the builder
builder = BuilderAgent()

# Generate an agent from a description
json_path, py_path = builder.build_agent_from_request(
    "Create an agent that answers questions using RAG with a vector database",
    output_dir="my_agents"
)

print(f"Agent created at: {py_path}")
```

## Example

See `example.py` for a complete example of building a Twitter scraper agent:

```bash
python example.py
```

This will create a Twitter scraper agent that:
1. Takes a search query (default: "Elon Musk")
2. Retrieves tweets using the Twitter API
3. Analyzes each tweet with an LLM to determine sentiment
4. Returns only the positive tweets
5. Formats the results nicely

## How It Works

Agent Maker works in three main steps:

1. **Analysis**: An LLM analyzes your natural language request to extract key requirements
2. **JSON Generation**: The requirements are transformed into a structured JSON definition
3. **Code Generation**: The JSON is compiled into Python code using the LangGraph framework

The JSON follows a specific schema with:
- **Metadata**: Name, description, version, and author
- **Nodes**: Processing components (LLMs, tools, functions, conditions)
- **Edges**: Connections between nodes defining workflow
- **State Definition**: How the agent maintains state

## Customizing Agents

After generating an agent, you can:
1. Edit the JSON file to modify the architecture
2. Edit the Python file directly to customize the implementation
3. Recompile the JSON file using `json_to_langgraph.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 