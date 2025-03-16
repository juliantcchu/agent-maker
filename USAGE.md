# Agent Maker Usage Guide

This guide will walk you through the process of using the Agent Maker framework to create, visualize, and run LangGraph agents.

## Prerequisites

Make sure you have all the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Step 1: Define Your Agent in JSON

First, create a JSON file that defines your agent following the schema. You can use `example_agent.json` as a starting point.

For a simple example, see `src/example_agent.json`.

## Step 2: Convert the JSON to LangGraph Code

Run the converter to transform your JSON definition into executable LangGraph code:

```bash
# Convert the example agent
python src/convert_example.py

# Or convert your own agent
python src/json_to_langgraph.py path/to/your-agent.json
```

This will generate a Python file with the LangGraph implementation of your agent.

## Step 3: Visualize the Agent Graph

To see the structure of your agent graph, use the Jupyter notebook:

```bash
# Open the Jupyter notebook
jupyter notebook src/test.ipynb
```

The notebook contains cells for loading, visualizing, and testing your agent definitions.

## Step 4: Edit the Generated Code (Optional)

You may need to edit the generated code to implement the actual functionality of your tools or functions. The generated code includes placeholder implementations for tools.

## Step 5: Run Your Agent

You can run your agent directly by executing the generated Python file:

```bash
python path/to/generated_agent.py
```

For example:
```bash
python src/example_agent.py
```

## Working with the React Visualization

The `AgentFlowDiagram.jsx` component provides a React-based visualization of your agent. To use it in a React application:

```jsx
import AgentFlowDiagram from './AgentFlowDiagram';
import agentDefinition from './example_agent.json';

function App() {
  return (
    <div className="App">
      <AgentFlowDiagram agentDefinition={agentDefinition} />
    </div>
  );
}
```

## Creating Custom Node Types

To add new node types:

1. Add the type to the `enum` in `schema.json`
2. Add a rendering component in `AgentFlowDiagram.jsx`
3. Add code generation logic in `json_to_langgraph.py`

## Tips for Debugging

- Check the visualization in the Jupyter notebook to make sure your graph structure is as expected
- Look for placeholder comments in the generated code that need to be replaced
- For complex agent behaviors, you may need to modify the generated code
- Use print statements in your node functions to debug the flow

## Example Workflow

Here's a complete example workflow:

```bash
# Navigate to the agent-maker directory
cd agent-maker

# Convert the example agent
python src/convert_example.py

# Visualize the agent in Jupyter
jupyter notebook src/test.ipynb

# Run the agent
python src/example_agent.py
```

This will convert the example agent, visualize its structure in the notebook, and run it with a sample input. 