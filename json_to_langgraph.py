"""
JSON to LangGraph Converter

This module converts a JSON agent definition to executable LangGraph code.
"""

import json
import os
import ast
from typing import Dict, List, Any, Optional
import re


class LangGraphConverter:
    """Converts JSON agent definitions to LangGraph Python code."""
    
    def __init__(self, json_path: str):
        """
        Initialize the converter with a JSON file path.
        
        Args:
            json_path: Path to the JSON file containing the agent definition
        """
        with open(json_path, 'r') as file:
            self.definition = json.load(file)
        
        self.validate_definition()
    
    def validate_definition(self):
        """Validate that the JSON definition has the required structure."""
        required_keys = ["metadata", "nodes", "edges"]
        for key in required_keys:
            if key not in self.definition:
                raise ValueError(f"Missing required key: {key}")
                
        # Check for common node/edge naming issues
        self._validate_node_edge_names()
    
    def _validate_node_edge_names(self):
        """Validate node and edge names for common errors with START/END constants."""
        edges = self.definition.get("edges", [])
        nodes = self.definition.get("nodes", [])
        
        # Get all node IDs
        node_ids = [node.get("id") for node in nodes if "id" in node]
        
        # Check for problematic node IDs
        problematic_node_ids = []
        for node_id in node_ids:
            if node_id in ["start_node", "end_node"]:
                problematic_node_ids.append(node_id)
        
        if problematic_node_ids:
            print(f"Warning: Found potentially problematic node IDs: {', '.join(problematic_node_ids)}")
            print("Use 'start' and 'end' for start/end nodes instead of 'start_node' and 'end_node'")
        
        # Check edges for references to 'start_node' and 'end_node'
        problematic_edges = []
        for i, edge in enumerate(edges):
            source = edge.get("source", "")
            target = edge.get("target", "")
            
            if source == "start_node":
                problematic_edges.append(f"edge[{i}].source = 'start_node'")
            if target == "end_node":
                problematic_edges.append(f"edge[{i}].target = 'end_node'")
                
        if problematic_edges:
            print(f"Warning: Found references to 'start_node' or 'end_node' in edges:")
            for issue in problematic_edges:
                print(f"  - {issue}")
            print("LangGraph uses START and END constants instead")
            
        # While we can fix these issues, it's good to warn about them
    
    def validate_python_syntax(self, code: str, context: str = "code") -> tuple[bool, Optional[str]]:
        """
        Validate that the generated code is syntactically valid Python.
        
        Args:
            code: The Python code to validate
            context: A description of what is being validated (for error messages)
            
        Returns:
            A tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"Syntax error in {context}: {str(e)}"
            return False, error_msg
        except Exception as e:
            error_msg = f"Error validating {context}: {str(e)}"
            return False, error_msg
    
    def generate_imports(self) -> str:
        """Generate import statements for LangGraph."""
        imports = [
            "from typing import Dict, List, Any, Tuple, Literal, TypedDict, Optional",
            "from langgraph.graph import StateGraph, START, END",
            "from langgraph.prebuilt import ToolNode",
            "import json",
            "import operator",
            "from langchain_openai import ChatOpenAI",
            "from langchain_core.messages import HumanMessage, SystemMessage, AIMessage",
        ]
        
        # Check if we need specific imports based on node types
        node_types = [node["type"] for node in self.definition["nodes"]]
        
        if "tool" in node_types:
            imports.append("from langchain_core.tools import tool")
        
        if self.definition.get("stateDefinition", {}).get("type") == "messages":
            # Updated import for MessagesState
            imports.append("from langgraph.graph.message import MessagesState")
        
        return "\n".join(imports) + "\n\n"
    
    def generate_state_class(self) -> str:
        """Generate state class definition based on stateDefinition."""
        state_def = self.definition.get("stateDefinition", {"type": "messages"})
        
        if state_def["type"] == "messages":
            return "# Using built-in MessagesState\n"
        
        # For custom state, generate a TypedDict
        state_schema = state_def.get("schema", {})
        class_lines = ["class AgentState(TypedDict):", "    \"\"\"State object for the agent.\"\"\""]
        
        for key, value_type in state_schema.items():
            # Convert 'string' to 'str' for Python type annotations
            if value_type == "string":
                value_type = "str"
            
            # Replace hyphens with underscores in keys
            safe_key = key.replace('-', '_')
            class_lines.append(f"    {safe_key}: {value_type}")
        
        return "\n".join(class_lines) + "\n\n"
    
    def generate_node_functions(self) -> str:
        """Generate Python functions for each node in the graph."""
        functions = []
        
        for node in self.definition["nodes"]:
            node_id = node["id"]
            node_type = node["type"]
            
            # Skip start and end nodes
            if node_type in ["start", "end"]:
                continue
            
            # Generate function based on node type
            if node_type == "llm":
                functions.append(self._generate_llm_function(node))
            elif node_type == "tool":
                functions.append(self._generate_tool_function(node))
            elif node_type == "function":
                functions.append(self._generate_custom_function(node))
            elif node_type == "condition":
                functions.append(self._generate_condition_function(node))
            
        return "\n\n".join(functions) + "\n\n"
    
    def _generate_llm_function(self, node: Dict[str, Any]) -> str:
        """Generate a function for an LLM node."""
        node_id = node["id"]
        data = node["data"]
        model = data.get("model", "gpt-4")
        prompt = data.get("prompt", "")
        additional_context = data.get("additional_context", None)
        
        # Format the prompt with template variables
        formatted_prompt = self._format_template_string(prompt)
        
        # Create a safe key for the state return value (replace hyphens with underscores)
        safe_key = node_id.replace('-', '_')
        
        # If there's an additional_context field, make sure it's accessed in the prompt
        if additional_context:
            # Extract variable name without 'state[' and ']' if present
            context_var = additional_context
            if context_var.startswith("state['") and context_var.endswith("']"):
                context_var = context_var[7:-2]
            elif context_var.startswith("state.get('") and context_var.endswith("')"):
                context_var = context_var[11:-2]
            elif context_var.startswith("state."):
                context_var = context_var[6:]
                
            # Add code to ensure the context variable is properly referenced
            context_access = f"""
    # Get context data from state
    context_data = state.get('{context_var}', '')
    # Replace context variable in the prompt if needed
    formatted_prompt = formatted_prompt.replace('{{{context_var}}}', str(context_data))"""
        else:
            context_access = ""
            
        function_str = f"""def {node_id.replace('-', '_')}(state):
    \"\"\"
    {data.get('description', 'LLM node function')}
    \"\"\"
    # Initialize the LLM
    llm = ChatOpenAI(model="{model}")
    
    # Format the prompt
    formatted_prompt = f'''{formatted_prompt}'''{context_access}
    
    # Call the LLM
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=formatted_prompt)
    ]
    response = llm.invoke(messages)
    
    # Return the updated state
    return {{"{safe_key}": response.content}}"""
        
        # Validate the generated code
        is_valid, error = self.validate_python_syntax(function_str, f"LLM node '{node_id}'")
        if not is_valid:
            raise ValueError(error)
            
        return function_str
    
    def _generate_tool_function(self, node: Dict[str, Any]) -> str:
        """Generate a function for a tool node."""
        node_id = node["id"]
        data = node["data"]
        tool_def = data.get("toolDefinition", {})
        
        # Convert return type from "string" to "str" if needed
        return_type = tool_def.get('returnType', 'str')
        if return_type == "string":
            return_type = "str"
        
        # Get all parameters for the tool
        parameters = tool_def.get('parameters', {})
        param_names = list(parameters.keys())
        
        # Generate tool function definition
        tool_function = f"""@tool
def {tool_def.get('name', node_id.replace('-', '_'))}({', '.join(param_names)}) -> {return_type}:
    \"\"\"
    {tool_def.get('description', 'Tool function')}
    \"\"\"
    # Implement the tool logic here
    # This is a placeholder that will need to be replaced with actual implementation
    pass"""
        
        # Generate parameter mapping for tool call
        param_mapping = []
        for param in param_names:
            # Map each parameter to corresponding state value
            # Use the parameter name as the state key
            safe_param = param.replace('-', '_')
            param_mapping.append(f"{param}=state.get('{safe_param}', None)")
        
        # Generate the node function
        node_function = f"""
def {node_id.replace('-', '_')}(state):
    \"\"\"
    {data.get('description', 'Tool node function')}
    \"\"\"
    # Call the tool function
    result = {tool_def.get('name', node_id.replace('-', '_'))}(
        {', '.join(param_mapping)}
    )
    
    # Return the updated state
    return {{"{node_id}": result}}"""
        
        # Combine tool and node functions
        function_str = tool_function + node_function
        
        # Validate the generated code
        is_valid, error = self.validate_python_syntax(function_str, f"tool node '{node_id}'")
        if not is_valid:
            raise ValueError(error)
            
        return function_str
    
    def _generate_custom_function(self, node: Dict[str, Any]) -> str:
        """Generate a function for a custom function node."""
        node_id = node["id"]
        data = node["data"]
        code = data.get("code", "# Custom code goes here")
        
        # Check if code has a return statement, if not add a default one
        if "return" not in code:
            code += "\n# Adding default return since none was provided\nreturn {\"result\": \"placeholder\"}"
        
        # Check indentation - all lines should be properly indented or empty
        normalized_code = ""
        for line in code.split("\n"):
            if line.strip() and not line.startswith("    "):
                # Add indentation if line isn't empty and doesn't have it
                normalized_code += "    " + line + "\n"
            else:
                normalized_code += line + "\n"
                
        # Use the normalized code
        code = normalized_code.strip()
        
        function_str = f"""def {node_id.replace('-', '_')}(state):
    \"\"\"
    {data.get('description', 'Custom function')}
    \"\"\"
    # Custom function logic
{self._indent_code(code, 4)}
"""
        
        # Validate the generated code
        is_valid, error = self.validate_python_syntax(function_str, f"custom node '{node_id}'")
        if not is_valid:
            raise ValueError(error)
            
        return function_str
    
    def _generate_condition_function(self, node: Dict[str, Any]) -> str:
        """Generate a function for a condition node."""
        node_id = node["id"]
        data = node["data"]
        condition_logic = data.get("conditionLogic", "return 'default'")
        
        # Replace hyphens with underscores in variable access in condition logic
        # Simple replacement for common patterns - more complex logic might need parsing
        condition_logic = condition_logic.replace("state['", "state['").replace("-", "_")
        
        # Ensure condition logic returns a string for routing
        # If it doesn't start with return, add it
        if not condition_logic.strip().startswith("return "):
            condition_logic = f"return {condition_logic}"
        
        # Check if the return value is wrapped in quotes, if not, ensure it returns a string
        # Looking for patterns like: return xyz vs return 'xyz' or return "xyz"
        if "return" in condition_logic and not re.search(r"return\s+['\"]", condition_logic):
            # If it's not returning a string literal, wrap it with str()
            if not re.search(r"return\s+str\(", condition_logic):
                # Replace simple 'return x' with 'return str(x)'
                condition_logic = re.sub(r"return\s+([^'\"]\S*)", r"return str(\1)", condition_logic)
        
        # Check for nested functions, conditionals, or loops which might cause indentation issues
        has_complex_structures = any(keyword in condition_logic for keyword in ["def ", "if ", "for ", "while "])
        
        # If we have complex structures that could cause indentation issues, simplify
        if has_complex_structures:
            print(f"Warning: Simplifying complex condition node '{node_id}' to avoid indentation issues")
            # Use a very simple, safe default
            function_str = f"""def {node_id.replace('-', '_')}(state):
    \"\"\"
    {data.get('description', 'Condition function')}
    \"\"\"
    # Simplified condition logic to avoid indentation issues
    return "default"  # Default routing value"""
        else:
            # For simple return statements, use a direct return instead of nested function
            if condition_logic.strip().startswith("return "):
                function_str = f"""def {node_id.replace('-', '_')}(state):
    \"\"\"
    {data.get('description', 'Condition function')}
    \"\"\"
    # Condition logic
    {condition_logic}"""
            else:
                # For more complex but not nested logic, use the nested function approach
                function_str = f"""def {node_id.replace('-', '_')}(state):
    \"\"\"
    {data.get('description', 'Condition function')}
    \"\"\"
    # Condition logic
    def evaluate_condition(state):
        {condition_logic}
    
    return evaluate_condition(state)"""
        
        # Validate the generated code
        is_valid, error = self.validate_python_syntax(function_str, f"condition node '{node_id}'")
        if not is_valid:
            print(f"Warning: Invalid condition node '{node_id}'. Error: {error}")
            print("Applying fallback implementation")
            # Fallback to an ultra-simple implementation that's guaranteed to work
            function_str = f"""def {node_id.replace('-', '_')}(state):
    \"\"\"
    {data.get('description', 'Condition function - fallback implementation')}
    \"\"\"
    # Fallback implementation to ensure valid syntax
    return "default"  # Default routing value"""
            
        return function_str
    
    def _indent_code(self, code: str, indent_level: int) -> str:
        """Indent each line of code by a specified number of spaces."""
        indent = " " * indent_level
        return "\n".join(indent + line for line in code.split("\n"))
    
    def _format_template_string(self, template: str) -> str:
        """Format a template string to use Python's f-string syntax."""
        if not template:
            return ""
            
        # Handle two types of template variables:
        # 1. Handlebars style: {{variable}}
        # 2. Python-style curly braces: {variable}
        
        # First, handle handlebars-style variables {{variable}}
        matches = re.findall(r'\{\{(\w+)\}\}', template)
        for var in matches:
            # Replace hyphens with underscores for state access
            safe_var = var.replace('-', '_')
            template = template.replace(f'{{{{{var}}}}}', f"{{state.get('{safe_var}', '')}}")
        
        # Then handle Python-style {variable} 
        # We need to be careful not to replace escaped braces or f-string syntax
        simple_var_matches = re.findall(r'(?<!\{)\{(?!\{)(\w+)(?!\})\}(?!\})', template)
        for var in simple_var_matches:
            # Replace hyphens with underscores for state access
            safe_var = var.replace('-', '_')
            # Replace {variable} with {state.get('variable', '')}
            template = re.sub(r'(?<!\{)\{' + var + r'\}(?!\})', f"{{state.get('{safe_var}', '')}}", template)
        
        return template
    
    def generate_graph_construction(self) -> str:
        """Generate code to construct the LangGraph graph."""
        # Determine the state class to use
        state_def = self.definition.get("stateDefinition", {"type": "messages"})
        state_class = "MessagesState" if state_def["type"] == "messages" else "AgentState"
        
        # Collect node IDs and edge information
        nodes = self.definition["nodes"]
        edges = self.definition["edges"]
        
        # Start the graph construction code
        construction_lines = [
            "def create_agent_graph():",
            "    \"\"\"Create and return the agent's StateGraph.\"\"\"",
            f"    # Initialize the graph",
            f"    graph = StateGraph({state_class})",
            "",
            "    # Add nodes to the graph"
        ]
        
        # Add nodes to the graph
        for node in nodes:
            node_id = node["id"]
            node_type = node["type"]
            
            # Skip adding the start and end nodes
            if node_type in ["start", "end"]:
                continue
                
            construction_lines.append(f"    graph.add_node(\"{node_id}\", {node_id.replace('-', '_')})")
        
        construction_lines.append("")
        construction_lines.append("    # Add edges to the graph")
        
        # Add regular edges (without conditions)
        regular_edges = []
        conditional_edges = {}
        
        # Group edges by source
        edge_groups = {}
        for edge in edges:
            source = edge["source"]
            if source not in edge_groups:
                edge_groups[source] = []
            edge_groups[source].append(edge)
        
        # Process regular edges
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            condition = edge.get("condition", None)
            
            # Regular edge (no condition)
            if condition is None:
                # Handle START/END constant usage
                if source in ["start", "start_node", "START"]:
                    # Use the actual START constant, not a string
                    source_code = "START"
                else:
                    # Use string representation for other nodes
                    source_code = f"\"{source}\""
                    
                if target in ["end", "end_node", "END"]:
                    # Use the actual END constant, not a string
                    target_code = "END"
                else:
                    # Use string representation for other nodes
                    target_code = f"\"{target}\""
                    
                # Add the edge with proper constant/string usage
                construction_lines.append(f"    graph.add_edge({source_code}, {target_code})")
        
        # Process conditional edges grouped by source
        for source, source_edges in edge_groups.items():
            # Skip if there are no conditional edges from this source
            if not any("condition" in edge for edge in source_edges):
                continue
                
            # Skip start node variants
            if source == "start" or source == "start_node" or source == "START":
                continue
            
            # Add conditional edges mapping
            construction_lines.append("")
            construction_lines.append(f"    # Add conditional edges from {source}")
            construction_lines.append(f"    graph.add_conditional_edges(")
            construction_lines.append(f"        \"{source}\",")
            construction_lines.append(f"        # This should reflect the actual condition logic")
            construction_lines.append(f"        lambda x: x,  # Convert result to edge label")
            construction_lines.append(f"        {{")
            
            # Add each condition to target mapping
            for edge in source_edges:
                condition = edge.get("condition")
                target = edge["target"]
                
                if condition is not None:
                    # Handle END constant for conditional edges
                    if target in ["end", "end_node", "END"]:
                        # Use the actual END constant
                        target_code = "END"
                        construction_lines.append(f"            \"{condition}\": {target_code},")
                    else:
                        # Use string for regular node targets
                        construction_lines.append(f"            \"{condition}\": \"{target}\",")
            
            construction_lines.append(f"        }}")
            construction_lines.append(f"    )")
        
        # Complete the function
        construction_lines.append("")
        construction_lines.append("    # Compile the graph")
        construction_lines.append("    return graph.compile()")
        
        return "\n".join(construction_lines) + "\n"
    
    def generate_main_function(self) -> str:
        """Generate a main function to create and run the agent."""
        # Determine state type for initialization
        state_def = self.definition.get("stateDefinition", {"type": "messages"})
        
        # Customize the main function based on the state type
        if state_def["type"] == "messages":
            main_code = [
                "def main():",
                "    \"\"\"Create and run the agent.\"\"\"",
                "    # Create the agent graph",
                "    agent = create_agent_graph()",
                "",
                "    # Initialize the state with messages",
                "    initial_state = {\"messages\": [HumanMessage(content=\"What is the weather today?\")]}",
            ]
        else:
            # For custom state, initialize with schema fields
            state_schema = state_def.get("schema", {})
            state_init_parts = []
            
            # Generate initialization for each state field
            for key in state_schema:
                # Initialize based on type (empty lists, dicts, etc.)
                state_init_parts.append(f'"{key}": None')
            
            state_init = ", ".join(state_init_parts)
            
            main_code = [
                "def main():",
                "    \"\"\"Create and run the agent.\"\"\"",
                "    # Create the agent graph",
                "    agent = create_agent_graph()",
                "",
                "    # Initialize the state with required fields",
                f"    initial_state = {{{state_init}}}" if state_init else "    initial_state = {}",
            ]
        
        # Add code to invoke the agent
        main_code.extend([
            "",
            "    # Invoke the agent",
            "    result = agent.invoke(initial_state)",
            "",
            "    # Print the result",
            "    print(json.dumps(result, indent=2))",
            "",
            "    return result",
            "",
            "",
            "if __name__ == \"__main__\":",
            "    main()",
            ""
        ])
        
        return "\n".join(main_code)
    
    def generate_langgraph_code(self) -> str:
        """Generate the complete LangGraph code."""
        try:
            code_sections = [
                self.generate_imports(),
                self.generate_state_class(),
                self.generate_node_functions(),
                self.generate_graph_construction(),
                self.generate_main_function()
            ]
            
            # Add a header comment with metadata
            header = [
                "\"\"\"",
                f"LangGraph Agent: {self.definition['metadata']['name']}",
                "",
                f"Description: {self.definition['metadata']['description']}",
                f"Version: {self.definition['metadata']['version']}",
                f"Author: {self.definition['metadata'].get('author', 'Unknown')}",
                "",
                "This code was automatically generated from a JSON agent definition.",
                "\"\"\""
            ]
            
            complete_code = "\n".join(header) + "\n\n" + "\n".join(code_sections)
            
            # Validate the final complete code
            is_valid, error = self.validate_python_syntax(complete_code, "complete generated code")
            if not is_valid:
                raise ValueError(error)
                
            return complete_code
        except Exception as e:
            if isinstance(e, ValueError):
                # Re-raise ValueError directly
                raise
            else:
                # Wrap other exceptions with more context
                raise ValueError(f"Error generating LangGraph code: {str(e)}")
    
    def save_langgraph_code(self, output_path: str):
        """
        Save the generated LangGraph code to a Python file.
        
        Args:
            output_path: Path where to save the generated code
        """
        try:
            # Generate the code
            code = self.generate_langgraph_code()
            
            # Save to file
            with open(output_path, 'w') as file:
                file.write(code)
            
            print(f"LangGraph code generated and saved to {output_path}")
        except ValueError as e:
            # Propagate the validation error message
            raise ValueError(f"Failed to generate valid Python code: {str(e)}")


def convert_json_to_langgraph(json_path: str, output_path: str = None, validate_only: bool = False):
    """
    Convert a JSON agent definition to LangGraph Python code.
    
    Args:
        json_path: Path to the JSON file containing the agent definition
        output_path: Path where to save the generated code. Required unless validate_only=True.
        validate_only: If True, only validates the syntax without generating the file.
        
    Returns:
        If validate_only=True, returns a tuple (is_valid, error_message).
        Otherwise, returns None.
        
    Raises:
        ValueError: If validation fails during code generation/saving
    """
    try:
        # First validate that the JSON file exists and is valid JSON
        if not os.path.exists(json_path):
            error_msg = f"JSON file not found: {json_path}"
            if validate_only:
                return False, error_msg
            else:
                raise ValueError(error_msg)
        
        # Check if file contains valid JSON
        try:
            with open(json_path, 'r') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON file: {str(e)}"
            if validate_only:
                return False, error_msg
            else:
                raise ValueError(error_msg)
        
        # Initialize the converter and proceed with validation/generation
        converter = LangGraphConverter(json_path)
        
        if validate_only:
            # Generate the code to validate it, but don't save
            try:
                converter.generate_langgraph_code()
                return True, None
            except ValueError as e:
                return False, str(e)
        else:
            if output_path is None:
                raise ValueError("output_path is required when validate_only=False")
            converter.save_langgraph_code(output_path)
            return None
    except Exception as e:
        if validate_only:
            return False, str(e)
        else:
            raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert a JSON agent definition to LangGraph code")
    parser.add_argument("json_path", help="Path to the JSON file containing the agent definition")
    parser.add_argument("--output", "-o", help="Path where to save the generated code", default=None)
    
    args = parser.parse_args()
    
    # If output path is not specified, use the same name as the input file but with .py extension
    output_path = args.output
    if output_path is None:
        output_path = os.path.splitext(args.json_path)[0] + ".py"
    
    convert_json_to_langgraph(args.json_path, output_path) 