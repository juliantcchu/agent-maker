"""
Builder Agent using LangGraph

This script creates an agent that converts natural language requests into functional LangGraph agents.
The agent itself is implemented as a LangGraph agent.
"""

import json
import os
import re
import ast
import tempfile
import importlib.util
from typing import Dict, Any, List, TypedDict, Literal, Optional, Annotated
import argparse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

# Import langgraph components
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

# Import the converter
from json_to_langgraph import convert_json_to_langgraph

# Import visualization utilities
from utils import visualize_agent, save_agent_visualization

# Define the state schema
class BuilderState(TypedDict):
    """State for the builder agent."""
    request: str
    requirements: Optional[Dict[str, Any]]
    agent_json: Optional[Dict[str, Any]]
    is_valid: Optional[bool]
    json_path: Optional[str]
    py_path: Optional[str]
    viz_path: Optional[str]  # Path to visualization file
    output_dir: str
    error: Optional[str]
    model_name: str
    debug: Optional[bool]  # Debug flag

def debug_print(state: BuilderState, node_name: str) -> None:
    """Print debug information if debug flag is enabled."""
    if not state.get('debug', False):
        return
    
    print(f"\n==== DEBUG: {node_name} ====")
    
    # Print key state fields (avoiding large JSON dumps)
    for key, value in state.items():
        if key == 'agent_json' and value:
            print(f"  {key}: <JSON object with {len(json.dumps(value))} chars>")
        elif key == 'requirements' and value:
            print(f"  {key}: <Dictionary with {len(value)} items>")
        elif key == 'error' and value:
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * (len(node_name) + 14))

def analyze_request(state: BuilderState) -> Dict[str, Any]:
    """
    Analyze the natural language request and extract structured requirements.
    """
    debug_print(state, "analyze_request - START")
    print("Analyzing request...")
    
    request = state['request']
    model_name = state.get('model_name', 'o3-mini')
    
    llm = ChatOpenAI(model=model_name)
    
    system_prompt = """
    You are an expert at understanding requirements for AI agents.
    Given a natural language request, extract the key requirements for an agent:
    
    1. Determine the agent's primary purpose and goal
    2. Identify required capabilities and tools
    3. Determine data input/output requirements
    4. Note any constraints or special behaviors
    
    Provide the requirements in a structured JSON format.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Request: {request}\n\nExtract the agent requirements as JSON.")
    ]
    
    response = llm.invoke(messages)
    
    # Try to parse the response as JSON
    try:
        extracted_requirements = json.loads(response.content)
    except json.JSONDecodeError:
        # Extract JSON from code blocks if present
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response.content)
        if json_match:
            try:
                extracted_requirements = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                return {"error": "Could not extract requirements from response."}
        else:
            try:
                # Last attempt: try to extract anything that looks like a dict
                dict_match = re.search(r'\{[\s\S]*\}', response.content)
                if dict_match:
                    extracted_requirements = json.loads(dict_match.group(0))
                else:
                    return {"error": "Could not extract requirements from response."}
            except json.JSONDecodeError:
                return {"error": "Could not extract requirements from response."}
    
    result = {"requirements": extracted_requirements}
    debug_print({**state, **result}, "analyze_request - END")
    return result

def generate_agent_json(state: BuilderState) -> Dict[str, Any]:
    """
    Generate a JSON definition for the agent based on requirements.
    """
    debug_print(state, "generate_agent_json - START")
    print("Generating agent JSON definition...")
    
    requirements = state['requirements']
    llm = ChatOpenAI(model=state.get('model_name', 'gpt-4'))
    
    system_prompt = """
    You are an expert LangGraph agent designer. You need to create a JSON definition
    for an agent based on the given requirements.
    
    The JSON must follow this schema:
    
    {
        "metadata": {
            "name": string,
            "description": string,
            "version": string,
            "author": string
        },
        "nodes": [
            {
                "id": string,
                "type": string (one of: "start", "end", "llm", "tool", "function", "condition"),
                "data": {
                    // Depends on node type
                }
            }
        ],
        "edges": [
            {
                "source": string (node id),
                "target": string (node id),
                "condition": string (optional)
            }
        ],
        "stateDefinition": {
            "type": string (e.g., "messages"),
            "schema": {
                // Optional key-value pairs defining state schema
            }
        }
    }
    
    IMPORTANT:
    1. For all returnType fields, use ONLY valid Python types:
       - For single values: "str", "int", "float", "bool", "dict", etc.
       - For collections: "list", "dict", "set", or with type parameters like "List[str]", "Dict[str, int]", etc.
       - DO NOT use natural language descriptions like "list of tweets" - use "List[str]" or similar
    
    2. For condition nodes, the conditionLogic MUST contain valid Python code that returns a string, for example:
       - "return 'path_a'" or "return 'path_b'"
       - "return 'success' if state['result'] else 'failure'"
       - Make sure to include a return statement and proper Python syntax
    
    3. For function nodes, the code MUST be valid Python code and should include:
       - Proper indentation 
       - Valid Python syntax
       - Return a dictionary that will be merged with the state
    
    Create a complete, valid JSON that would work with LangGraph.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Requirements: {json.dumps(requirements, indent=2)}\n\nGenerate a complete LangGraph agent JSON definition based on these requirements.")
    ]
    
    response = llm.invoke(messages)
    
    # Try to parse the response as JSON
    try:
        agent_json = json.loads(response.content)
    except json.JSONDecodeError:
        # Extract JSON from code blocks if present
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response.content)
        if json_match:
            try:
                agent_json = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                return {"error": "Generated JSON is invalid. The LLM output could not be parsed.", "is_valid": False}
        else:
            return {"error": "Generated JSON is invalid. The LLM output could not be parsed.", "is_valid": False}
    
    # Post-process the JSON to ensure valid Python types and code
    agent_json = fix_return_types(agent_json)
    
    result = {"agent_json": agent_json}
    debug_print({**state, **result}, "generate_agent_json - END")
    return result

def validate_python_syntax(code: str, context: str = "code") -> tuple[bool, str]:
    """
    Validate that a string contains syntactically valid Python code.
    
    Args:
        code: The Python code to validate
        context: A description of what is being validated (for error messages)
    
    Returns:
        A tuple of (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        error_msg = f"Syntax error in {context}: {str(e)}"
        return False, error_msg
    except Exception as e:
        error_msg = f"Error validating {context}: {str(e)}"
        return False, error_msg

def validate_json(state: BuilderState) -> Dict[str, Any]:
    """
    Validate that the generated JSON follows the required schema and
    contains valid Python code where applicable.
    """
    debug_print(state, "validate_json - START")
    print("Validating agent JSON...")
    
    agent_json = state['agent_json']
    output_dir = state.get('output_dir', '.')
    
    # Set default
    is_valid = True
    validation_errors = []
    temp_py_path = None
    temp_viz_path = None
    
    # Part 1: Validate JSON structure
    
    # Check required top-level keys
    required_keys = ["metadata", "nodes", "edges"]
    for key in required_keys:
        if key not in agent_json:
            validation_errors.append(f"Missing required key: {key}")
            is_valid = False
    
    # Check metadata keys
    required_metadata = ["name", "description", "version"]
    if "metadata" in agent_json:
        for key in required_metadata:
            if key not in agent_json["metadata"]:
                validation_errors.append(f"Missing required metadata key: {key}")
                is_valid = False
    
    # Check for valid nodes array
    if "nodes" not in agent_json or not isinstance(agent_json["nodes"], list):
        validation_errors.append("Missing or invalid 'nodes' array")
        is_valid = False
    else:
        # Check that there is a start and end node
        node_ids = [node["id"] for node in agent_json["nodes"]]
        node_types = [node["type"] for node in agent_json["nodes"]]
        
        if "start" not in node_types:
            validation_errors.append("Missing start node")
            is_valid = False
        
        if "end" not in node_types:
            validation_errors.append("Missing end node")
            is_valid = False
        
        # Check for duplicate node IDs
        duplicates = set([x for x in node_ids if node_ids.count(x) > 1])
        if duplicates:
            validation_errors.append(f"Duplicate node IDs found: {', '.join(duplicates)}")
            is_valid = False
    
    # Check for valid edges array
    if "edges" not in agent_json or not isinstance(agent_json["edges"], list):
        validation_errors.append("Missing or invalid 'edges' array")
        is_valid = False
    elif "nodes" in agent_json and isinstance(agent_json["nodes"], list):
        # Check that all edges reference valid nodes
        node_ids = [node["id"] for node in agent_json["nodes"]]
        
        # Add START and END to valid node IDs to check against
        valid_node_ids = node_ids + ["START", "END"]
        
        # Check for incorrect start/end node references
        common_node_name_errors = {
            "start_node": "START",
            "start": "START",
            "end_node": "END",
            "end": "END"
        }
        
        for edge in agent_json["edges"]:
            if "source" not in edge or "target" not in edge:
                validation_errors.append(f"Edge missing source or target: {edge}")
                is_valid = False
                continue
            
            # Check for common incorrect node names
            for incorrect, correct in common_node_name_errors.items():
                if edge["source"] == incorrect:
                    validation_errors.append(f"Edge uses '{incorrect}' instead of '{correct}' constant: {edge}. Please replace with uppercase constant.")
                    is_valid = False
                
                if edge["target"] == incorrect:
                    validation_errors.append(f"Edge uses '{incorrect}' instead of '{correct}' constant: {edge}. Please replace with uppercase constant.")
                    is_valid = False
            
            if edge["source"] not in valid_node_ids:
                validation_errors.append(f"Edge references non-existent source node: {edge['source']}")
                is_valid = False
            
            if edge["target"] not in valid_node_ids:
                validation_errors.append(f"Edge references non-existent target node: {edge['target']}")
                is_valid = False
    
    # Part 2: Validate Python code in nodes
    if "nodes" in agent_json and isinstance(agent_json["nodes"], list):
        for node in agent_json["nodes"]:
            node_id = node.get("id", "unknown")
            node_type = node.get("type", "unknown")
            
            # Skip nodes without data
            if "data" not in node:
                continue
            
            # Validate condition logic (Python code)
            if node_type == "condition" and "data" in node:
                # Validate condition logic
                condition_logic = node["data"].get("conditionLogic", "")
                if condition_logic:
                    # Wrap the condition logic in a function to validate its syntax
                    test_code = f"def test_condition(state):\n    {condition_logic}\n"
                    is_syntax_valid, error = validate_python_syntax(test_code, f"condition node '{node_id}'")
                    if not is_syntax_valid:
                        validation_errors.append(error)
                        is_valid = False
                    
                    # Check that it contains a return statement
                    if "return" not in condition_logic:
                        validation_errors.append(f"Condition node '{node_id}' missing return statement")
                        is_valid = False
            
            # Validate function code (Python code)
            elif node_type == "function" and "data" in node:
                # Validate custom function code
                code = node["data"].get("code", "")
                if code:
                    is_syntax_valid, error = validate_python_syntax(code, f"function node '{node_id}'")
                    if not is_syntax_valid:
                        validation_errors.append(error)
                        is_valid = False
                    
                    # Check that it contains a return statement
                    if "return" not in code:
                        validation_errors.append(f"Function node '{node_id}' missing return statement")
                        is_valid = False
            
            # Validate tool return types
            elif node_type == "tool" and "data" in node and "toolDefinition" in node["data"]:
                tool_def = node["data"]["toolDefinition"]
                if "returnType" in tool_def:
                    return_type = tool_def["returnType"]
                    if not is_valid_python_type(return_type):
                        validation_errors.append(f"Invalid Python return type in tool node '{node_id}': {return_type}")
                        is_valid = False
    
    # Part 3: Validate whether it can compile to LangGraph code
    if is_valid:
        # Apply direct fix for edge constants - fail-safe mechanism
        edge_fixes_applied = False
        if "edges" in agent_json:
            for edge in agent_json["edges"]:
                if "source" in edge:
                    if edge["source"] in ["start", "start_node"]:
                        edge["source"] = "START"
                        edge_fixes_applied = True
                if "target" in edge:
                    if edge["target"] in ["end", "end_node"]:
                        edge["target"] = "END"
                        edge_fixes_applied = True
        
        if edge_fixes_applied:
            print("Applied automatic fixes to edge constants (START/END)")
            
        # Save JSON to a temporary file for validation
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_json:
            with open(temp_json.name, 'w') as f:
                json.dump(agent_json, f)
            
            # Try to validate the Python code generation
            code_is_valid, error_msg = convert_json_to_langgraph(temp_json.name, validate_only=True)
            
            if not code_is_valid:
                validation_errors.append(f"Python code generation validation failed: {error_msg}")
                is_valid = False
            else:
                # If validation passes, try to compile and visualize the agent
                try:
                    # Generate temporary Python file
                    agent_name = agent_json["metadata"]["name"].lower().replace(" ", "_")
                    temp_py_path = os.path.join(tempfile.gettempdir(), f"{agent_name}.py")
                    convert_json_to_langgraph(temp_json.name, temp_py_path)
                    
                    # Load the module and create the agent
                    spec = importlib.util.spec_from_file_location(agent_name, temp_py_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Try to create the agent
                        if hasattr(module, 'create_agent_graph'):
                            agent = module.create_agent_graph()
                            
                            # Create visualization of the agent
                            temp_viz_path = os.path.join(tempfile.gettempdir(), f"{agent_name}_viz.html")
                            try:
                                save_agent_visualization(
                                    agent, 
                                    temp_viz_path, 
                                    format="html"
                                )
                                print(f"Generated visualization for validation: {temp_viz_path}")
                            except Exception as e:
                                validation_errors.append(f"Failed to generate visualization: {str(e)}")
                                # This is not a critical error, so we don't set is_valid to False
                        else:
                            validation_errors.append("Generated module doesn't have a create_agent_graph function")
                            is_valid = False
                    else:
                        validation_errors.append("Failed to load generated module")
                        is_valid = False
                except Exception as e:
                    validation_errors.append(f"Error during agent compilation: {str(e)}")
                    is_valid = False
            
            # Clean up
            try:
                os.unlink(temp_json.name)
            except:
                pass
    
    print(f"Validation {'passed' if is_valid else 'failed'}")
    if not is_valid:
        print("\n".join(validation_errors))
        # Clean up temporary files if validation failed
        if temp_py_path and os.path.exists(temp_py_path):
            try:
                os.unlink(temp_py_path)
            except:
                pass
        if temp_viz_path and os.path.exists(temp_viz_path):
            try:
                os.unlink(temp_viz_path)
            except:
                pass
    
    result = {
        "is_valid": is_valid,
        "error": "\n".join(validation_errors) if validation_errors else None
    }
    
    # If validation passed, add temporary file paths to result
    if is_valid:
        result["py_path"] = temp_py_path
        result["viz_path"] = temp_viz_path
    
    debug_print({**state, **result}, "validate_json - END")
    return result

def repair_json(state: BuilderState) -> Dict[str, Any]:
    """
    Repair the JSON based on validation errors.
    """
    debug_print(state, "repair_json - START")
    print("Attempting to repair agent JSON...")
    
    error = state.get('error', '')
    agent_json = state['agent_json']
    model_name = state['model_name']
    
    llm = ChatOpenAI(model=model_name)
    
    # Determine the type of error to guide the repair
    is_json_structure_error = any([
        "Missing" in error,
        "invalid" in error.lower(),
        "Duplicate" in error,
        "references non-existent" in error
    ])
    
    is_python_error = any([
        "Syntax error" in error,
        "Error validating" in error,
        "Python code generation validation failed" in error,
        "missing return statement" in error
    ])
    
    # Create appropriate system message based on error type
    if is_python_error:
        system_message = """You are an expert at fixing Python code in LangGraph agents.
You need to fix invalid Python code within the agent JSON definition.

Focus especially on these common issues in node definitions:
1. Condition nodes: Fix the "conditionLogic" field which should contain valid Python code that returns a string.
2. Function nodes: Fix the "code" field which should contain valid Python code.
3. Make sure indentation is consistent.
4. Check for missing return statements, unclosed brackets/parentheses/quotes.
5. Ensure the Python syntax is valid when the code is used in the context of a function.

Return the COMPLETE fixed JSON with all nodes, not just the fixed parts.
Do not remove any nodes from the JSON, only fix the code within them."""
    elif is_json_structure_error:
        system_message = """You are an expert at fixing LangGraph agent JSON definitions.
You need to fix structural issues in the agent JSON.

Focus especially on these common issues:
1. Missing required keys (metadata, nodes, edges)
2. Missing metadata fields (name, description, version)
3. Missing start or end nodes
4. Duplicate node IDs
5. Edges referencing non-existent nodes
6. Make sure all required fields are present in each node
7. VERY IMPORTANT: For edge connections, use uppercase 'START' and 'END' constants:
   - Use 'START' (not 'start' or 'start_node') for connections from the start node
   - Use 'END' (not 'end' or 'end_node') for connections to the end node

Return the COMPLETE fixed JSON with all required elements."""
    else:
        # General case - both issues might be present
        system_message = """You are an expert at fixing LangGraph agent JSON definitions.
Fix the problematic agent JSON based on the validation errors.

This could involve:
1. Fixing the JSON structure (missing keys, invalid relationships)
2. Fixing Python code within condition and function nodes
3. Ensuring proper connectivity between nodes
4. EXTREMELY IMPORTANT: Use uppercase 'START' and 'END' constants for edges:
   - Change any 'start' or 'start_node' sources to 'START'
   - Change any 'end' or 'end_node' targets to 'END'

Return the COMPLETE fixed JSON, not just the fixed parts."""
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"""Here is the agent JSON that needs repair:

```json
{json.dumps(agent_json, indent=2)}
```

Error message:
{error}

Please fix the JSON and return only the complete fixed JSON.""")
    ]
    
    response = llm.invoke(messages)
    
    # Extract JSON from the response
    try:
        pattern = r"```json\s*([\s\S]*?)\s*```"
        match = re.search(pattern, response.content)
        if match:
            fixed_json_str = match.group(1)
            try:
                fixed_json = json.loads(fixed_json_str)
            except json.JSONDecodeError as e:
                # If the fixed JSON is still invalid JSON syntax, try a direct repair
                result = {
                    "is_valid": False,
                    "error": f"Repair attempt produced invalid JSON: {str(e)}",
                    "agent_json": agent_json  # Keep the original for next repair attempt
                }
                debug_print({**state, **result}, "repair_json - END (failed)")
                return result
        else:
            # Try to load the entire response as JSON
            try:
                fixed_json = json.loads(response.content)
            except json.JSONDecodeError as e:
                result = {
                    "is_valid": False,
                    "error": f"Repair attempt produced invalid JSON: {str(e)}",
                    "agent_json": agent_json  # Keep the original for next repair attempt
                }
                debug_print({**state, **result}, "repair_json - END (failed)")
                return result
        
        # Post-process the JSON to ensure valid Python types
        fixed_json = fix_return_types(fixed_json)
        
        # Validate the fixed JSON
        validation_result = validate_json({"agent_json": fixed_json, "debug": state.get('debug', False)})
        if not validation_result["is_valid"]:
            print(f"Initial repair attempt failed. Attempting deeper repair...")
            
            # Extract specific validation errors to provide more context
            detailed_error = validation_result["error"]
            
            # Second repair attempt with more specific error information
            messages.append(AIMessage(content=response.content))
            messages.append(HumanMessage(content=f"""
The fixed JSON still has validation errors:

{detailed_error}

Please fix these specific issues and return the complete fixed JSON."""))
            
            response = llm.invoke(messages)
            
            # Try to extract JSON again
            match = re.search(pattern, response.content)
            if match:
                fixed_json_str = match.group(1)
                try:
                    fixed_json = json.loads(fixed_json_str)
                    fixed_json = fix_return_types(fixed_json)
                except json.JSONDecodeError:
                    # If still invalid after two attempts, give up
                    result = {
                        "is_valid": False,
                        "error": "Could not repair JSON after multiple attempts",
                        "agent_json": agent_json  # Keep the original
                    }
                    debug_print({**state, **result}, "repair_json - END (failed)")
                    return result
            else:
                try:
                    fixed_json = json.loads(response.content)
                    fixed_json = fix_return_types(fixed_json)
                except json.JSONDecodeError:
                    # If still invalid after two attempts, give up
                    result = {
                        "is_valid": False,
                        "error": "Could not repair JSON after multiple attempts",
                        "agent_json": agent_json  # Keep the original
                    }
                    debug_print({**state, **result}, "repair_json - END (failed)")
                    return result
        
        print("JSON repair successful.")
        result = {"agent_json": fixed_json, "is_valid": True}
        debug_print({**state, **result}, "repair_json - END (success)")
        return result
    except Exception as e:
        print(f"JSON repair failed: {str(e)}")
        result = {
            "is_valid": False,
            "error": f"Could not repair JSON: {str(e)}",
            "agent_json": agent_json  # Keep the original
        }
        debug_print({**state, **result}, "repair_json - END (exception)")
        return result

def save_json(state: BuilderState) -> Dict[str, Any]:
    """
    Save the agent JSON definition to a file.
    """
    debug_print(state, "save_json - START")
    agent_json = state['agent_json']
    output_dir = state['output_dir']
    temp_py_path = state.get('py_path')
    temp_viz_path = state.get('viz_path')
    
    # Create a filename from the agent name
    agent_name = agent_json["metadata"]["name"].lower().replace(" ", "_")
    json_path = os.path.join(output_dir, f"{agent_name}.json")
    py_path = os.path.join(output_dir, f"{agent_name}.py")
    viz_path = os.path.join(output_dir, f"{agent_name}_viz.html")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the JSON
    with open(json_path, "w") as f:
        json.dump(agent_json, f, indent=2)
    
    print(f"Agent JSON saved to: {json_path}")
    
    # Copy the Python file if it was already created during validation
    result = {"json_path": json_path}
    
    if temp_py_path and os.path.exists(temp_py_path):
        # Copy the Python file to the output directory
        with open(temp_py_path, 'r') as src_file, open(py_path, 'w') as dest_file:
            dest_file.write(src_file.read())
        print(f"Agent Python code saved to: {py_path}")
        result["py_path"] = py_path
        
        # Clean up temporary Python file
        try:
            os.unlink(temp_py_path)
        except:
            pass
    
    # Copy the visualization file if it was created during validation
    if temp_viz_path and os.path.exists(temp_viz_path):
        # Copy the visualization file to the output directory
        with open(temp_viz_path, 'r') as src_file, open(viz_path, 'w') as dest_file:
            dest_file.write(src_file.read())
        print(f"Agent visualization saved to: {viz_path}")
        result["viz_path"] = viz_path
        
        # Clean up temporary visualization file
        try:
            os.unlink(temp_viz_path)
        except:
            pass
    
    debug_print({**state, **result}, "save_json - END")
    return result

def compile_code(state: BuilderState) -> Dict[str, Any]:
    """
    Compile the JSON to LangGraph Python code.
    """
    debug_print(state, "compile_code - START")
    print("Compiling agent to Python code...")
    
    agent_json = state['agent_json']
    json_path = state['json_path']
    output_dir = state['output_dir']
    py_path = state.get('py_path')
    
    # If Python file was already created during validation and saved, just use that
    if py_path and os.path.exists(py_path):
        print(f"Using previously generated Python code at: {py_path}")
        result = {"py_path": py_path}
        
        # If visualization was also already created, include it in the result
        viz_path = state.get('viz_path')
        if viz_path and os.path.exists(viz_path):
            print(f"Using previously generated visualization at: {viz_path}")
            result["viz_path"] = viz_path
            
        debug_print({**state, **result}, "compile_code - END (reusing)")
        return result
    
    # Otherwise, generate Python code from the JSON
    agent_name = agent_json["metadata"]["name"].lower().replace(" ", "_")
    py_path = os.path.join(output_dir, f"{agent_name}.py")
    viz_path = os.path.join(output_dir, f"{agent_name}_viz.html")
    
    try:
        # First validate that the JSON can be converted to valid Python code
        is_valid, error_msg = convert_json_to_langgraph(json_path, validate_only=True)
        
        if not is_valid:
            # Special handling for START/END node errors - make one more attempt
            if "Found edge starting at unknown node 'START'" in error_msg or "Found edge ending at unknown node 'END'" in error_msg:
                print("Detected START/END node reference error. Attempting to fix...")
                # Make a deep copy and apply fixes
                agent_json_fixed = json.loads(json.dumps(agent_json))
                
                # Fix edges directly
                if "edges" in agent_json_fixed:
                    for edge in agent_json_fixed["edges"]:
                        # Fix START node references
                        if edge.get("source") in ["START", "start", "start_node"]:
                            # Make sure there's a matching start node
                            has_start_node = False
                            for node in agent_json_fixed.get("nodes", []):
                                if node.get("id") in ["start", "START"] and node.get("type") == "start":
                                    has_start_node = True
                                    break
                            
                            if not has_start_node:
                                # Add a start node if needed
                                agent_json_fixed["nodes"].append({
                                    "id": "start",
                                    "type": "start",
                                    "data": {}
                                })
                                print("Added missing start node to repair edge reference")
                        
                        # Fix END node references
                        if edge.get("target") in ["END", "end", "end_node"]:
                            # Make sure there's a matching end node
                            has_end_node = False
                            for node in agent_json_fixed.get("nodes", []):
                                if node.get("id") in ["end", "END"] and node.get("type") == "end":
                                    has_end_node = True
                                    break
                            
                            if not has_end_node:
                                # Add an end node if needed
                                agent_json_fixed["nodes"].append({
                                    "id": "end",
                                    "type": "end",
                                    "data": {}
                                })
                                print("Added missing end node to repair edge reference")
                
                # Save fixed JSON
                temp_json_path = json_path + ".fixed"
                with open(temp_json_path, 'w') as f:
                    json.dump(agent_json_fixed, f, indent=2)
                
                # Try compilation again with fixed JSON
                is_valid, error_msg = convert_json_to_langgraph(temp_json_path, validate_only=True)
                if is_valid:
                    # Use the fixed version for the actual compilation
                    print("Fixed START/END node issue successfully")
                    json_path = temp_json_path
                else:
                    print(f"Fixed JSON still has validation issues: {error_msg}")
            
            if not is_valid:
                print(f"Python validation failed: {error_msg}")
                result = {
                    "is_valid": False,
                    "error": f"Python validation error: {error_msg}"
                }
                debug_print({**state, **result}, "compile_code - END (failed)")
                return result
        
        # If validation passes, generate the Python file
        convert_json_to_langgraph(json_path, py_path)
        print(f"Agent Python code saved to: {py_path}")
        
        # Try to load the module and visualize the agent
        try:
            spec = importlib.util.spec_from_file_location(agent_name, py_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, 'create_agent_graph'):
                    agent = module.create_agent_graph()
                    
                    # Create visualization of the agent
                    save_agent_visualization(
                        agent, 
                        viz_path, 
                        format="html"
                    )
                    print(f"Agent visualization saved to: {viz_path}")
                    
                    result = {"py_path": py_path, "viz_path": viz_path}
                else:
                    print(f"Warning: Generated module doesn't have a create_agent_graph function")
                    result = {"py_path": py_path}
            else:
                print(f"Warning: Could not load generated module for visualization")
                result = {"py_path": py_path}
        except Exception as e:
            print(f"Warning: Failed to visualize agent: {str(e)}")
            result = {"py_path": py_path}
        
        debug_print({**state, **result}, "compile_code - END (success)")
        return result
    except ValueError as e:
        error_message = str(e)
        print(f"Error in code generation: {error_message}")
        result = {
            "is_valid": False,
            "error": f"Python validation error: {error_message}"
        }
        debug_print({**state, **result}, "compile_code - END (exception)")
        return result

def decide_repair_condition(state: BuilderState) -> Literal["repair", "valid"]:
    """
    Check if the JSON needs repair and route accordingly.
    """
    debug_print(state, "decide_repair_condition")
    return "repair" if not state["is_valid"] else "valid"

# Helper functions
def fix_return_types(agent_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix Python return types and code blocks in the agent JSON.
    
    1. Ensures return types are valid Python types
    2. Fixes Python syntax in condition nodes and function nodes
    3. Fixes common node ID and edge naming issues
    """
    if not agent_json or not isinstance(agent_json, dict):
        return agent_json
    
    # Create a deep copy to avoid modifying the input directly
    fixed_json = json.loads(json.dumps(agent_json))
    
    # Fix Node IDs
    if "nodes" in fixed_json:
        # First make a map of all node IDs for validation
        node_ids = set()
        
        for i, node in enumerate(fixed_json["nodes"]):
            # Record all node IDs
            if "id" in node:
                node_ids.add(node["id"])
                
            # Fix common node ID naming issues
            if "id" in node:
                if node["id"] == "start_node":
                    node["id"] = "start"
                    node_ids.remove("start_node")
                    node_ids.add("start")
                elif node["id"] == "end_node":
                    node["id"] = "end"
                    node_ids.remove("end_node")
                    node_ids.add("end")
                
            # Skip nodes without data
            if "data" not in node:
                continue
                
            # Fix tool node return types
            if node["type"] == "tool" and "toolDefinition" in node["data"]:
                tool_def = node["data"]["toolDefinition"]
                if "returnType" in tool_def:
                    return_type = tool_def["returnType"]
                    # Convert JSON/TS-like types to Python types
                    if return_type == "string":
                        tool_def["returnType"] = "str"
                    elif return_type == "number":
                        tool_def["returnType"] = "float"
                    elif return_type == "boolean":
                        tool_def["returnType"] = "bool"
                    elif return_type == "any":
                        tool_def["returnType"] = "Any"
            
            # Fix Python code in function nodes
            if node["type"] == "function" and "code" in node["data"]:
                # Ensure code has return statement
                code = node["data"]["code"]
                if "return" not in code:
                    fixed_code = code.strip()
                    if not fixed_code.endswith(":"):  # Not ending with a colon (not a control structure)
                        # Add return statement if it doesn't end with one
                        fixed_code += "\nreturn {'result': 'Function execution completed'}"
                        node["data"]["code"] = fixed_code
            
            # Fix condition node logic
            if node["type"] == "condition" and "conditionLogic" in node["data"]:
                # Make sure condition returns a string value
                logic = node["data"]["conditionLogic"]
                
                # If logic doesn't have a return statement, add one
                if "return" not in logic:
                    fixed_logic = logic.strip()
                    if not fixed_logic.endswith(":"):  # Not ending with a colon
                        # Add a default return statement
                        fixed_logic += "\nreturn 'default'"
                        node["data"]["conditionLogic"] = fixed_logic
                elif not any(["return " in line and not line.strip().startswith("#") for line in logic.split("\n")]):
                    # If there's no non-commented return statement, add one
                    fixed_logic = logic.strip()
                    fixed_logic += "\nreturn 'default'"
                    node["data"]["conditionLogic"] = fixed_logic
    
    # Fix Edges
    if "edges" in fixed_json:
        for edge in fixed_json["edges"]:
            # Fix source references
            if "source" in edge:
                if edge["source"] in ["start_node", "start"]:
                    edge["source"] = "START"
                
            # Fix target references
            if "target" in edge:
                if edge["target"] in ["end_node", "end"]:
                    edge["target"] = "END"
    
    # Check for missing start/end nodes
    has_start_edge = False
    has_end_edge = False
    
    if "edges" in fixed_json:
        for edge in fixed_json["edges"]:
            if edge.get("source") == "START":
                has_start_edge = True
            if edge.get("target") == "END":
                has_end_edge = True
    
    # Add a warning if there are START/END references but no corresponding nodes
    if has_start_edge and not any(node.get("type") == "start" for node in fixed_json.get("nodes", [])):
        print("Warning: Found START edge references but no 'start' node. This may cause compilation issues.")
    
    if has_end_edge and not any(node.get("type") == "end" for node in fixed_json.get("nodes", [])):
        print("Warning: Found END edge references but no 'end' node. This may cause compilation issues.")
    
    # Fix State Definition
    if "stateDefinition" in fixed_json and "schema" in fixed_json["stateDefinition"]:
        schema = fixed_json["stateDefinition"]["schema"]
        for key, value in schema.items():
            # Convert JSON/TS-like types to Python types
            if value == "string":
                schema[key] = "str"
            elif value == "number":
                schema[key] = "float"
            elif value == "boolean":
                schema[key] = "bool"
            elif value == "any":
                schema[key] = "Any"
    
    return fixed_json

def capitalize_first(s: str) -> str:
    """Capitalize the first letter of a string."""
    if not s:
        return s
    return s[0].upper() + s[1:]

def is_valid_python_type(type_str: str) -> bool:
    """Check if a string represents a valid Python type."""
    valid_basic_types = ["str", "int", "float", "bool", "dict", "list", "set", "tuple", "None", "Any"]
    valid_complex_patterns = [
        r'^(List|Dict|Set|Tuple|Optional)\[.*\]$',  # e.g., List[str], Dict[str, int]
        r'^Union\[.*\]$',                          # e.g., Union[str, int]
        r'^Literal\[.*\]$',                        # e.g., Literal['a', 'b']
    ]
    
    if type_str in valid_basic_types:
        return True
    
    for pattern in valid_complex_patterns:
        if re.match(pattern, type_str):
            return True
    
    return False

# Create the LangGraph agent
def create_builder_graph():
    """Create a LangGraph graph for the builder agent."""
    # Define the graph
    graph = StateGraph(BuilderState)
    
    # Add nodes
    graph.add_node("analyze_request", analyze_request)
    graph.add_node("generate_agent_json", generate_agent_json)
    graph.add_node("validate_json", validate_json)
    graph.add_node("repair_json", repair_json)
    graph.add_node("save_json", save_json)
    graph.add_node("compile_code", compile_code)
    
    # Define edges
    graph.add_edge(START, "analyze_request")
    graph.add_edge("analyze_request", "generate_agent_json")
    graph.add_edge("generate_agent_json", "validate_json")
    
    # Add conditional edges from validate_json
    graph.add_conditional_edges(
        "validate_json",
        decide_repair_condition,
        {
            "repair": "repair_json",
            "valid": "save_json"
        }
    )
    
    graph.add_edge("repair_json", "validate_json")
    graph.add_edge("save_json", "compile_code")
    graph.add_edge("compile_code", END)
    
    return graph.compile()

def build_agent_from_request(request: str, output_dir: str = ".", model_name: str = "gpt-4", debug: bool = False) -> tuple:
    """
    Build an agent from a natural language request.
    
    Args:
        request: The natural language request
        output_dir: Directory to save the generated agent
        model_name: Name of the model to use
        debug: Whether to enable debug output
    
    Returns:
        Tuple of (success, result)
    """
    # Create the builder graph
    builder = create_builder_graph()
    
    # Initialize the state
    initial_state = {
        "request": request,
        "output_dir": output_dir,
        "model_name": model_name,
        "debug": debug  # Add debug flag to state
    }
    
    # Run the agent
    try:
        result = builder.invoke(initial_state)
        success = True
    except Exception as e:
        print(f"Error building agent: {str(e)}")
        result = {"error": str(e)}
        success = False
    
    if debug:
        print("\n==== FINAL STATE ====")
        for key, value in result.items():
            if key in ['agent_json']:
                print(f"  {key}: <JSON object>")
            elif key == 'error' and value:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        print("====================\n")
    
    return success, result
    
def main():
    """
    Command-line interface for the builder agent.
    """
    parser = argparse.ArgumentParser(description="Build a LangGraph agent from a natural language request")
    parser.add_argument("request", nargs="?", help="Natural language request for the agent")
    parser.add_argument("--output-dir", "-o", default="generated_agents", help="Directory to save the generated agent")
    parser.add_argument("--model", "-m", default="gpt-4", help="OpenAI model to use")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # If no request provided, prompt the user
    request = args.request
    if not request:
        request = input("Enter your agent request: ")
    
    # Build the agent
    success, result = build_agent_from_request(
        request, 
        output_dir=args.output_dir, 
        model_name=args.model,
        debug=args.debug
    )
    
    if success:
        print("\nAgent created successfully!")
        if result.get("py_path"):
            print(f"- Python file: {result['py_path']}")
        if result.get("json_path"):
            print(f"- JSON file: {result['json_path']}")
        if result.get("viz_path"):
            print(f"- Visualization: {result['viz_path']}")
            print(f"  (Open this HTML file in a browser to view the agent visualization)")
    else:
        print("\nFailed to create agent.")
        if result.get("error"):
            print(f"Error: {result['error']}")
    
    return 0 if success else 1

if __name__ == "__main__":
    main() 