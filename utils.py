"""
Visualization utilities for LangGraph agents.

This module provides functions to visualize LangGraph agents using Mermaid and Graphviz.
"""

from typing import Optional, Union, Any
from IPython.display import display, Image, HTML
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

def visualize_agent(
    agent: Any,
    display_mermaid_text: bool = True,
    display_mermaid_image: bool = True,
    display_graphviz: bool = True,
    curve_style: CurveStyle = CurveStyle.LINEAR,
    node_colors: Optional[NodeStyles] = None,
    wrap_label_n_words: int = 9,
    jupyter: bool = True
) -> None:
    """
    Visualize a LangGraph agent using Mermaid and/or Graphviz.
    
    Args:
        agent: The LangGraph agent to visualize
        display_mermaid_text: Whether to display the Mermaid diagram as text
        display_mermaid_image: Whether to display the Mermaid diagram as an image
        display_graphviz: Whether to display the Graphviz visualization
        curve_style: The curve style for the Mermaid diagram
        node_colors: Custom node colors for the Mermaid diagram
        wrap_label_n_words: Number of words to wrap labels at in the Mermaid diagram
        jupyter: Whether the function is being called from a Jupyter notebook
    
    Returns:
        None
    """
    if node_colors is None:
        node_colors = NodeStyles(
            first="#ffdfba",  # Light orange for start node
            last="#baffc9",   # Light green for end node
            default="#f2f0ff"  # Light purple for other nodes
        )
    
    graph = agent.get_graph()
    
    # Display Mermaid text representation
    if display_mermaid_text:
        mermaid_text = graph.draw_mermaid()
        print("Agent Mermaid Diagram:")
        print(mermaid_text)
        
        # Return HTML for rendering in non-Jupyter environments
        if not jupyter:
            html = f"""
            <pre class="mermaid">
            {mermaid_text}
            </pre>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{startOnLoad:true}});</script>
            """
            return HTML(html)
    
    # Display Mermaid image using the API
    if display_mermaid_image and jupyter:
        try:
            mermaid_img = graph.draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
                curve_style=curve_style,
                node_colors=node_colors,
                wrap_label_n_words=wrap_label_n_words,
            )
            print("\nMermaid Diagram Image:")
            display(Image(mermaid_img))
        except Exception as e:
            print(f"Failed to generate Mermaid image: {str(e)}")
    
    # Display Graphviz visualization
    if display_graphviz and jupyter:
        try:
            graphviz_img = graph.draw_png()
            print("\nGraphviz Visualization:")
            display(Image(graphviz_img))
        except ImportError:
            print("To use graphviz visualization, install pygraphviz: pip install pygraphviz")
        except Exception as e:
            print(f"Failed to generate Graphviz visualization: {str(e)}")

def get_agent_visualization_html(agent: Any) -> str:
    """
    Get HTML code for visualizing a LangGraph agent that can be embedded in a web page.
    
    Args:
        agent: The LangGraph agent to visualize
        
    Returns:
        str: HTML code for visualizing the agent
    """
    mermaid_text = agent.get_graph().draw_mermaid()
    
    html = f"""
    <div class="mermaid-diagram">
        <pre class="mermaid">
        {mermaid_text}
        </pre>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                curve: 'linear',
                useMaxWidth: true
            }}
        }});
    </script>
    """
    
    return html

def save_agent_visualization(
    agent: Any,
    output_path: str,
    format: str = "png",
    use_mermaid: bool = True,
    curve_style: CurveStyle = CurveStyle.LINEAR,
    node_colors: Optional[NodeStyles] = None,
) -> None:
    """
    Save a visualization of a LangGraph agent to a file.
    
    Args:
        agent: The LangGraph agent to visualize
        output_path: Path where to save the visualization
        format: Format to save the visualization in ('png', 'svg', or 'html')
        use_mermaid: Whether to use Mermaid (True) or Graphviz (False)
        curve_style: The curve style for the Mermaid diagram
        node_colors: Custom node colors for the Mermaid diagram
        
    Returns:
        None
    """
    if node_colors is None:
        node_colors = NodeStyles(
            first="#ffdfba",  # Light orange for start node
            last="#baffc9",   # Light green for end node
            default="#f2f0ff"  # Light purple for other nodes
        )
    
    graph = agent.get_graph()
    
    if format.lower() == 'html':
        # Save as HTML with embedded Mermaid
        mermaid_text = graph.draw_mermaid()
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agent Visualization</title>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>
                mermaid.initialize({{
                    startOnLoad: true,
                    theme: 'default',
                    flowchart: {{
                        curve: 'linear',
                        useMaxWidth: true
                    }}
                }});
            </script>
        </head>
        <body>
            <div class="mermaid-diagram">
                <pre class="mermaid">
                {mermaid_text}
                </pre>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
        print(f"HTML visualization saved to {output_path}")
        
    elif format.lower() in ['png', 'svg']:
        if use_mermaid:
            # Save using Mermaid
            try:
                if format.lower() == 'png':
                    img_data = graph.draw_mermaid_png(
                        draw_method=MermaidDrawMethod.API,
                        curve_style=curve_style,
                        node_colors=node_colors,
                    )
                    with open(output_path, 'wb') as f:
                        f.write(img_data)
                    print(f"PNG visualization saved to {output_path}")
                else:  # svg
                    img_data = graph.draw_mermaid_svg(
                        draw_method=MermaidDrawMethod.API,
                        curve_style=curve_style,
                        node_colors=node_colors,
                    )
                    with open(output_path, 'wb') as f:
                        f.write(img_data)
                    print(f"SVG visualization saved to {output_path}")
            except Exception as e:
                print(f"Failed to save Mermaid visualization: {str(e)}")
        else:
            # Save using Graphviz
            try:
                if format.lower() == 'png':
                    img_data = graph.draw_png()
                    with open(output_path, 'wb') as f:
                        f.write(img_data)
                    print(f"PNG visualization saved to {output_path}")
                else:  # svg
                    img_data = graph.draw_svg()
                    with open(output_path, 'wb') as f:
                        f.write(img_data)
                    print(f"SVG visualization saved to {output_path}")
            except ImportError:
                print("To use graphviz visualization, install pygraphviz: pip install pygraphviz")
            except Exception as e:
                print(f"Failed to save Graphviz visualization: {str(e)}")
    else:
        print(f"Unsupported format: {format}. Use 'png', 'svg', or 'html'.") 