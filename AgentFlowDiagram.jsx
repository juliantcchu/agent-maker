import React, { useState, useCallback, useEffect } from 'react';
import ReactFlow, {
  Controls,
  Background,
  Panel,
  useNodesState,
  useEdgesState,
  addEdge,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';

// Node types
const nodeTypes = {
  start: ({ data }) => (
    <div className="node-start">
      <div className="node-header">Start</div>
      <div className="node-content">{data.label}</div>
    </div>
  ),
  end: ({ data }) => (
    <div className="node-end">
      <div className="node-header">End</div>
      <div className="node-content">{data.label}</div>
    </div>
  ),
  llm: ({ data }) => (
    <div className="node-llm">
      <div className="node-header">LLM</div>
      <div className="node-content">
        <div className="node-label">{data.label}</div>
        {data.description && <div className="node-description">{data.description}</div>}
        {data.model && <div className="node-model">Model: {data.model}</div>}
      </div>
    </div>
  ),
  tool: ({ data }) => (
    <div className="node-tool">
      <div className="node-header">Tool</div>
      <div className="node-content">
        <div className="node-label">{data.label}</div>
        {data.description && <div className="node-description">{data.description}</div>}
        {data.toolDefinition && data.toolDefinition.name && 
          <div className="node-tool-name">Tool: {data.toolDefinition.name}</div>}
      </div>
    </div>
  ),
  function: ({ data }) => (
    <div className="node-function">
      <div className="node-header">Function</div>
      <div className="node-content">
        <div className="node-label">{data.label}</div>
        {data.description && <div className="node-description">{data.description}</div>}
      </div>
    </div>
  ),
  condition: ({ data }) => (
    <div className="node-condition">
      <div className="node-header">Condition</div>
      <div className="node-content">
        <div className="node-label">{data.label}</div>
        {data.description && <div className="node-description">{data.description}</div>}
      </div>
    </div>
  ),
  custom: ({ data }) => (
    <div className="node-custom">
      <div className="node-header">Custom: {data.customType || 'Unknown'}</div>
      <div className="node-content">
        <div className="node-label">{data.label}</div>
        {data.description && <div className="node-description">{data.description}</div>}
      </div>
    </div>
  ),
};

// Helper to convert JSON definition to React Flow elements
const jsonToReactFlow = (agentDefinition) => {
  // Extract nodes and set proper React Flow format
  const nodes = agentDefinition.nodes.map((node) => ({
    id: node.id,
    type: node.type,
    position: node.position,
    data: node.data,
  }));
  
  // Extract edges and set proper React Flow format
  const edges = agentDefinition.edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    label: edge.label || '',
    markerEnd: {
      type: MarkerType.ArrowClosed,
    },
    style: { stroke: '#555' },
    // Add handle identifiers if present
    sourceHandle: edge.sourceHandle,
    targetHandle: edge.targetHandle,
  }));
  
  return { nodes, edges };
};

// Main component
const AgentFlowDiagram = ({ agentDefinition }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Initialize diagram when agentDefinition changes
  useEffect(() => {
    if (agentDefinition) {
      const { nodes: flowNodes, edges: flowEdges } = jsonToReactFlow(agentDefinition);
      setNodes(flowNodes);
      setEdges(flowEdges);
    }
  }, [agentDefinition, setNodes, setEdges]);

  // Handle connections when edges are manually created
  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  // Styles
  const flowStyles = {
    width: '100%',
    height: '800px',
  };

  return (
    <div style={flowStyles}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
      >
        <Controls />
        <Background />
        <Panel position="top-left">
          <h3>{agentDefinition?.metadata?.name || 'Agent Flow Diagram'}</h3>
          <p>{agentDefinition?.metadata?.description || ''}</p>
        </Panel>
      </ReactFlow>
    </div>
  );
};

export default AgentFlowDiagram; 