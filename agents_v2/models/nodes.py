"""
Node and topology graph data models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from .components import Point, BoundingBox
import json


@dataclass
class NodeInfo:
    """Electrical node information with spatial data"""
    
    id: str                          # "VDD", "0", "node_1", etc.
    type: str = "internal"           # "power", "ground", "signal", "internal"
    
    # Spatial extent
    positions: List[Point] = field(default_factory=list)  # All points where node exists
    
    # Connectivity
    connected_terminals: List[str] = field(default_factory=list)  # ["R1.pin1", "M1.drain"]
    connected_wires: List[str] = field(default_factory=list)  # ["W1", "W5"]
    
    # Electrical properties
    is_power: bool = False
    is_ground: bool = False
    voltage_label: Optional[str] = None  # "VDD", "5V", etc.
    
    # Confidence
    confidence: float = 0.0
    source: str = "agent_identified"
    
    @property
    def degree(self) -> int:
        """Number of connections"""
        return len(self.connected_terminals)
    
    @property
    def centroid(self) -> Optional[Point]:
        """Average position of all node points"""
        if not self.positions:
            return None
        x = sum(p.x for p in self.positions) // len(self.positions)
        y = sum(p.y for p in self.positions) // len(self.positions)
        return Point(x=x, y=y)
    
    @property
    def bounding_region(self) -> Optional[BoundingBox]:
        """Bounding box of all node points"""
        if not self.positions:
            return None
        x_coords = [p.x for p in self.positions]
        y_coords = [p.y for p in self.positions]
        return BoundingBox(
            x_min=min(x_coords),
            y_min=min(y_coords),
            x_max=max(x_coords),
            y_max=max(y_coords)
        )
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "positions": [p.to_dict() for p in self.positions],
            "connected_terminals": self.connected_terminals,
            "connected_wires": self.connected_wires,
            "is_power": self.is_power,
            "is_ground": self.is_ground,
            "voltage_label": self.voltage_label,
            "confidence": self.confidence,
            "source": self.source,
            "degree": self.degree,
            "centroid": self.centroid.to_dict() if self.centroid else None
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'NodeInfo':
        return cls(
            id=d["id"],
            type=d.get("type", "internal"),
            positions=[Point.from_dict(p) for p in d.get("positions", [])],
            connected_terminals=d.get("connected_terminals", []),
            connected_wires=d.get("connected_wires", []),
            is_power=d.get("is_power", False),
            is_ground=d.get("is_ground", False),
            voltage_label=d.get("voltage_label"),
            confidence=d.get("confidence", 0.0),
            source=d.get("source", "agent_identified")
        )


@dataclass
class GraphNode:
    """Node in topology graph for GNN training"""
    
    node_id: int
    node_type: str                    # "component" or "electrical_node"
    name: str                         # Original name (R1, VDD, etc.)
    
    # If component
    component_id: Optional[str] = None
    component_type: Optional[str] = None
    
    # If electrical node
    electrical_node_id: Optional[str] = None
    is_power: bool = False
    is_ground: bool = False
    
    # Features for GNN (can be extended)
    features: List[float] = field(default_factory=list)
    
    # Position for visualization
    position: Optional[Point] = None
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "component_id": self.component_id,
            "component_type": self.component_type,
            "electrical_node_id": self.electrical_node_id,
            "is_power": self.is_power,
            "is_ground": self.is_ground,
            "features": self.features,
            "position": self.position.to_dict() if self.position else None
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'GraphNode':
        return cls(
            node_id=d["node_id"],
            node_type=d["node_type"],
            name=d["name"],
            component_id=d.get("component_id"),
            component_type=d.get("component_type"),
            electrical_node_id=d.get("electrical_node_id"),
            is_power=d.get("is_power", False),
            is_ground=d.get("is_ground", False),
            features=d.get("features", []),
            position=Point.from_dict(d["position"]) if d.get("position") else None
        )


@dataclass
class GraphEdge:
    """Edge in topology graph"""
    
    source: int                       # Source node ID
    target: int                       # Target node ID
    edge_type: str = "connection"     # "connection", "terminal", etc.
    
    # Edge attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # For training
    weight: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "attributes": self.attributes,
            "weight": self.weight
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'GraphEdge':
        return cls(
            source=d["source"],
            target=d["target"],
            edge_type=d.get("edge_type", "connection"),
            attributes=d.get("attributes", {}),
            weight=d.get("weight", 1.0)
        )


@dataclass
class TopologyAnnotation:
    """Graph structure for GNN training"""
    
    # Node list (components + electrical nodes)
    graph_nodes: List[GraphNode] = field(default_factory=list)
    
    # Edge list
    graph_edges: List[GraphEdge] = field(default_factory=list)
    
    # Mappings
    name_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_name: Dict[int, str] = field(default_factory=dict)
    
    # Statistics
    num_components: int = 0
    num_electrical_nodes: int = 0
    num_edges: int = 0
    
    def add_component_node(self, component_id: str, component_type: str, 
                           position: Optional[Point] = None) -> int:
        """Add a component node to the graph"""
        node_id = len(self.graph_nodes)
        node = GraphNode(
            node_id=node_id,
            node_type="component",
            name=component_id,
            component_id=component_id,
            component_type=component_type,
            position=position
        )
        self.graph_nodes.append(node)
        self.name_to_id[component_id] = node_id
        self.id_to_name[node_id] = component_id
        self.num_components += 1
        return node_id
    
    def add_electrical_node(self, node_id_str: str, is_power: bool = False,
                            is_ground: bool = False, position: Optional[Point] = None) -> int:
        """Add an electrical node to the graph"""
        graph_node_id = len(self.graph_nodes)
        node = GraphNode(
            node_id=graph_node_id,
            node_type="electrical_node",
            name=node_id_str,
            electrical_node_id=node_id_str,
            is_power=is_power,
            is_ground=is_ground,
            position=position
        )
        self.graph_nodes.append(node)
        self.name_to_id[node_id_str] = graph_node_id
        self.id_to_name[graph_node_id] = node_id_str
        self.num_electrical_nodes += 1
        return graph_node_id
    
    def add_edge(self, source_name: str, target_name: str, 
                 edge_type: str = "connection", **attributes) -> Optional[GraphEdge]:
        """Add an edge between two nodes"""
        if source_name not in self.name_to_id or target_name not in self.name_to_id:
            return None
        
        edge = GraphEdge(
            source=self.name_to_id[source_name],
            target=self.name_to_id[target_name],
            edge_type=edge_type,
            attributes=attributes
        )
        self.graph_edges.append(edge)
        self.num_edges += 1
        return edge
    
    def get_adjacency_list(self) -> Dict[int, List[int]]:
        """Get adjacency list representation"""
        adj = {i: [] for i in range(len(self.graph_nodes))}
        for edge in self.graph_edges:
            adj[edge.source].append(edge.target)
            adj[edge.target].append(edge.source)  # Undirected
        return adj
    
    def get_adjacency_matrix(self) -> List[List[int]]:
        """Get adjacency matrix representation"""
        n = len(self.graph_nodes)
        matrix = [[0] * n for _ in range(n)]
        for edge in self.graph_edges:
            matrix[edge.source][edge.target] = 1
            matrix[edge.target][edge.source] = 1  # Undirected
        return matrix
    
    def to_networkx_json(self) -> str:
        """Export to NetworkX-compatible JSON"""
        data = {
            "directed": False,
            "multigraph": False,
            "graph": {},
            "nodes": [
                {"id": n.node_id, **n.to_dict()}
                for n in self.graph_nodes
            ],
            "links": [
                {"source": e.source, "target": e.target, **e.to_dict()}
                for e in self.graph_edges
            ]
        }
        return json.dumps(data, indent=2)
    
    def to_dict(self) -> Dict:
        return {
            "graph_nodes": [n.to_dict() for n in self.graph_nodes],
            "graph_edges": [e.to_dict() for e in self.graph_edges],
            "name_to_id": self.name_to_id,
            "id_to_name": {str(k): v for k, v in self.id_to_name.items()},
            "num_components": self.num_components,
            "num_electrical_nodes": self.num_electrical_nodes,
            "num_edges": self.num_edges
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TopologyAnnotation':
        topo = cls(
            graph_nodes=[GraphNode.from_dict(n) for n in d.get("graph_nodes", [])],
            graph_edges=[GraphEdge.from_dict(e) for e in d.get("graph_edges", [])],
            name_to_id=d.get("name_to_id", {}),
            id_to_name={int(k): v for k, v in d.get("id_to_name", {}).items()},
            num_components=d.get("num_components", 0),
            num_electrical_nodes=d.get("num_electrical_nodes", 0),
            num_edges=d.get("num_edges", 0)
        )
        return topo

