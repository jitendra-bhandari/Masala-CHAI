"""
Training data collection models for multi-head YOLO training
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from .components import Point, BoundingBox, ComponentInfo
from .wires import WireInfo, SegmentAnnotation
from .nodes import NodeInfo, TopologyAnnotation
from .difficulty import DifficultyScores, DifficultyFactors


@dataclass
class ComponentAnnotation:
    """Annotation for component detection head (YOLO format)"""
    
    # YOLO-format bbox (normalized 0-1)
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    
    # Additional attributes for multi-task learning
    confidence: float = 0.0
    orientation: int = 0              # 0=horizontal, 1=vertical, 2=diagonal
    value_present: bool = False
    terminal_count: int = 2
    
    # Source tracking
    source: str = "yolo_confirmed"    # "yolo_confirmed", "agent_added", "agent_corrected"
    original_bbox: Optional[List[float]] = None
    correction_type: Optional[str] = None
    
    # Difficulty contribution
    detection_difficulty: float = 0.0
    
    def to_yolo_line(self) -> str:
        """Export as YOLO annotation line"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"
    
    def to_dict(self) -> Dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "orientation": self.orientation,
            "value_present": self.value_present,
            "terminal_count": self.terminal_count,
            "source": self.source,
            "original_bbox": self.original_bbox,
            "correction_type": self.correction_type,
            "detection_difficulty": self.detection_difficulty
        }


@dataclass
class WireAnnotation:
    """Annotation for wire detection head"""
    
    # Wire ID
    wire_id: str
    
    # Polyline (normalized coordinates)
    points: List[Tuple[float, float]] = field(default_factory=list)
    
    # Segment-based representation
    segments: List[SegmentAnnotation] = field(default_factory=list)
    
    # Properties
    is_horizontal: bool = False
    is_vertical: bool = False
    is_diagonal: bool = False
    total_length: float = 0.0
    bend_count: int = 0
    
    # Connectivity
    start_connection: str = ""
    end_connection: str = ""
    electrical_node: str = ""
    
    # Source tracking
    source_hough_ids: List[int] = field(default_factory=list)
    source: str = "hough_kept"
    
    # Difficulty
    tracing_difficulty: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "wire_id": self.wire_id,
            "points": self.points,
            "segments": [s.to_dict() for s in self.segments],
            "is_horizontal": self.is_horizontal,
            "is_vertical": self.is_vertical,
            "is_diagonal": self.is_diagonal,
            "total_length": self.total_length,
            "bend_count": self.bend_count,
            "start_connection": self.start_connection,
            "end_connection": self.end_connection,
            "electrical_node": self.electrical_node,
            "source_hough_ids": self.source_hough_ids,
            "source": self.source,
            "tracing_difficulty": self.tracing_difficulty
        }


@dataclass
class NodeAnnotation:
    """Annotation for node/junction detection head"""
    
    # Position (normalized 0-1)
    x: float
    y: float
    
    # Properties
    node_type: str = "junction"       # "junction", "crossing", "terminal", "corner"
    class_id: int = 0                 # For classification head
    is_junction: bool = True
    has_dot: bool = False
    
    # Connectivity
    connected_wire_count: int = 0
    connected_terminal_count: int = 0
    electrical_node_id: str = ""
    
    # Source
    source: str = "hough_intersection"
    
    # Difficulty
    classification_difficulty: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "x": self.x,
            "y": self.y,
            "node_type": self.node_type,
            "class_id": self.class_id,
            "is_junction": self.is_junction,
            "has_dot": self.has_dot,
            "connected_wire_count": self.connected_wire_count,
            "connected_terminal_count": self.connected_terminal_count,
            "electrical_node_id": self.electrical_node_id,
            "source": self.source,
            "classification_difficulty": self.classification_difficulty
        }


@dataclass
class TerminalAnnotation:
    """Annotation for terminal/pin detection head"""
    
    # Position (normalized 0-1)
    x: float
    y: float
    
    # Parent component
    component_id: str
    terminal_name: str
    
    # Properties
    terminal_type: str = "signal"     # "power", "signal", "ground"
    class_id: int = 0
    
    # Connection
    connected_to_node: str = ""
    connected_via_wire: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "x": self.x,
            "y": self.y,
            "component_id": self.component_id,
            "terminal_name": self.terminal_name,
            "terminal_type": self.terminal_type,
            "class_id": self.class_id,
            "connected_to_node": self.connected_to_node,
            "connected_via_wire": self.connected_via_wire
        }


@dataclass
class ChangeLogEntry:
    """Individual change made by agent"""
    
    agent: str                        # "agent1", "agent2", "agent3"
    iteration: int
    action: str                       # "add", "remove", "modify", "merge", "confirm"
    target_type: str                  # "component", "wire", "node", "terminal"
    target_id: str
    before: Optional[Dict] = None
    after: Optional[Dict] = None
    reason: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "agent": self.agent,
            "iteration": self.iteration,
            "action": self.action,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "before": self.before,
            "after": self.after,
            "reason": self.reason,
            "confidence": self.confidence
        }


@dataclass
class RefinementDeltas:
    """What the agents changed from raw CV output"""
    
    # Component changes
    components_confirmed: int = 0
    components_rejected: int = 0
    components_added: int = 0
    components_type_corrected: int = 0
    components_bbox_adjusted: int = 0
    
    # Wire changes
    wires_kept: int = 0
    wires_rejected: int = 0
    wires_merged: int = 0
    wires_added: int = 0
    
    # Node changes
    nodes_confirmed: int = 0
    nodes_reclassified: int = 0
    nodes_added: int = 0
    nodes_merged: int = 0
    
    # Terminal changes
    terminals_located: int = 0
    terminals_connected: int = 0
    
    # Detailed change log
    change_log: List[ChangeLogEntry] = field(default_factory=list)
    
    def add_change(self, agent: str, iteration: int, action: str,
                   target_type: str, target_id: str, **kwargs):
        """Add a change to the log"""
        entry = ChangeLogEntry(
            agent=agent,
            iteration=iteration,
            action=action,
            target_type=target_type,
            target_id=target_id,
            **kwargs
        )
        self.change_log.append(entry)
        
        # Update counters
        if target_type == "component":
            if action == "confirm":
                self.components_confirmed += 1
            elif action == "remove":
                self.components_rejected += 1
            elif action == "add":
                self.components_added += 1
            elif action == "modify":
                self.components_type_corrected += 1
        elif target_type == "wire":
            if action == "confirm":
                self.wires_kept += 1
            elif action == "remove":
                self.wires_rejected += 1
            elif action == "merge":
                self.wires_merged += 1
            elif action == "add":
                self.wires_added += 1
    
    def to_dict(self) -> Dict:
        return {
            "components_confirmed": self.components_confirmed,
            "components_rejected": self.components_rejected,
            "components_added": self.components_added,
            "components_type_corrected": self.components_type_corrected,
            "components_bbox_adjusted": self.components_bbox_adjusted,
            "wires_kept": self.wires_kept,
            "wires_rejected": self.wires_rejected,
            "wires_merged": self.wires_merged,
            "wires_added": self.wires_added,
            "nodes_confirmed": self.nodes_confirmed,
            "nodes_reclassified": self.nodes_reclassified,
            "nodes_added": self.nodes_added,
            "nodes_merged": self.nodes_merged,
            "terminals_located": self.terminals_located,
            "terminals_connected": self.terminals_connected,
            "change_log": [c.to_dict() for c in self.change_log]
        }


@dataclass
class TrainingDataSample:
    """Complete training sample for multi-head YOLOv11m"""
    
    # ============= SAMPLE IDENTITY =============
    sample_id: str
    image_path: str
    image_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # ============= IMAGE DATA =============
    image_width: int = 0
    image_height: int = 0
    image_channels: int = 3
    
    # ============= QUALITY FLAGS =============
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    pipeline_status: str = "pending"  # SUCCESS | PARTIAL | FAILED
    quality_tier: str = "pending"     # VERIFIED | CONFIDENT | LOW_CONFIDENCE | FAILED
    overall_confidence: float = 0.0
    
    # ============= DIFFICULTY SCORING =============
    difficulty_scores: Optional[DifficultyScores] = None
    
    # ============= ANNOTATIONS =============
    component_annotations: List[ComponentAnnotation] = field(default_factory=list)
    wire_annotations: List[WireAnnotation] = field(default_factory=list)
    node_annotations: List[NodeAnnotation] = field(default_factory=list)
    terminal_annotations: List[TerminalAnnotation] = field(default_factory=list)
    
    # ============= REFINEMENT DELTAS =============
    refinement_deltas: Optional[RefinementDeltas] = None
    
    # ============= RAW CV DATA =============
    raw_yolo_detections: List[Dict] = field(default_factory=list)
    raw_hough_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    raw_junction_candidates: List[Tuple[int, int]] = field(default_factory=list)
    
    # ============= TOPOLOGY =============
    topology: Optional[TopologyAnnotation] = None
    
    # ============= SPICE OUTPUT =============
    spice_netlist: str = ""
    simulation_successful: bool = False
    
    def export_yolo_annotations(self, class_map: Dict[str, int]) -> str:
        """Export component annotations in YOLO format"""
        lines = []
        for ann in self.component_annotations:
            lines.append(ann.to_yolo_line())
        return "\n".join(lines)
    
    def export_coco_annotations(self, image_id: int, class_map: Dict[str, int]) -> Dict:
        """Export in COCO format"""
        annotations = []
        for i, ann in enumerate(self.component_annotations):
            # Convert normalized to pixels
            x = (ann.x_center - ann.width / 2) * self.image_width
            y = (ann.y_center - ann.height / 2) * self.image_height
            w = ann.width * self.image_width
            h = ann.height * self.image_height
            
            annotations.append({
                "id": i,
                "image_id": image_id,
                "category_id": ann.class_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "attributes": {
                    "orientation": ann.orientation,
                    "value_present": ann.value_present,
                    "source": ann.source
                }
            })
        
        return {
            "image": {
                "id": image_id,
                "file_name": self.image_path,
                "width": self.image_width,
                "height": self.image_height
            },
            "annotations": annotations
        }
    
    def to_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "image_hash": self.image_hash,
            "timestamp": self.timestamp,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_channels": self.image_channels,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "pipeline_status": self.pipeline_status,
            "quality_tier": self.quality_tier,
            "overall_confidence": self.overall_confidence,
            "difficulty_scores": self.difficulty_scores.to_dict() if self.difficulty_scores else None,
            "component_annotations": [a.to_dict() for a in self.component_annotations],
            "wire_annotations": [a.to_dict() for a in self.wire_annotations],
            "node_annotations": [a.to_dict() for a in self.node_annotations],
            "terminal_annotations": [a.to_dict() for a in self.terminal_annotations],
            "refinement_deltas": self.refinement_deltas.to_dict() if self.refinement_deltas else None,
            "raw_yolo_detections": self.raw_yolo_detections,
            "raw_hough_lines": self.raw_hough_lines,
            "raw_junction_candidates": self.raw_junction_candidates,
            "topology": self.topology.to_dict() if self.topology else None,
            "spice_netlist": self.spice_netlist,
            "simulation_successful": self.simulation_successful
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingDataSample':
        return cls(
            sample_id=d["sample_id"],
            image_path=d["image_path"],
            image_hash=d.get("image_hash", ""),
            timestamp=d.get("timestamp", ""),
            image_width=d.get("image_width", 0),
            image_height=d.get("image_height", 0),
            image_channels=d.get("image_channels", 3),
            needs_review=d.get("needs_review", False),
            review_reasons=d.get("review_reasons", []),
            pipeline_status=d.get("pipeline_status", "pending"),
            quality_tier=d.get("quality_tier", "pending"),
            overall_confidence=d.get("overall_confidence", 0.0),
            difficulty_scores=DifficultyScores.from_dict(d["difficulty_scores"]) if d.get("difficulty_scores") else None,
            component_annotations=[],  # Would need to implement from_dict for these
            wire_annotations=[],
            node_annotations=[],
            terminal_annotations=[],
            refinement_deltas=None,
            raw_yolo_detections=d.get("raw_yolo_detections", []),
            raw_hough_lines=d.get("raw_hough_lines", []),
            raw_junction_candidates=d.get("raw_junction_candidates", []),
            topology=TopologyAnnotation.from_dict(d["topology"]) if d.get("topology") else None,
            spice_netlist=d.get("spice_netlist", ""),
            simulation_successful=d.get("simulation_successful", False)
        )

