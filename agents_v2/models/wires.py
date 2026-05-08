"""
Wire-related data models with spatial information
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from .components import Point, LineSegment, BoundingBox


@dataclass
class SegmentAnnotation:
    """Individual wire segment annotation for training"""
    
    # Line segment (x1, y1, x2, y2) in pixels
    x1: int
    y1: int
    x2: int
    y2: int
    
    # Properties
    length: float = 0.0
    angle: float = 0.0               # Degrees from horizontal
    
    # Classification
    segment_type: str = "wire"       # "wire", "component_internal", "noise", "text"
    confidence: float = 0.0
    
    # Source
    hough_line_id: Optional[int] = None
    
    def __post_init__(self):
        if self.length == 0.0:
            import math
            self.length = math.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
        if self.angle == 0.0:
            import math
            self.angle = math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))
    
    def to_line_segment(self) -> LineSegment:
        return LineSegment(
            start=Point(x=self.x1, y=self.y1),
            end=Point(x=self.x2, y=self.y2)
        )
    
    def to_normalized(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Get normalized coordinates"""
        return (
            self.x1 / img_width,
            self.y1 / img_height,
            self.x2 / img_width,
            self.y2 / img_height
        )
    
    def to_dict(self) -> Dict:
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "length": self.length,
            "angle": self.angle,
            "segment_type": self.segment_type,
            "confidence": self.confidence,
            "hough_line_id": self.hough_line_id
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'SegmentAnnotation':
        return cls(**d)


@dataclass
class JunctionInfo:
    """Wire junction/intersection information"""
    
    id: str
    position: Point
    type: str = "junction"           # "junction", "crossing", "corner", "T_junction"
    
    # What meets here
    connected_wires: List[str] = field(default_factory=list)
    connected_terminals: List[str] = field(default_factory=list)
    
    # Visual info
    has_junction_dot: bool = False   # Explicit dot in schematic
    is_crossing_not_junction: bool = False  # Wires cross but don't connect
    
    # Electrical node
    electrical_node: str = ""
    
    # Confidence
    confidence: float = 0.0
    source: str = "hough_intersection"  # or "agent_identified"
    
    @property
    def connection_count(self) -> int:
        return len(self.connected_wires) + len(self.connected_terminals)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "type": self.type,
            "connected_wires": self.connected_wires,
            "connected_terminals": self.connected_terminals,
            "has_junction_dot": self.has_junction_dot,
            "is_crossing_not_junction": self.is_crossing_not_junction,
            "electrical_node": self.electrical_node,
            "confidence": self.confidence,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'JunctionInfo':
        return cls(
            id=d["id"],
            position=Point.from_dict(d["position"]),
            type=d.get("type", "junction"),
            connected_wires=d.get("connected_wires", []),
            connected_terminals=d.get("connected_terminals", []),
            has_junction_dot=d.get("has_junction_dot", False),
            is_crossing_not_junction=d.get("is_crossing_not_junction", False),
            electrical_node=d.get("electrical_node", ""),
            confidence=d.get("confidence", 0.0),
            source=d.get("source", "hough_intersection")
        )


@dataclass
class WireInfo:
    """Wire segment information with path data"""
    
    id: str                          # "W1", "W2", etc.
    
    # Path (ordered list of points)
    path: List[Point] = field(default_factory=list)
    
    # What it connects
    start_connection: str = ""       # "R1.pin1" or "junction_3"
    end_connection: str = ""
    electrical_node: str = ""        # Which electrical node this wire is part of
    
    # Segments
    segments: List[SegmentAnnotation] = field(default_factory=list)
    
    # Junctions along this wire
    junctions: List[str] = field(default_factory=list)  # Junction IDs
    
    # Properties (computed)
    total_length: float = 0.0
    bend_count: int = 0
    
    # Source info
    source: str = "hough_kept"       # "hough_kept", "hough_merged", "agent_traced"
    source_hough_ids: List[int] = field(default_factory=list)
    confidence: float = 0.0
    
    @property
    def start_point(self) -> Optional[Point]:
        return self.path[0] if self.path else None
    
    @property
    def end_point(self) -> Optional[Point]:
        return self.path[-1] if self.path else None
    
    @property
    def is_horizontal(self) -> bool:
        if len(self.path) < 2:
            return False
        return all(abs(self.path[i].y - self.path[i+1].y) < 5 
                   for i in range(len(self.path)-1))
    
    @property
    def is_vertical(self) -> bool:
        if len(self.path) < 2:
            return False
        return all(abs(self.path[i].x - self.path[i+1].x) < 5 
                   for i in range(len(self.path)-1))
    
    def compute_properties(self):
        """Compute derived properties from path"""
        if len(self.path) < 2:
            return
        
        # Total length
        self.total_length = sum(
            self.path[i].distance_to(self.path[i+1])
            for i in range(len(self.path) - 1)
        )
        
        # Bend count (significant direction changes)
        if len(self.path) >= 3:
            import math
            bends = 0
            for i in range(1, len(self.path) - 1):
                # Calculate angle change at this point
                v1 = (self.path[i].x - self.path[i-1].x, 
                      self.path[i].y - self.path[i-1].y)
                v2 = (self.path[i+1].x - self.path[i].x,
                      self.path[i+1].y - self.path[i].y)
                
                # Cross product for angle
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                dot = v1[0] * v2[0] + v1[1] * v2[1]
                angle = abs(math.degrees(math.atan2(cross, dot)))
                
                if angle > 20:  # More than 20 degrees = bend
                    bends += 1
            
            self.bend_count = bends
    
    def get_bounding_box(self) -> Optional[BoundingBox]:
        """Get bounding box of wire path"""
        if not self.path:
            return None
        
        x_coords = [p.x for p in self.path]
        y_coords = [p.y for p in self.path]
        
        return BoundingBox(
            x_min=min(x_coords),
            y_min=min(y_coords),
            x_max=max(x_coords),
            y_max=max(y_coords)
        )
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "path": [p.to_dict() for p in self.path],
            "start_connection": self.start_connection,
            "end_connection": self.end_connection,
            "electrical_node": self.electrical_node,
            "segments": [s.to_dict() for s in self.segments],
            "junctions": self.junctions,
            "total_length": self.total_length,
            "bend_count": self.bend_count,
            "source": self.source,
            "source_hough_ids": self.source_hough_ids,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'WireInfo':
        wire = cls(
            id=d["id"],
            path=[Point.from_dict(p) for p in d.get("path", [])],
            start_connection=d.get("start_connection", ""),
            end_connection=d.get("end_connection", ""),
            electrical_node=d.get("electrical_node", ""),
            segments=[SegmentAnnotation.from_dict(s) for s in d.get("segments", [])],
            junctions=d.get("junctions", []),
            total_length=d.get("total_length", 0.0),
            bend_count=d.get("bend_count", 0),
            source=d.get("source", "hough_kept"),
            source_hough_ids=d.get("source_hough_ids", []),
            confidence=d.get("confidence", 0.0)
        )
        return wire

