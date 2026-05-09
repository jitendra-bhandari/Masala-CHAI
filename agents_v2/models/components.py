"""
Component-related data models with spatial information
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math


@dataclass
class Point:
    """2D point in pixel coordinates"""
    x: int
    y: int
    
    def to_normalized(self, img_width: int, img_height: int) -> Tuple[float, float]:
        """Convert to normalized coordinates (0-1)"""
        return (self.x / img_width, self.y / img_height)
    
    def distance_to(self, other: 'Point') -> float:
        """Euclidean distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y}
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Point':
        return cls(x=d["x"], y=d["y"])


@dataclass
class BoundingBox:
    """Bounding box in pixel coordinates"""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    
    @property
    def width(self) -> int:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> int:
        return self.y_max - self.y_min
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Point:
        return Point(
            x=(self.x_min + self.x_max) // 2,
            y=(self.y_min + self.y_max) // 2
        )
    
    def to_yolo_format(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert to YOLO format (x_center, y_center, width, height) normalized"""
        x_center = (self.x_min + self.x_max) / 2 / img_width
        y_center = (self.y_min + self.y_max) / 2 / img_height
        width = self.width / img_width
        height = self.height / img_height
        return (x_center, y_center, width, height)
    
    def to_coco_format(self) -> Tuple[int, int, int, int]:
        """Convert to COCO format (x, y, width, height) in pixels"""
        return (self.x_min, self.y_min, self.width, self.height)
    
    def contains_point(self, point: Point) -> bool:
        """Check if point is inside bounding box"""
        return (self.x_min <= point.x <= self.x_max and 
                self.y_min <= point.y <= self.y_max)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bbox"""
        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)
        
        if x_min >= x_max or y_min >= y_max:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'BoundingBox':
        return cls(
            x_min=d["x_min"],
            y_min=d["y_min"],
            x_max=d["x_max"],
            y_max=d["y_max"]
        )


@dataclass
class LineSegment:
    """Straight line segment"""
    start: Point
    end: Point
    
    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)
    
    @property
    def angle(self) -> float:
        """Angle in degrees from horizontal"""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return math.degrees(math.atan2(dy, dx))
    
    @property
    def is_horizontal(self) -> bool:
        """Within 10 degrees of horizontal"""
        return abs(self.angle) < 10 or abs(self.angle) > 170
    
    @property
    def is_vertical(self) -> bool:
        """Within 10 degrees of vertical"""
        angle = abs(self.angle)
        return 80 < angle < 100
    
    @property
    def midpoint(self) -> Point:
        return Point(
            x=(self.start.x + self.end.x) // 2,
            y=(self.start.y + self.end.y) // 2
        )
    
    def to_dict(self) -> Dict:
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict()
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'LineSegment':
        return cls(
            start=Point.from_dict(d["start"]),
            end=Point.from_dict(d["end"])
        )


@dataclass
class TerminalInfo:
    """Terminal/pin information with position"""
    
    name: str                        # "pin1", "drain", "gate", etc.
    position: Point                  # Pixel coordinates
    connected_to_node: str = ""      # Node this terminal connects to
    connected_via_wire: Optional[str] = None  # Wire ID if applicable
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "position": self.position.to_dict(),
            "connected_to_node": self.connected_to_node,
            "connected_via_wire": self.connected_via_wire
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TerminalInfo':
        return cls(
            name=d["name"],
            position=Point.from_dict(d["position"]),
            connected_to_node=d.get("connected_to_node", ""),
            connected_via_wire=d.get("connected_via_wire")
        )


@dataclass
class ComponentInfo:
    """Detailed component information with spatial data"""
    
    # Identity
    id: str                          # "R1", "M1", etc.
    type: str                        # "Resistor", "NMOS", "PMOS", etc.
    value: Optional[str] = None      # "10k", "1u", etc.
    
    # Spatial (pixel coordinates)
    bbox: Optional[BoundingBox] = None
    center: Optional[Point] = None
    orientation: str = "horizontal"  # "horizontal", "vertical", "diagonal"
    
    # Terminals with positions
    terminals: List[TerminalInfo] = field(default_factory=list)
    
    # Refinement info
    source: str = "yolo"             # "yolo", "agent_added", "agent_corrected"
    original_detection: Optional[Dict] = None  # Original YOLO detection
    confidence: float = 0.0
    refinement_notes: str = ""       # What agent did to this component
    
    # Connection info (derived)
    connected_nodes: Dict[str, str] = field(default_factory=dict)  # {terminal: node}
    
    def get_terminal(self, name: str) -> Optional[TerminalInfo]:
        """Get terminal by name"""
        for terminal in self.terminals:
            if terminal.name == name:
                return terminal
        return None
    
    def get_connected_nodes(self) -> List[str]:
        """Get list of all connected nodes"""
        return list(set(t.connected_to_node for t in self.terminals if t.connected_to_node))
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "value": self.value,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "center": self.center.to_dict() if self.center else None,
            "orientation": self.orientation,
            "terminals": [t.to_dict() for t in self.terminals],
            "source": self.source,
            "original_detection": self.original_detection,
            "confidence": self.confidence,
            "refinement_notes": self.refinement_notes,
            "connected_nodes": self.connected_nodes
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ComponentInfo':
        return cls(
            id=d["id"],
            type=d["type"],
            value=d.get("value"),
            bbox=BoundingBox.from_dict(d["bbox"]) if d.get("bbox") else None,
            center=Point.from_dict(d["center"]) if d.get("center") else None,
            orientation=d.get("orientation", "horizontal"),
            terminals=[TerminalInfo.from_dict(t) for t in d.get("terminals", [])],
            source=d.get("source", "yolo"),
            original_detection=d.get("original_detection"),
            confidence=d.get("confidence", 0.0),
            refinement_notes=d.get("refinement_notes", ""),
            connected_nodes=d.get("connected_nodes", {})
        )

