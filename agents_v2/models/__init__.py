"""
Data models for Agents V2 Pipeline
"""

from .components import (
    Point, BoundingBox, LineSegment,
    TerminalInfo, ComponentInfo
)
from .wires import (
    WireInfo, SegmentAnnotation, JunctionInfo
)
from .nodes import (
    NodeInfo, GraphNode, GraphEdge, TopologyAnnotation
)
from .difficulty import (
    DifficultyFactors, DifficultyScores
)
from .training_data import (
    ComponentAnnotation, WireAnnotation, NodeAnnotation,
    TerminalAnnotation, RefinementDeltas, ChangeLogEntry,
    TrainingDataSample
)
from .output import (
    StageResult, IterationRecord, RawCVOutput,
    NetlistOutput, PipelineOutput
)

__all__ = [
    # Basic types
    "Point", "BoundingBox", "LineSegment",
    # Components
    "TerminalInfo", "ComponentInfo",
    # Wires
    "WireInfo", "SegmentAnnotation", "JunctionInfo",
    # Nodes
    "NodeInfo", "GraphNode", "GraphEdge", "TopologyAnnotation",
    # Difficulty
    "DifficultyFactors", "DifficultyScores",
    # Training data
    "ComponentAnnotation", "WireAnnotation", "NodeAnnotation",
    "TerminalAnnotation", "RefinementDeltas", "ChangeLogEntry",
    "TrainingDataSample",
    # Output
    "StageResult", "IterationRecord", "RawCVOutput",
    "NetlistOutput", "PipelineOutput"
]

