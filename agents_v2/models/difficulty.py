"""
Difficulty scoring for curriculum learning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class DifficultyFactors:
    """Factors that contribute to sample difficulty"""
    
    # Density factors
    component_count: int = 0
    component_density: float = 0.0    # Components per 1000 sq pixels
    wire_count: int = 0
    wire_density: float = 0.0
    junction_count: int = 0
    
    # Complexity factors
    crossing_count: int = 0           # Wire crossings (not junctions)
    overlapping_components: int = 0   # Components that touch/overlap
    multi_terminal_components: int = 0  # Components with >2 terminals (MOSFETs, etc.)
    
    # Noise/quality factors
    image_quality_score: float = 1.0  # 0-1, lower = worse quality
    ocr_text_density: float = 0.0     # Amount of text in image
    line_noise_ratio: float = 0.0     # Hough noise lines / total lines
    
    # Detection difficulty (from CV performance)
    yolo_avg_confidence: float = 0.0  # Higher = easier
    yolo_miss_rate: float = 0.0       # Agent additions / final count
    yolo_fp_rate: float = 0.0         # Agent rejections / yolo count
    
    # Connectivity difficulty
    avg_node_degree: float = 0.0      # Higher = more complex
    max_node_degree: int = 0
    floating_terminals_before: int = 0  # Before agent refinement
    
    # Agent effort (proxy for difficulty)
    agent1_iterations: int = 1
    agent2_iterations: int = 1
    agent3_iterations: int = 1
    total_llm_calls: int = 0
    
    # Refinement metrics
    components_added: int = 0
    components_rejected: int = 0
    wires_filtered: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "component_count": self.component_count,
            "component_density": self.component_density,
            "wire_count": self.wire_count,
            "wire_density": self.wire_density,
            "junction_count": self.junction_count,
            "crossing_count": self.crossing_count,
            "overlapping_components": self.overlapping_components,
            "multi_terminal_components": self.multi_terminal_components,
            "image_quality_score": self.image_quality_score,
            "ocr_text_density": self.ocr_text_density,
            "line_noise_ratio": self.line_noise_ratio,
            "yolo_avg_confidence": self.yolo_avg_confidence,
            "yolo_miss_rate": self.yolo_miss_rate,
            "yolo_fp_rate": self.yolo_fp_rate,
            "avg_node_degree": self.avg_node_degree,
            "max_node_degree": self.max_node_degree,
            "floating_terminals_before": self.floating_terminals_before,
            "agent1_iterations": self.agent1_iterations,
            "agent2_iterations": self.agent2_iterations,
            "agent3_iterations": self.agent3_iterations,
            "total_llm_calls": self.total_llm_calls,
            "components_added": self.components_added,
            "components_rejected": self.components_rejected,
            "wires_filtered": self.wires_filtered
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'DifficultyFactors':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DifficultyScores:
    """Multi-dimensional difficulty scoring for curriculum learning"""
    
    # Overall difficulty (weighted combination)
    overall_difficulty: float = 0.0   # 0.0 (easy) to 1.0 (hard)
    
    # Per-task difficulty
    component_difficulty: float = 0.0  # How hard is component detection?
    wire_difficulty: float = 0.0       # How hard is wire detection/tracing?
    node_difficulty: float = 0.0       # How hard is node detection?
    topology_difficulty: float = 0.0   # How hard is understanding connections?
    
    # Curriculum stage assignment
    curriculum_stage: int = 1          # 1 (easiest) to 5 (hardest)
    
    # EMA tracking (updated during training)
    ema_loss_component: Optional[float] = None
    ema_loss_wire: Optional[float] = None
    ema_loss_node: Optional[float] = None
    ema_loss_overall: Optional[float] = None
    
    # Raw factors
    factors: Optional[DifficultyFactors] = None
    
    @classmethod
    def compute(cls, factors: DifficultyFactors, 
                weights: Optional[Dict[str, float]] = None,
                stage_thresholds: Optional[List[float]] = None) -> 'DifficultyScores':
        """Compute difficulty scores from factors"""
        
        if weights is None:
            weights = {
                "component_count": 0.08,
                "component_density": 0.08,
                "wire_density": 0.08,
                "crossing_count": 0.10,
                "yolo_miss_rate": 0.12,
                "yolo_fp_rate": 0.08,
                "agent_iterations": 0.15,
                "line_noise_ratio": 0.10,
                "floating_terminals": 0.08,
                "multi_terminal": 0.08,
                "image_quality": 0.05
            }
        
        if stage_thresholds is None:
            stage_thresholds = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Compute component difficulty
        component_diff = cls._normalize(
            factors.component_count * 0.3 +
            factors.yolo_miss_rate * 30 +
            factors.yolo_fp_rate * 20 +
            (1 - factors.yolo_avg_confidence) * 10 +
            factors.multi_terminal_components * 0.5
        )
        
        # Compute wire difficulty
        wire_diff = cls._normalize(
            factors.wire_count * 0.1 +
            factors.line_noise_ratio * 30 +
            factors.crossing_count * 2 +
            factors.wires_filtered * 0.2
        )
        
        # Compute node difficulty
        node_diff = cls._normalize(
            factors.junction_count * 0.3 +
            factors.crossing_count * 1.5 +
            factors.floating_terminals_before * 2 +
            factors.avg_node_degree * 0.5
        )
        
        # Compute topology difficulty
        topo_diff = cls._normalize(
            factors.component_count * 0.2 +
            factors.avg_node_degree * 1.0 +
            factors.max_node_degree * 0.3 +
            factors.crossing_count * 1.0 +
            (factors.agent1_iterations + factors.agent2_iterations) * 0.3
        )
        
        # Overall difficulty (weighted average)
        overall = (
            component_diff * 0.30 +
            wire_diff * 0.25 +
            node_diff * 0.20 +
            topo_diff * 0.25
        )
        
        # Add penalty for agent effort
        agent_effort = (
            factors.agent1_iterations + 
            factors.agent2_iterations + 
            factors.agent3_iterations
        ) / 9.0  # Normalize by max (3+3+3)
        overall = min(1.0, overall + agent_effort * 0.15)
        
        # Determine curriculum stage
        stage = 1
        for i, threshold in enumerate(stage_thresholds):
            if overall <= threshold:
                stage = i + 1
                break
        
        return cls(
            overall_difficulty=overall,
            component_difficulty=component_diff,
            wire_difficulty=wire_diff,
            node_difficulty=node_diff,
            topology_difficulty=topo_diff,
            curriculum_stage=stage,
            factors=factors
        )
    
    @staticmethod
    def _normalize(value: float, max_value: float = 20.0) -> float:
        """Normalize a value to 0-1 range using sigmoid-like function"""
        return 1.0 / (1.0 + math.exp(-value / max_value * 4 + 2))
    
    def update_ema(self, loss_component: float, loss_wire: float, 
                   loss_node: float, decay: float = 0.9):
        """Update EMA loss values during training"""
        if self.ema_loss_component is None:
            self.ema_loss_component = loss_component
            self.ema_loss_wire = loss_wire
            self.ema_loss_node = loss_node
        else:
            self.ema_loss_component = decay * self.ema_loss_component + (1 - decay) * loss_component
            self.ema_loss_wire = decay * self.ema_loss_wire + (1 - decay) * loss_wire
            self.ema_loss_node = decay * self.ema_loss_node + (1 - decay) * loss_node
        
        self.ema_loss_overall = (
            self.ema_loss_component * 0.4 +
            self.ema_loss_wire * 0.3 +
            self.ema_loss_node * 0.3
        )
    
    def to_dict(self) -> Dict:
        return {
            "overall_difficulty": self.overall_difficulty,
            "component_difficulty": self.component_difficulty,
            "wire_difficulty": self.wire_difficulty,
            "node_difficulty": self.node_difficulty,
            "topology_difficulty": self.topology_difficulty,
            "curriculum_stage": self.curriculum_stage,
            "ema_loss_component": self.ema_loss_component,
            "ema_loss_wire": self.ema_loss_wire,
            "ema_loss_node": self.ema_loss_node,
            "ema_loss_overall": self.ema_loss_overall,
            "factors": self.factors.to_dict() if self.factors else None
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'DifficultyScores':
        return cls(
            overall_difficulty=d.get("overall_difficulty", 0.0),
            component_difficulty=d.get("component_difficulty", 0.0),
            wire_difficulty=d.get("wire_difficulty", 0.0),
            node_difficulty=d.get("node_difficulty", 0.0),
            topology_difficulty=d.get("topology_difficulty", 0.0),
            curriculum_stage=d.get("curriculum_stage", 1),
            ema_loss_component=d.get("ema_loss_component"),
            ema_loss_wire=d.get("ema_loss_wire"),
            ema_loss_node=d.get("ema_loss_node"),
            ema_loss_overall=d.get("ema_loss_overall"),
            factors=DifficultyFactors.from_dict(d["factors"]) if d.get("factors") else None
        )

