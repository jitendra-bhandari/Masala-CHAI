"""
Connectivity Refinement Agent (Agent 2) - SIMPLIFIED

LLM only validates Hough lines. No coordinate generation.
All spatial data comes from Hough transform.
"""

import re
import json
from typing import Dict, Any, List

from .base_agent import BaseReActAgent, AgentContext
from ..models.components import Point
from ..models.wires import WireInfo
from ..models.nodes import NodeInfo
from ..config import PipelineConfig


class ConnectivityRefinementAgent(BaseReActAgent):
    """
    Agent 2: Wire Validation
    
    LLM ONLY outputs: which Hough lines to keep as wires
    ALL coordinates come from Hough, never from LLM
    """
    
    def __init__(self, model: str, config: PipelineConfig):
        super().__init__(
            agent_name="connectivity_refinement",
            model=model,
            config=config,
            requires_vision=True
        )
    
    def get_max_iterations(self) -> int:
        return 2
    
    def get_pass_threshold(self) -> float:
        return 0.65
    
    def get_min_viable_threshold(self) -> float:
        return 0.40
    
    def get_system_prompt(self) -> str:
        return """You validate Hough line detections from circuit schematics.

For each line, decide if it's a wire (connecting components) or noise/edge.

Output JSON with decisions and any missed wires:
```json
{
  "decisions": [
    {"idx": 0, "wire": true},
    {"idx": 1, "wire": true},
    {"idx": 2, "wire": false, "reason": "component edge"}
  ],
  "missed_connections": [
    {"from": "R1.pin2", "to": "M1.gate", "reason": "wire visible but not detected by Hough"}
  ]
}
```

For missed wires: describe what components should connect. DO NOT generate coordinates."""
    
    def build_user_prompt(self, context: AgentContext, iteration: int) -> str:
        hough_lines = context.original_input.get("hough_lines", [])
        components = context.original_input.get("components", [])
        
        prompt = f"Validate {len(hough_lines)} Hough lines.\n\n"
        prompt += f"Components: {len(components)}\n"
        
        for i, comp in enumerate(components):
            if hasattr(comp, 'id'):
                prompt += f"  {comp.id}: {comp.type}\n"
            else:
                prompt += f"  {comp.get('id', '?')}: {comp.get('type', '?')}\n"
        
        prompt += f"\nHough lines:\n"
        for i, line in enumerate(hough_lines):
            s = line.get("start", {})
            e = line.get("end", {})
            prompt += f"[{i}] ({s.get('x',0)},{s.get('y',0)}) → ({e.get('x',0)},{e.get('y',0)})\n"
        
        return prompt
    
    def parse_response(self, response: str, context: AgentContext) -> Dict[str, Any]:
        """Parse wire decisions and build from Hough data"""
        
        hough_lines = context.original_input.get("hough_lines", [])
        
        # Try JSON object first (new format), then array
        decisions = []
        missed_connections = []
        
        json_obj_match = re.search(r'\{[\s\S]*\}', response)
        if json_obj_match:
            try:
                data = json.loads(json_obj_match.group(0))
                decisions = data.get("decisions", [])
                missed_connections = data.get("missed_connections", [])
            except:
                pass
        
        # Fallback to array format
        if not decisions:
            json_arr_match = re.search(r'\[[\s\S]*\]', response)
            if json_arr_match:
                try:
                    decisions = json.loads(json_arr_match.group(0))
                except:
                    decisions = []
        
        # If no valid decisions, keep all lines as wires
        if not decisions:
            print(f"    No valid decisions, keeping all {len(hough_lines)} Hough lines as wires")
            decisions = [{"idx": i, "wire": True} for i in range(len(hough_lines))]
        
        # Build wires from Hough data
        wires = []
        rejected_count = 0
        
        for dec in decisions:
            idx = dec.get("idx", dec.get("index", -1))
            is_wire = dec.get("wire", True)
            
            if idx < 0 or idx >= len(hough_lines):
                continue
            
            if not is_wire:
                rejected_count += 1
                continue
            
            hough = hough_lines[idx]
            start = hough.get("start", {})
            end = hough.get("end", {})
            
            wire = WireInfo(
                id=f"W{len(wires)+1}",
                path=[
                    Point(x=int(start.get("x", 0)), y=int(start.get("y", 0))),
                    Point(x=int(end.get("x", 0)), y=int(end.get("y", 0)))
                ],
                source_hough_ids=[idx],
                source="hough",
                confidence=0.7
            )
            wire.compute_properties()
            wires.append(wire)
        
        # Create basic nodes
        nodes = [
            NodeInfo(id="0", type="ground", is_ground=True),
            NodeInfo(id="VDD", type="power", is_power=True),
        ]
        
        return {
            "wires": wires,
            "nodes": nodes,
            "rejected_count": rejected_count,
            "actions": [f"Created {len(wires)} wires, rejected {rejected_count} lines"]
        }
    
    def evaluate_metrics(self, output: Dict[str, Any], context: AgentContext) -> Dict[str, float]:
        hough_lines = context.original_input.get("hough_lines", [])
        hough_count = len(hough_lines)
        components = context.original_input.get("components", [])
        wires = output.get("wires", [])
        nodes = output.get("nodes", [])
        rejected_count = output.get("rejected_count", 0)
        
        processed = len(wires) + rejected_count
        
        metrics = {}
        
        # 1. Coverage: Did we process all Hough lines?
        metrics["coverage"] = processed / max(hough_count, 1)
        
        # 2. Wire-to-component ratio: Should have ~1-3 wires per component
        comp_count = len(components)
        if comp_count > 0:
            ratio = len(wires) / comp_count
            if 0.5 <= ratio <= 5.0:
                metrics["wire_ratio"] = 1.0
            elif ratio < 0.5:
                metrics["wire_ratio"] = ratio * 2  # Too few wires
            else:
                metrics["wire_ratio"] = max(0.5, 1.0 - (ratio - 5.0) / 10.0)
        else:
            metrics["wire_ratio"] = 0.5
        
        # 3. Node essentials: Must have ground, should have power
        has_ground = any(
            (n.is_ground if hasattr(n, 'is_ground') else n.get('is_ground', False)) or
            (n.id if hasattr(n, 'id') else n.get('id', '')) == '0'
            for n in nodes
        )
        has_power = any(
            (n.is_power if hasattr(n, 'is_power') else n.get('is_power', False)) or
            (n.id if hasattr(n, 'id') else n.get('id', '')) in ['VDD', 'VCC', 'vdd', 'vcc']
            for n in nodes
        )
        metrics["essential_nodes"] = (0.6 if has_ground else 0) + (0.4 if has_power else 0)
        
        # 4. Filtering sanity: Should filter some lines (not all, not none)
        if hough_count > 0:
            filter_rate = rejected_count / hough_count
            if 0.05 <= filter_rate <= 0.8:
                metrics["filter_sanity"] = 1.0
            elif filter_rate < 0.05:
                metrics["filter_sanity"] = 0.8  # Keeping almost all
            else:
                metrics["filter_sanity"] = max(0.3, 1.0 - filter_rate)
        else:
            metrics["filter_sanity"] = 0.5
        
        return metrics
    
    def build_error_context(self, output: Dict[str, Any], metrics: Dict[str, float], context: AgentContext) -> AgentContext:
        new_ctx = AgentContext(original_input=context.original_input)
        errors = []
        
        if metrics.get("essential_nodes", 0) < 0.6:
            errors.append("Missing ground node (id='0')")
        if metrics.get("wire_ratio", 0) < 0.5:
            errors.append("Too few wires for component count")
        if metrics.get("filter_sanity", 0) < 0.5:
            errors.append("Wire filtering seems off - check decisions")
        
        new_ctx.errors = errors
        new_ctx.previous_output = output
        return new_ctx
    
    def _get_metric_weights(self) -> Dict[str, float]:
        return {
            "coverage": 0.25,
            "wire_ratio": 0.25,
            "essential_nodes": 0.30,
            "filter_sanity": 0.20
        }
    
    def _get_metric_target(self, metric: str) -> float:
        return {
            "coverage": 0.90,
            "wire_ratio": 0.80,
            "essential_nodes": 0.80,
            "filter_sanity": 0.70
        }.get(metric, 0.7)
