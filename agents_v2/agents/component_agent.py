"""
Component Refinement Agent (Agent 1) - SIMPLIFIED

LLM only validates YOLO detections. No coordinate generation.
All spatial data comes from YOLO.
"""

import re
import json
from typing import Dict, Any, List

from .base_agent import BaseReActAgent, AgentContext
from ..models.components import ComponentInfo, BoundingBox, Point, TerminalInfo
from ..config import PipelineConfig


class ComponentRefinementAgent(BaseReActAgent):
    """
    Agent 1: Component Validation
    
    LLM ONLY outputs: which YOLO detections to keep/reject
    ALL coordinates come from YOLO, never from LLM
    """
    
    def __init__(self, model: str, config: PipelineConfig):
        super().__init__(
            agent_name="component_refinement",
            model=model,
            config=config,
            requires_vision=True
        )
    
    def get_max_iterations(self) -> int:
        return 2
    
    def get_pass_threshold(self) -> float:
        return 0.70
    
    def get_min_viable_threshold(self) -> float:
        return 0.50
    
    def get_system_prompt(self) -> str:
        return """You validate YOLO detections from circuit schematics.

For each detection:
- "keep": true/false - is this a real component?
- "type": the CORRECT component type (may differ from YOLO's guess)
- "corrected": true if you changed the type from YOLO's classification

PAY ATTENTION TO CONFIDENCE:
- High confidence (>0.8): YOLO is likely correct, verify type
- Medium confidence (0.5-0.8): Check carefully, may be misclassified
- Low confidence (<0.5): YOLO is uncertain - use your judgment to correct

Common misclassifications:
- MOSFET vs Capacitor (similar shapes)
- Resistor vs Inductor (zigzag vs coil)
- Ground vs generic symbol
- Current_Source vs Voltage_Source

Output JSON:
```json
{
  "decisions": [
    {"idx": 0, "keep": true, "type": "Resistor", "corrected": false},
    {"idx": 1, "keep": true, "type": "NMOS", "corrected": true, "yolo_said": "Capacitor", "reason": "shape is clearly a MOSFET"},
    {"idx": 2, "keep": false, "reason": "text label, not a component"},
    {"idx": 3, "keep": true, "type": "Ground", "corrected": false}
  ],
  "missed": [
    {"type": "Ground", "location": "bottom center", "reason": "visible ground symbol not detected"}
  ]
}
```

DO NOT generate coordinates. Just validate/correct types."""
    
    def build_user_prompt(self, context: AgentContext, iteration: int) -> str:
        yolo_detections = context.original_input.get("yolo_detections", [])
        
        prompt = f"Validate these {len(yolo_detections)} YOLO detections:\n\n"
        
        for i, det in enumerate(yolo_detections):
            conf = det.get('confidence', 0)
            cls = det.get('class', '?')
            
            # Confidence indicator
            if conf >= 0.8:
                conf_indicator = "✓ HIGH"
            elif conf >= 0.5:
                conf_indicator = "⚠ MEDIUM"
            else:
                conf_indicator = "❌ LOW - likely wrong"
            
            prompt += f"[{i}] {cls} (conf: {conf:.2f}) {conf_indicator}\n"
        
        if iteration > 1 and context.errors:
            prompt += f"\n\nFix these issues:\n"
            for e in context.errors:
                prompt += f"- {e}\n"
        
        prompt += "\n\nFor LOW confidence detections, check if the type is correct and correct if needed."
        
        return prompt
    
    def parse_response(self, response: str, context: AgentContext) -> Dict[str, Any]:
        """Parse LLM decisions and build components from YOLO"""
        
        yolo_detections = context.original_input.get("yolo_detections", [])
        
        # Try to extract JSON object first (new format with missed), then array
        decisions = []
        missed = []
        
        json_obj_match = re.search(r'\{[\s\S]*\}', response)
        if json_obj_match:
            try:
                data = json.loads(json_obj_match.group(0))
                decisions = data.get("decisions", [])
                missed = data.get("missed", [])
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
        
        # If no valid decisions, just keep all YOLO detections
        if not decisions:
            print(f"    No valid decisions, keeping all {len(yolo_detections)} YOLO detections")
            decisions = [{"idx": i, "keep": True, "type": d.get("class", "Unknown")} 
                        for i, d in enumerate(yolo_detections)]
        
        # Build components from YOLO data + LLM decisions
        components = []
        rejected = []
        corrected = []
        
        for dec in decisions:
            idx = dec.get("idx", dec.get("index", -1))
            keep = dec.get("keep", True)
            
            if idx < 0 or idx >= len(yolo_detections):
                continue
            
            yolo = yolo_detections[idx]
            yolo_type = yolo.get("class", "Unknown")
            yolo_conf = yolo.get("confidence", 0)
            
            if not keep:
                rejected.append({
                    "idx": idx, 
                    "reason": dec.get("reason", "rejected"),
                    "yolo_type": yolo_type,
                    "yolo_conf": yolo_conf
                })
                continue
            
            # Build component from YOLO coordinates
            bbox = BoundingBox(
                x_min=int(yolo.get("x_min", 0)),
                y_min=int(yolo.get("y_min", 0)),
                x_max=int(yolo.get("x_max", 0)),
                y_max=int(yolo.get("y_max", 0))
            )
            
            # Check if type was corrected
            comp_type = dec.get("type", yolo_type)
            was_corrected = dec.get("corrected", False) or (comp_type != yolo_type)
            
            if was_corrected:
                corrected.append({
                    "idx": idx,
                    "yolo_type": yolo_type,
                    "corrected_type": comp_type,
                    "yolo_conf": yolo_conf,
                    "reason": dec.get("reason", "type corrected by LLM")
                })
                print(f"    🔄 Corrected [{idx}]: {yolo_type} → {comp_type} (was {yolo_conf:.2f} conf)")
            
            terminals = self._auto_terminals(comp_type, bbox)
            
            comp = ComponentInfo(
                id=f"{self._prefix(comp_type)}{len(components)+1}",
                type=comp_type,
                bbox=bbox,
                center=bbox.center,
                orientation="vertical" if bbox.height > bbox.width else "horizontal",
                terminals=terminals,
                source="yolo_corrected" if was_corrected else "yolo",
                original_detection=yolo,
                confidence=float(yolo_conf),
                refinement_notes=dec.get("reason", "") if was_corrected else ""
            )
            components.append(comp)
        
        # Log missed components (flagged by LLM but no coordinates)
        if missed:
            print(f"    ⚠ LLM flagged {len(missed)} potentially missed components:")
            for m in missed:
                print(f"      - {m.get('type', '?')} at {m.get('location', '?')}: {m.get('reason', '')}")
        
        # Summary
        print(f"    ✓ Kept {len(components)}, rejected {len(rejected)}, corrected {len(corrected)}, flagged {len(missed)} missed")
        
        return {
            "components": components,
            "rejected": rejected,
            "corrected": corrected,  # Type corrections made by LLM
            "missed_flagged": missed,  # Components LLM thinks YOLO missed
            "actions": [
                f"Kept {len(components)} components",
                f"Rejected {len(rejected)} false positives",
                f"Corrected {len(corrected)} misclassifications",
                f"Flagged {len(missed)} potentially missed"
            ]
        }
    
    def _prefix(self, comp_type: str) -> str:
        prefixes = {"Resistor": "R", "Capacitor": "C", "NMOS": "M", "PMOS": "M", 
                   "MOSFET": "M", "Ground": "GND", "Current_Source": "I", "Voltage_Source": "V"}
        return prefixes.get(comp_type, "X")
    
    def _auto_terminals(self, comp_type: str, bbox: BoundingBox) -> List[TerminalInfo]:
        """Auto-generate terminal positions from bbox"""
        cx, cy = bbox.center.x, bbox.center.y
        
        if comp_type in ["NMOS", "PMOS", "MOSFET"]:
            return [
                TerminalInfo(name="drain", position=Point(x=cx, y=bbox.y_min)),
                TerminalInfo(name="gate", position=Point(x=bbox.x_min, y=cy)),
                TerminalInfo(name="source", position=Point(x=cx, y=bbox.y_max)),
            ]
        elif comp_type in ["Ground", "GND", "Current_Source", "Voltage_Source"]:
            return [TerminalInfo(name="terminal", position=Point(x=cx, y=bbox.y_min))]
        else:
            # 2-terminal default
            if bbox.height > bbox.width:
                return [
                    TerminalInfo(name="pin1", position=Point(x=cx, y=bbox.y_min)),
                    TerminalInfo(name="pin2", position=Point(x=cx, y=bbox.y_max)),
                ]
            else:
                return [
                    TerminalInfo(name="pin1", position=Point(x=bbox.x_min, y=cy)),
                    TerminalInfo(name="pin2", position=Point(x=bbox.x_max, y=cy)),
                ]
    
    def evaluate_metrics(self, output: Dict[str, Any], context: AgentContext) -> Dict[str, float]:
        yolo_detections = context.original_input.get("yolo_detections", [])
        yolo_count = len(yolo_detections)
        components = output.get("components", [])
        rejected = output.get("rejected", [])
        corrected = output.get("corrected", [])
        
        processed = len(components) + len(rejected)
        
        metrics = {}
        
        # 1. Coverage: Did we process all detections?
        metrics["coverage"] = processed / max(yolo_count, 1)

        # 2. Engagement: Did the agent actively validate detections?
        # Rewards: corrections made (any confidence) + low-conf items caught
        # Penalizes: zero engagement on a batch that had uncertain detections
        low_conf_items = [d for d in yolo_detections if d.get('confidence', 1.0) < 0.7]
        n_low_conf = len(low_conf_items)

        if n_low_conf == 0:
            # All detections were high-confidence — pass-through is acceptable
            metrics["smart_corrections"] = 1.0
        else:
            # At least some uncertain detections exist — agent should have engaged
            # Count how many low-conf items were corrected or rejected (i.e. acted on)
            low_conf_ids = {i for i, d in enumerate(yolo_detections) if d.get('confidence', 1.0) < 0.7}
            corrected_ids = {c.get('idx', -1) for c in corrected}
            rejected_ids = {r.get('idx', -1) for r in rejected}
            acted_on = low_conf_ids & (corrected_ids | rejected_ids)
            # Partial credit: any corrections at all (even high-conf) show the agent looked
            any_corrections = len(corrected) > 0 or len(rejected) > 0
            engagement = len(acted_on) / n_low_conf
            metrics["smart_corrections"] = engagement if engagement > 0 else (0.5 if any_corrections else 0.0)

        # 3. Rejection sanity: Rejection rate should be reasonable (0-50%)
        rejection_rate = len(rejected) / max(yolo_count, 1)
        if rejection_rate <= 0.5:
            metrics["rejection_sanity"] = 1.0
        else:
            metrics["rejection_sanity"] = max(0, 1.0 - (rejection_rate - 0.5) * 2)
        
        # 5. Type diversity: Circuit should have realistic component mix
        types = set(comp.type if hasattr(comp, 'type') else comp.get('type', '') for comp in components)
        has_active = bool(types & {'NMOS', 'PMOS', 'MOSFET', 'NPN', 'PNP', 'OpAmp'})
        has_passive = bool(types & {'Resistor', 'Capacitor', 'Inductor'})
        has_reference = bool(types & {'Ground', 'VDD', 'Voltage_Source', 'Current_Source'})
        
        # Most circuits need at least ground/power reference
        metrics["type_completeness"] = (0.5 if has_reference else 0) + (0.25 if has_active else 0.25) + (0.25 if has_passive else 0.25)
        
        return metrics
    
    def build_error_context(self, output: Dict[str, Any], metrics: Dict[str, float], context: AgentContext) -> AgentContext:
        new_ctx = AgentContext(original_input=context.original_input)
        errors = []
        
        if metrics.get("coverage", 0) < 0.9:
            errors.append(f"Process all {len(context.original_input.get('yolo_detections', []))} detections")
        
        if metrics.get("terminal_validity", 0) < 0.8:
            errors.append("Some components have wrong terminal count - check component types")
        
        if metrics.get("type_completeness", 0) < 0.6:
            errors.append("Circuit missing essential components (ground/power reference)")
        
        if metrics.get("rejection_sanity", 0) < 0.7:
            errors.append("Too many rejections - verify each rejection is justified")
        
        new_ctx.errors = errors
        new_ctx.previous_output = output
        return new_ctx
    
    def _get_metric_weights(self) -> Dict[str, float]:
        return {
            "coverage": 0.50,           # Process all detections
            "smart_corrections": 0.25,  # Agent engaged with uncertain detections
            "rejection_sanity": 0.25    # Reasonable rejection rate
        }
    
    def _get_metric_target(self, metric: str) -> float:
        return {
            "coverage": 0.95,
            "terminal_validity": 0.90,
            "smart_corrections": 0.70,
            "rejection_sanity": 0.80,
            "type_completeness": 0.75
        }.get(metric, 0.7)
