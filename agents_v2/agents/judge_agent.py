"""
LLM-as-Judge: Cross-Validation Agent (MANDATORY)

Performs semantic validation between:
- Original topology from Agent 2
- Generated SPICE netlist from Agent 3

Uses GPT-5.2-pro for highest reasoning capability.
"""

import json
import re
import os
import time
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple

from openai import OpenAI
from PIL import Image as PILImage

from ..config import PipelineConfig
from ..models.components import ComponentInfo
from ..models.nodes import NodeInfo


_SCAFFOLDING_PREFIXES = (
    "VSUP_", "VB_", "VTEST_", "ISUP_", "IBIAS_", "ITEST_",
    "RSHUNT_", "RPROBE_",
)
_SCAFFOLDING_EXACT = {"VDD", "VSS", "VCC", "VEE", "AVDD", "DVDD"}
_SCAFFOLDING_SINGLE_LETTER = set("BGEFDH")  # behavioral / controlled sources


def is_scaffolding_component(token: str) -> bool:
    """Return True if this netlist component ID is a known scaffolding element."""
    upper = token.upper()
    if upper in _SCAFFOLDING_EXACT:
        return True
    if any(upper.startswith(p) for p in _SCAFFOLDING_PREFIXES):
        return True
    if len(token) >= 1 and token[0].upper() in _SCAFFOLDING_SINGLE_LETTER:
        return True
    return False


def translate_spice_errors(logs: str) -> List[str]:
    """Convert common ngspice error strings into human-readable fix suggestions."""
    fixes = []
    log_lower = logs.lower()
    if "unknown parameter" in log_lower or "unknown device" in log_lower:
        fixes.append("Unknown component type — check model cards and component prefixes.")
    if "node is floating" in log_lower or "floating node" in log_lower:
        fixes.append("Floating node detected — every node must connect to at least two components.")
    if "singular matrix" in log_lower:
        fixes.append("Singular matrix — likely a floating node or disconnected subcircuit.")
    if "undefined node" in log_lower:
        fixes.append("Undefined node name — check for typos in node labels.")
    if "no dc path" in log_lower:
        fixes.append("No DC path to ground — ensure all nodes have a path to node 0.")
    return fixes


class LLMJudge:
    """
    LLM-as-Judge for mandatory cross-validation.
    
    This is NOT a ReAct agent - it's a single-pass evaluator
    that produces a judgment with issues.
    """
    
    def __init__(self, model: str = "gpt-5.2", config: Optional[PipelineConfig] = None):
        self.model = model
        self.config = config or PipelineConfig()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=api_key)
        self.total_tokens = 0
        self.llm_call_history: List[Dict] = []
    
    def judge(
        self,
        topology_components: List[Any],
        topology_nodes: List[Any],
        spice_code: str,
        retry_on_fail: bool = True,
        simulation_logs: str = "",
        simulation_success: bool = False,
        image: Optional[PILImage.Image] = None,
        corrected: Optional[List[Dict]] = None,
        agent_added: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Judge whether the SPICE netlist correctly represents the topology.

        Args:
            topology_components: Validated components from Agent 1
            topology_nodes: (unused, kept for API compat)
            spice_code: SPICE netlist from Agent 2
            simulation_logs: ngspice stdout/stderr
            simulation_success: ngspice pass/fail (objective)
            image: Original circuit image for visual topology verification
            corrected: List of type corrections Agent 1 made
            agent_added: List of components Agent 1 added (missed by YOLO)

        Returns:
            {
                "verdict": "PASS" | "FAIL",
                "confidence": 0.0-1.0,  # criteria-based, not LLM vibe
                "criteria": {...},       # C1-C5 individual results
                "component_mapping": {...},
                "node_mapping": {...},
                "issues": [...],
                "explanation": "..."
            }
        """
        
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_user_prompt(
            topology_components, topology_nodes, spice_code,
            simulation_logs=simulation_logs,
            simulation_success=simulation_success,
            corrected=corrected or [],
            agent_added=agent_added or [],
        )
        
        # Build user message content — text + optional image
        user_content: List[Any] = []
        if image is not None:
            # Cap at 1024px longest side — sufficient for vision, avoids token explosion
            _img = image.copy()
            if max(_img.size) > 1024:
                _img.thumbnail((1024, 1024))
            if _img.mode in ("RGBA", "P"):
                _img = _img.convert("RGB")
            buffered = BytesIO()
            _img.save(buffered, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        user_content.append({"type": "text", "text": user_prompt})

        max_attempts = self.config.iterations.judge_max_retries + 1 if retry_on_fail else 1

        for attempt in range(max_attempts):
            try:
                # Build API params - handle different parameter names
                api_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0.1  # Low temperature for consistent judgment
                }
                
                # Newer models use max_completion_tokens
                if any(x in self.model.lower() for x in ["o1", "o3", "o4", "gpt-5", "5.2"]):
                    api_params["max_completion_tokens"] = 2048
                else:
                    api_params["max_tokens"] = 2048
                
                response = self.client.chat.completions.create(**api_params)
                
                self.total_tokens += response.usage.total_tokens if response.usage else 0
                content = response.choices[0].message.content or ""

                # Log this LLM call for debugging/analysis
                cache_stats = {}
                if response.usage and hasattr(response.usage, 'prompt_tokens_details'):
                    details = response.usage.prompt_tokens_details
                    cache_stats = {
                        'total_tokens': response.usage.total_tokens,
                        'cached_tokens': getattr(details, 'cached_tokens', 0),
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens
                    }

                self.llm_call_history.append({
                    "iteration": attempt + 1,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "model": self.model,
                    "prompt": {
                        "system": system_prompt,
                        "user": user_prompt
                    },
                    "response": content,
                    "tokens": cache_stats if cache_stats else {"total_tokens": response.usage.total_tokens if response.usage else 0},
                    "retry_attempt": attempt
                })

                result = self._parse_judgment(content, simulation_success)

                # Validate result
                if "verdict" in result and "confidence" in result:
                    return result

            except Exception as e:
                if attempt == max_attempts - 1:
                    return {
                        "verdict": "FAIL",
                        "confidence": 0.0,
                        "criteria": {},
                        "issues": [f"Judge error: {str(e)}"],
                        "explanation": "Failed to evaluate due to error"
                    }

        return {
            "verdict": "FAIL",
            "confidence": 0.0,
            "criteria": {},
            "issues": ["Failed to produce valid judgment"],
            "explanation": "Judge could not evaluate the netlist"
        }
    
    def _get_system_prompt(self) -> str:
        return """You are an expert circuit verification engineer evaluating an auto-generated SPICE netlist.

**PIPELINE CONTEXT**:
1. Agent 1 (vision) validated YOLO detections → produced component list with IDs (M1, R1, etc.), corrected types, and flagged missed components
2. Agent 2 (vision) looked at the circuit image → generated SPICE netlist inferring connectivity
3. You verify both using 5 explicit criteria (C1–C5) and provide targeted feedback

**SPICE LINE STRUCTURE**:
- First token on each line = component ID (e.g. M1, R1, V1)
- Remaining tokens = node names (vdd, gnd, 0, in, out...) — NEVER treat these as components
- Example: `V1 vdd 0 DC 1.8` → ID=V1, nodes=vdd,0. Do NOT flag vdd as a hallucination.

**SCAFFOLDING (DO NOT FLAG)**:
- Supply names: VDD, VSS, VCC, VEE, AVDD, DVDD
- Prefixes: VSUP_*, VB_*, VTEST_*, ISUP_*, IBIAS_*, ITEST_*, RSHUNT_*, RPROBE_*, B*, G*, E*, F*, H*

**GROUND HANDLING**:
- Ground components are implicit in SPICE as node "0" — mark them as IMPLICIT, never expect a component line for them

═══════════════════════════════════════════════
EVALUATION CRITERIA (score each independently)
═══════════════════════════════════════════════

**C1 — Simulation validity** (HARD RULE — objective)
- PASS: ngspice simulation succeeded
- FAIL: ngspice simulation failed for any reason
- If C1=FAIL → overall verdict MUST be FAIL regardless of other criteria

**C2 — Component completeness** (HARD RULE — checkable)
- PASS: every validated component ID (except Ground) appears in the netlist exactly once, no duplicates
- FAIL: any validated component missing OR any ID duplicated
- If C2=FAIL → overall verdict MUST be FAIL regardless of other criteria

**C3 — Component type match** (visual check using the circuit image)
- Look at the circuit image. Do the component symbols you see match the types in the validated list?
- PASS: types are consistent with what you see in the image
- FAIL: clear mismatch (e.g. validated list says NMOS but image shows a resistor symbol in that bbox)
- Note any corrections Agent 1 already made — these are shown explicitly

**C4 — Topological plausibility** (visual check using the circuit image)
- Look at the circuit image. Do the netlist connections make sense given the visual layout?
- Check: power rail at top connects to correct terminals, ground at bottom, signal flow is coherent
- PASS: connections are plausible given the image
- FAIL: clear topological error visible (e.g. drain connected to ground when it clearly goes to VDD)
- Be lenient — you cannot verify every wire, only obvious errors

**C5 — No hallucinations** (checkable)
- PASS: no non-scaffolding component IDs in the netlist that are not in the validated list
- FAIL: extra non-scaffolding components invented by Agent 2

═══════════════════════════════════════════════
CONFIDENCE SCORING
═══════════════════════════════════════════════
Confidence is computed from C3+C4+C5 (the LLM-evaluated criteria):
- All 3 pass → confidence = 1.0
- 2 pass → confidence = 0.8
- 1 passes → confidence = 0.5
- 0 pass → confidence = 0.2
Do NOT invent your own confidence number — use the formula above.

Overall PASS requires: C1=PASS AND C2=PASS AND confidence ≥ 0.80

═══════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════
```json
{
  "thinking": "Step by step: check C1, C2, then look at image for C3/C4, then C5...",

  "criteria": {
    "C1_simulation": {"pass": true, "notes": "ngspice passed"},
    "C2_completeness": {"pass": true, "notes": "all IDs present exactly once"},
    "C3_type_match": {"pass": true, "notes": "visual symbols match validated types"},
    "C4_topology": {"pass": true, "notes": "connections consistent with image layout"},
    "C5_no_hallucinations": {"pass": true, "notes": "no extra non-scaffolding components"}
  },

  "verdict": "PASS",
  "confidence": 1.0,

  "component_mapping": {
    "M1": {"netlist": "M1", "status": "MATCH", "notes": ""},
    "GND1": {"netlist": null, "status": "IMPLICIT", "notes": "ground = node 0"}
  },

  "issues": [],

  "explanation": "Brief summary of what passed and what failed.",

  "feedback_for_agent1": "Specific feedback about component detection. Mention: wrong types, missed components, low-confidence items that look suspicious in the image. If all good, say so.",

  "feedback_for_agent2": "Specific feedback about netlist generation. Mention: wrong connections visible in image, missing/duplicate components, simulation errors to fix. If all good, say so."
}
```

**CRITICAL**: C1=FAIL or C2=FAIL → verdict=FAIL, no exceptions. Always fill both feedback fields."""
    
    def _build_user_prompt(
        self,
        components: List[Any],
        nodes: List[Any],
        spice_code: str,
        simulation_logs: str = "",
        simulation_success: bool = False,
        corrected: Optional[List[Dict]] = None,
        agent_added: Optional[List[Dict]] = None,
    ) -> str:
        """Build the verification prompt with component details, correction history, and sim results."""

        sim_status = "PASSED" if simulation_success else "FAILED"
        prompt = f"""Evaluate this circuit netlist using criteria C1–C5. Simulation result (C1) is already known: {sim_status}.

═══════════════════════════════════════════════════════════
ORIGINAL TOPOLOGY
═══════════════════════════════════════════════════════════

COMPONENTS:
"""
        for i, comp in enumerate(components, 1):
            if isinstance(comp, ComponentInfo):
                connections = ", ".join(
                    f"{t.name}→{t.connected_to_node}" 
                    for t in comp.terminals
                )
                prompt += f"  {comp.id} ({comp.type}): {connections}\n"
            elif isinstance(comp, dict):
                prompt += f"{i}. ID: {comp.get('id','?')}, Type: {comp.get('type','?')}"
                if comp.get('source'):
                    prompt += f", Source: {comp['source']}"
                if comp.get('confidence') is not None:
                    prompt += f", Confidence: {comp['confidence']:.2f}"
                if comp.get('refinement_notes'):
                    prompt += f", Notes: {comp['refinement_notes']}"
                prompt += "\n"

        # Agent 1 correction history — useful for C3 visual verification
        if corrected:
            prompt += f"\nAGENT 1 TYPE CORRECTIONS ({len(corrected)}):\n"
            for c in corrected:
                prompt += (f"  - [{c.get('idx','?')}] {c.get('yolo_type','?')} → "
                           f"{c.get('corrected_type','?')} "
                           f"(YOLO conf: {c.get('yolo_conf', 0):.2f}): {c.get('reason','')}\n")

        if agent_added:
            prompt += f"\nCOMPONENTS ADDED BY AGENT 1 (missed by YOLO, {len(agent_added)}):\n"
            for a in agent_added:
                prompt += f"  - {a.get('type','?')} at {a.get('location','?')}: {a.get('reason','')}\n"

        # Parse netlist component IDs
        core_parsed = []
        scaffolding_parsed = []
        for line in spice_code.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("*", ".", ";", "#", "//")):
                continue
            first_token = stripped.split()[0]
            if is_scaffolding_component(first_token):
                scaffolding_parsed.append(first_token)
            else:
                core_parsed.append(first_token)

        prompt += f"""
═══════════════════════════════════════════════════════════
GENERATED SPICE NETLIST:
═══════════════════════════════════════════════════════════

{spice_code}

CORE COMPONENT IDs IN NETLIST (must match validated list):
{', '.join(core_parsed) if core_parsed else 'None'}

SCAFFOLDING ELEMENTS (allowed — do NOT flag):
{', '.join(scaffolding_parsed) if scaffolding_parsed else 'None'}
"""

        # Simulation results
        prompt += f"""
═══════════════════════════════════════════════════════════
C1 — SIMULATION: {sim_status}
═══════════════════════════════════════════════════════════
"""
        if simulation_logs:
            prompt += f"{simulation_logs[:2000]}\n"
            if not simulation_success:
                translations = translate_spice_errors(simulation_logs)
                if translations:
                    prompt += "\nACTIONABLE FIXES:\n"
                    for t in translations:
                        prompt += f"  - {t}\n"

        prompt += """
═══════════════════════════════════════════════════════════

Evaluate C1–C5 as defined in the system prompt.
If an image was provided, use it for C3 (type match) and C4 (topology plausibility).
Compute confidence from C3+C4+C5 using the formula in the system prompt.
Output your judgment as JSON."""

        return prompt
    
    def _parse_judgment(self, response: str, simulation_success: bool = False) -> Dict[str, Any]:
        """Parse judge response, enforce hard C1/C2 rules, compute criteria-based confidence."""

        # Extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                return {
                    "verdict": "FAIL",
                    "confidence": 0.0,
                    "criteria": {},
                    "component_mapping": {},
                    "issues": ["Could not parse structured response"],
                    "explanation": response[:500]
                }

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "verdict": "FAIL",
                "confidence": 0.0,
                "criteria": {},
                "issues": ["Failed to parse judgment JSON"],
                "explanation": response[:500]
            }

        # Extract criteria block
        criteria = data.get("criteria", {})

        # C1: hard rule — use actual ngspice result, not LLM opinion
        c1_pass = simulation_success
        if "C1_simulation" in criteria:
            criteria["C1_simulation"]["pass"] = c1_pass
        else:
            criteria["C1_simulation"] = {"pass": c1_pass, "notes": "ngspice result (objective)"}

        # C2: use LLM judgment on completeness
        c2 = criteria.get("C2_completeness", {})
        c2_pass = bool(c2.get("pass", True))

        # C3, C4, C5: LLM-evaluated criteria
        c3_pass = bool(criteria.get("C3_type_match", {}).get("pass", True))
        c4_pass = bool(criteria.get("C4_topology", {}).get("pass", True))
        c5_pass = bool(criteria.get("C5_no_hallucinations", {}).get("pass", True))

        # Criteria-based confidence from C3+C4+C5 only
        llm_criteria_passed = sum([c3_pass, c4_pass, c5_pass])
        confidence_map = {3: 1.0, 2: 0.8, 1: 0.5, 0: 0.2}
        confidence = confidence_map[llm_criteria_passed]

        # Hard verdict rules: C1 or C2 fail → FAIL regardless
        issues = self._normalize_issue_list(data.get("issues", []))
        if not c1_pass:
            verdict = "FAIL"
            issues.insert(0, "C1 FAIL: ngspice simulation failed (hard rule)")
        elif not c2_pass:
            verdict = "FAIL"
            issues.insert(0, "C2 FAIL: component completeness check failed (hard rule)")
        else:
            verdict = "PASS" if confidence >= 0.80 else "FAIL"

        return {
            "verdict": verdict,
            "confidence": confidence,
            "criteria": criteria,
            "simulation_passed": c1_pass,
            "component_accounting_passed": c2_pass,
            "component_mapping": data.get("component_mapping", {}),
            "issues": issues,
            "explanation": str(data.get("explanation", "")),
            "thinking": data.get("thinking", ""),
            "feedback_for_agent1": str(data.get("feedback_for_agent1", "")),
            "feedback_for_agent2": str(data.get("feedback_for_agent2", "")),
        }

    def _normalize_issue_list(self, issues: Any) -> List[str]:
        """Ensure issues is a flat list of strings."""
        if isinstance(issues, list):
            return [str(i) for i in issues if i]
        if isinstance(issues, str):
            return [issues] if issues else []
        return []
    
    def get_pass_threshold(self) -> float:
        return self.config.thresholds.judge_pass_threshold
    
    def is_pass(self, result: Dict[str, Any]) -> bool:
        """Check if judgment is a pass"""
        return (
            result.get("verdict") == "PASS" and
            result.get("confidence", 0) >= self.get_pass_threshold()
        )
    
    def get_feedback_for_retry(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get structured feedback to send back to Agent 3 for retry"""
        
        issues = result.get("issues", [])
        component_mapping = result.get("component_mapping", {})
        node_mapping = result.get("node_mapping", {})
        
        # Find specific problems
        component_issues = []
        for comp_id, mapping in component_mapping.items():
            if isinstance(mapping, dict) and mapping.get("status") not in ["MATCH", "IMPLICIT", "OK"]:
                component_issues.append(
                    f"{comp_id}: {mapping.get('notes', 'Issue found')}"
                )
        
        node_issues = []
        for node_id, mapping in node_mapping.items():
            if isinstance(mapping, dict) and mapping.get("status") not in ["MATCH", "OK"]:
                node_issues.append(f"{node_id}: mismatch in netlist")
        
        return {
            "verdict": result.get("verdict"),
            "confidence": result.get("confidence"),
            "errors": issues + component_issues + node_issues,
            "feedback": result.get("explanation", ""),
            "needs_retry": not self.is_pass(result)
        }
