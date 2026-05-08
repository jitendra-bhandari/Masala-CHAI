# Masala-CHAI: A Large-Scale SPICE Netlist Dataset for Analog Circuits by Harnessing AI

## Abstract
Masala-CHAI is a fully automated framework leveraging large language models (LLMs) to generate Simulation Programs with Integrated Circuit Emphasis (SPICE) netlists. It addresses a long-standing challenge in circuit design automation: automating netlist generation for analog circuits. Automating this workflow could accelerate the creation of fine-tuned LLMs for analog circuit design and verification. In this work, we identify key challenges in automated netlist generation and evaluate multimodal capabilities of state-of-the-art LLMs, particularly GPT-4, in addressing them. We propose a three-step workflow to overcome existing limitations: labeling analog circuits, prompt tuning, and netlist verification. This approach enables end-to-end SPICE netlist generation from circuit schematic images, tackling the persistent challenge of accurate netlist generation. We utilize Masala-CHAI to collect a corpus of 7,500 schematics that span varying complexities in 10 textbooks and benchmark various open source and proprietary LLMs. Models fine-tuned on Masala-CHAI when used in LLM-agentic frameworks such as AnalogCoder achieve a notable **46% improvement in Pass@1 scores**. We open-source our dataset and code for community-driven development.

Full paper: https://arxiv.org/abs/2411.14299

> [!NOTE]
> **May 2026**: V2 multi-agent pipeline now available with ngspice simulation validation and iterative judge feedback. See [Pipeline V2](#pipeline-v2-multi-agent-with-judge) below.

---

## Setup

```bash
git clone <repository_url>
cd repository_name
conda env create -f environment.yml
conda activate autospice_env
pip install ngspice
```

Download Hough Transform model: https://drive.google.com/file/d/1mTwWWSMsYwhJW-GfKKVm21Lm5lVtMJ1y/view?usp=sharing — extract to `./hough/`

---

## Pipeline V1 (Original)

```bash
python main.py \
  --src ./data/sample-images/ \
  --tgt ./sample-output \
  --api_key <openai_api_key>
```

**Outputs per image:** `scanned_circuit.png`, `detected_components.png`, `component_removed_circuit.png`, `nodes_terminals.png`, `rebuilt_circuit.png`, `sample_statistics.json`, `spice.txt`

---

## Pipeline V2 (Multi-Agent with Judge)

V2 replaces the single LLM pass with three specialized agents and an ngspice simulation check:

- **Agent 1** — validates YOLO detections, corrects types, flags missed components
- **Agent 2** — traces connectivity visually, generates SPICE netlist, runs ngspice
- **Judge** — verifies component completeness, checks simulation, sends feedback to agents on failure (up to 2 revisions)

**Run:**
```bash
export OPENAI_API_KEY='your-key'

# Single image
python -m agents_v2.run --image data/sample-images/330.jpg --output agents_v2_output/

# Batch
python -m agents_v2.run --batch data/sample-images/ --output results/ --parallel --workers 4
```

**V2 outputs** (in `agents_v2_output/agents/<name>/`): `.sp` netlist, `_simulation.log`, `_judge_result.json`, `_validated_components.json`, `_llm_flow.md`

---

## Schematic Caption Generation

```bash
python utils/extract_page.py <path_to_pdf>       # extract schematics from PDF
python utils/caption-generator.py <path_to_pdf>  # generate GPT-4o captions
```

---

## Dataset

Download (7,500 schematics, 10 textbooks): https://drive.google.com/file/d/1aNC-8mye_Pbw9nYS0cmN2ggUaThJHUP2/view?usp=sharing

---

## Citation

```bibtex
@misc{bhandari2025masalachailargescalespicenetlist,
      title={Masala-CHAI: A Large-Scale SPICE Netlist Dataset for Analog Circuits by Harnessing AI},
      author={Jitendra Bhandari and Vineet Bhat and Yuheng He and Hamed Rahmani and Siddharth Garg and Ramesh Karri},
      year={2025},
      eprint={2411.14299},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2411.14299},
}
```
