#!/usr/bin/env python

"""
Example usage:
    1) Run this script:
       $ python finetuned_codellama_spice.py

    2) When prompted, enter a textual description of your circuit.
       For instance:
       "Write a spice netlist for a Common-source amplifier with R load.
        The input signal is applied to the gate, the source is grounded,
        and the output is taken from the drain through a resistor load."
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import (
    LoraConfig, get_peft_model, set_peft_model_state_dict
)
from safetensors.torch import load_file

def format_prompt(description: str) -> str:
    """
    Format the user description into the SPICE-style prompt
    that the model expects.
    """
    return f"""
You are a powerful Text-SPICE model. Your job is to use the textual description of a schematic 
and provide the most suitable and syntactically correct SPICE netlist for the circuit.

You must start your output with *SPICE and output the voltage sources, current sources and 
other circuit elements in the right syntax.

### Description:
{description}

### NetList:
"""

def run_inference_finetuned(
    base_model_name: str,
    finetuned_dir: str,
    description: str
) -> str:
    """
    Loads the base Code Llama model, merges LoRA weights from `finetuned_dir`,
    and generates text for the given circuit description.
    """

    # Load the base model (using 4-bit quantization for large models).
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # Change cache_dir if you wish to store the model elsewhere.
        cache_dir="/scratch/vrb9107/",
        trust_remote_code=True
    )

    # Create LoRA configuration and attach it to the base model.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    # Load the fine-tuned adapter weights.
    resume_from_checkpoint = f"{finetuned_dir}/adapter_model.safetensors"
    adapter_weights = load_file(resume_from_checkpoint, device="cpu")
    set_peft_model_state_dict(model, adapter_weights)

    # Load the tokenizer. 
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Format the prompt from the user's circuit description.
    prompt = format_prompt(description)

    # Encode the prompt and move to GPU.
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    model.eval()

    # Generate the SPICE netlist.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

    # Decode and return the text.
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    # Example question to help users frame their prompt:
    print(
        "Example question:\n"
        "  Write a spice netlist for a Common-source amplifier with R load.\n"
        "  The input signal is applied to the gate, the source is grounded,\n"
        "  and the output is taken from the drain through a resistor load.\n"
        "----------------------------------------------------"
    )

    # Prompt user for the circuit description.
    user_description = input("Please describe the circuit you want a SPICE netlist for:\n> ")
    print("\n================ Processing ================\n")

    # Replace these paths with your actual model and LoRA checkpoint directories.
    BASE_MODEL_NAME = "codellama/CodeLlama-70b-hf"
    FINETUNED_LORA_DIR = "./code_llama70B_spice_shortdesc_finetune/"

    # Run inference on the user's description.
    spice_output = run_inference_finetuned(
        base_model_name=BASE_MODEL_NAME,
        finetuned_dir=FINETUNED_LORA_DIR,
        description=user_description
    )

    # Print the result.
    print("\n================ SPICE Netlist Output ================\n")
    print(spice_output)
    print("\n------------------------------------------------------")


if __name__ == "__main__":
    main()
