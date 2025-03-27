# Instructions for Fine-Tuning CodeLlama-70B with MASALA-CHAI Descriptions-SPICE Dataset

This folder demonstrates how we fine-tuned **CodeLlama-70B** on a custom dataset of text-to-SPICE descriptions, derived from the MASALA-CHAI framework. Below, you will find information on how to set up your environment, access the fine-tuned model weights, and run inference scripts.

---

## 1. Overview

- **Model**: CodeLlama-70B  
- **Dataset**: MASALA-CHAI Descriptions-SPICE  
- **Objective**: To generate SPICE netlists from short textual descriptions of circuits.

The fine-tuned checkpoint is available through a Google Drive folder link: https://drive.google.com/drive/folders/1HVT_vbTpFWXPdujXHJozfS2cPYzF6BlQ?usp=sharing. This will allow you to replicate our results locally. Unzip the folder here.

---

## 2. Environment Setup

We have provided an `environment.yml` file to simplify the Conda environment setup. To create the Conda environment exactly as we used, follow these steps:

```bash
# Create environment from yml
conda env create -f environment.yml

# Activate the environment
conda activate codellama-env

## 3. GPU Requirements

Due to the size of the CodeLlama-70B model, you will need a GPU with at least 80GB of memory. Our recommended configuration is an NVIDIA A100 80GB for smooth inference.

## 4. Run inference script

```bash
# The script will ask for an input design question from the user
python codellama-inference.py


