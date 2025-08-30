# Smol Text Generator: A Custom 135M Parameter LLM

## Project Overview

This repository contains the complete source code for building, training, and deploying "Smol Text Generator," a custom-built 135 million parameter Large Language Model (LLM). The project demonstrates an end-to-end MLOps workflow, from defining a custom model architecture and training it from scratch on a large-scale dataset, to serving it for interactive use via a web interface.

This project serves as a capstone, showcasing the practical steps involved in creating a smaller-scale, yet capable, generative AI model.

## Key Features

*   **Custom Model Architecture**: A decoder-only transformer model (`model_smol.py`) inspired by modern architectures like Llama, featuring:
    *   **Grouped-Query Attention (GQA)** for efficient inference.
    *   **Rotary Positional Embeddings (RoPE)** for better sequence length handling.
    *   **RMS Normalization** for stable training.
    *   **SiLU (SwiGLU)** activation functions in the MLP layer.
*   **Training from Scratch**: The model is trained on the `HuggingFaceTB/smollm-corpus` dataset using a robust training script (`train.py`) built with `PyTorch`, `Accelerate`, and `Transformers Trainer`.
*   **Interactive Demo**: An easy-to-use web interface (`app.py`) powered by Gradio allows for real-time text generation and experimentation with the trained model.
*   **Reproducibility**: The repository includes detailed configuration (`config.yaml`), dependencies (`requirements.txt`), and training logs to ensure the results can be reproduced.

## Model Details

| Parameter               | Value        |
| ----------------------- | ------------ |
| **Total Parameters**    | ~135M        |
| **Vocabulary Size**     | 49,152       |
| **Architecture**        | Decoder-Only |
| **Transformer Layers**  | 30           |
| **Hidden Size (d_model)** | 576          |
| **FFN Intermediate Size** | 1536         |
| **Attention Heads**     | 9 Query, 3 Key/Value (GQA) |
| **Context Length**      | 2048 tokens  |

## Project Structure

```
.
├── app.py              # Gradio web application for inference
├── model_smol.py       # Defines the custom LLM architecture
├── train.py            # Script for training the model from scratch
├── config.yaml         # Detailed configuration for model and training
├── requirements.txt    # Python dependencies
├── checkpoints/        # Directory for saving model checkpoints (not in repo)
└── pytorch_model.bin   # Trained model weights (not in repo)
```

## Getting Started

### 1. Prerequisites

*   Python 3.8+
*   PyTorch 2.0+
*   An NVIDIA GPU (for CUDA) or Apple Silicon (for MPS) is highly recommended for training.

### 2. Installation

Clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/shrits-ai/CapstoneModelTrainingPart.git
cd CapstoneModelTrainingPart

# (Recommended) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

The `train.py` script handles the entire training process. It streams the dataset directly from the Hugging Face Hub, so no manual download is required.

To start training from scratch:
```bash
python train.py
```

The script will:
*   Initialize the model, tokenizer, and datasets.
*   Use `wandb` for logging (can be used offline).
*   Save checkpoints periodically to the `./checkpoints` directory.
*   Save the final model to `./checkpoints/final_5000`.

To resume training from a checkpoint, ensure the checkpoint exists in the `checkpoints` directory. The script will automatically detect and resume.

### 2. Running the Inference Demo

Before running the demo, you need a trained model file. Ensure your final trained model is saved as `pytorch_model.bin` in the root directory of the project.

```bash
# Example: copy the final model from the training output
cp checkpoints/final_5000/pytorch_model.bin .
```

Once the model file is in place, launch the Gradio application:
```bash
python app.py
```

This will start a local web server. Open the provided URL (e.g., `http://127.0.0.1:7860`) in your browser to interact with the model. A public `share` link will also be generated for temporary external access.

Model is deployed in huggingface: `https://huggingface.co/spaces/Shriti09/CapstoneSMOLTextGenerator`

###

## Roadmap & Future Improvements

This project provides a solid foundation. Future work could include:

*   **KV Cache Implementation**: Implement Key-Value caching in the `generate` method of `model_smol.py` to significantly accelerate inference speed.
*   **Rigorous Evaluation**: Add an evaluation pipeline using standard NLP benchmarks (e.g., perplexity on held-out sets, performance on downstream tasks) to formally assess model quality.
*   **Model Scaling**: Experiment with larger model configurations by adjusting parameters in `config.yaml` to train more powerful versions.
*   **Fine-Tuning**: Develop scripts for fine-tuning the base model on specific domains or tasks (e.g., instruction following, summarization).

---
