**Small Transformer Language Model from Scratch**

A learning-focused implementation of a small Transformer-based language model using PyTorch. This project explores the fundamental architecture and working principles behind modern Large Language Models (LLMs) by implementing core Transformer components, training a language model, and generating text.

The project is designed to build a practical understanding of how Transformer-based language models process sequential data, learn patterns from text, and predict the next token.

**Project Overview**

Large Language Models such as GPT are built on the Transformer architecture, which uses attention mechanisms to understand relationships between tokens in a sequence.

In this project, I implemented a small Transformer language model from scratch using PyTorch to understand the internal building blocks of modern Generative AI systems.

The implementation focuses on the complete learning workflow:

1. Preparing and tokenizing text data.
2. Converting tokens into numerical representations.
3. Creating token and positional embeddings.
4. Implementing self-attention and multi-head attention.
5. Building Transformer blocks.
6. Training the language model using next-token prediction.
7. Generating text using the trained model.

This project helped me move beyond using pre-trained AI models and understand the fundamental concepts involved in building language models.

**Project Objectives**

1. Understand the architecture of Transformer-based language models.
2. Learn how tokenization converts text into numerical data.
3. Implement self-attention to capture relationships between tokens.
4. Understand how positional embeddings represent token positions.
5. Build a small language model using PyTorch.
6. Explore the training process of a causal language model.
7. Understand how neural networks learn to predict the next token.
8. Experiment with text generation and model predictions.
9. Strengthen practical knowledge of Deep Learning and Generative AI.



**Key Features**

1. Character-Level Tokenization

Implemented character-level tokenization to convert text into numerical token IDs.

The tokenizer maps characters to integer values and converts them back into text during decoding.

This provides a simple way to understand how language models represent text as numerical sequences.

2. Token and Positional Embeddings

Implemented token embeddings to represent input tokens as vectors.

Positional embeddings are used to provide information about the position of each token in the sequence.

Together, these embeddings help the model process both token identity and sequence position.

3. Self-Attention Mechanism

Implemented self-attention to help the model learn relationships between different tokens in a sequence.

The attention mechanism allows the model to assign importance to relevant tokens when processing the current token.

4. Multi-Head Attention

Implemented multi-head attention to allow the model to learn different relationships and patterns within the input sequence.

Multiple attention heads process information in parallel and combine their representations.

5. Transformer Blocks

Built Transformer blocks containing:

  Multi-head self-attention.
  Layer Normalization.
  Feed-forward neural networks.
  Residual connections.
  Dropout.

These components form the core architecture of the small language model.

6. Causal Language Modeling

Implemented next-token prediction using a causal language modeling approach.

The model learns to predict the next token based on the previous tokens in the sequence.

This is a fundamental concept behind autoregressive language models such as GPT.

7. Model Training

Implemented a training workflow using PyTorch.

The training process includes:

⚫️ Input and target sequence preparation.
⚫️ Forward propagation.
⚫️ Loss calculation.
⚫️ Backpropagation.
⚫️ Parameter optimization.
⚫️ Periodic loss evaluation.

8. Text Generation

Implemented text generation using the trained model.

The model predicts the next token step by step and uses the generated output as part of the next input sequence.

This demonstrates the basic inference process used in autoregressive language models.


**Model Architecture**

The project follows a simplified Transformer language model architecture:
Input Text
    │
    ▼
Character-Level Tokenizer
    │
    ▼
Token IDs
    │
    ▼
Token Embeddings
    │
    ▼
Positional Embeddings
    │
    ▼
Transformer Blocks
    │
    ├── Multi-Head Self-Attention
    ├── Residual Connection
    ├── Layer Normalization
    ├── Feed-Forward Neural Network
    ├── Residual Connection
    └── Layer Normalization
    │
    ▼
Language Model Head
    │
    ▼
Logits / Token Predictions
    │
    ▼
Next-Token Prediction
    │
    ▼
Generated Text



# MiniGPT: Character-Level Transformer from Scratch

A lightweight, character-level Transformer language model built from scratch using pure PyTorch. This repository contains a complete implementation of a decoder-only autoregressive Transformer (GPT architecture), including manual implementations of causal self-attention, multi-head attention, residual connections, and step-by-step text generation[cite: 1].

---

## Model Configuration

* **Embedding Dimension (`d_model`)**: 128[cite: 1]
* **Number of Attention Heads (`n_heads`)**: 4[cite: 1]
* **Head Size**: 32[cite: 1]
* **Transformer Layers (`n_layers`)**: 4[cite: 1]
* **Block Size (Context Length)**: 128 tokens[cite: 1]
* **Total Parameters**: ~820,014[cite: 1]
* **Dropout**: 0.1[cite: 1]
* **Optimizer**: AdamW (`lr=3e-4`)[cite: 1]
* **Loss Function**: Cross-Entropy Loss[cite: 1]
* **Training Steps**: 3,000[cite: 1]

---

## Key Features

* **Manual Multi-Head Causal Attention**: Implements query, key, and value linear projections, scaled dot-product attention, causal triangular lower masks (`tril`), and head concatenation[cite: 1].
* **Character-Level Tokenizer**: Custom string-to-integer (`stoi`) and integer-to-string (`itos`) mapping over unique dataset characters[cite: 1].
* **Modular Transformer Block**: Includes Layer Normalization (Pre-LN formulation), residual skip connections, and a position-wise feed-forward network with a $4\times$ hidden dimension expansion[cite: 1].
* **Flexible Text Generation**:
  * Autoregressive multinomial next-token sampling[cite: 1].
  * Step-by-step debugging generator displaying individual token IDs, characters, and probability selections in real-time[cite: 1].

---

## Project Structure

* `Small_Transformer_Manual.ipynb`: The complete Jupyter/Colab notebook containing dataset setup, model architecture, training loop, loss logging, and sampling inference[cite: 1].

---

## Quickstart

### Prerequisites

* Python 3.8+
* PyTorch (with CUDA support recommended for GPU acceleration)[cite: 1]

```bash
pip install torch


# Standard Generation
prompt = "Question: What is Python?\nAnswer:"
print(generate(prompt, max_new_tokens=200))

# Step-by-Step Token Inspection
generate_step_by_step("Question: What is language ?", max_new_tokens=50)


## Technologies Used

**Python** — Core programming language used for implementation[cite: 1].
**PyTorch** — Deep learning framework used to build neural network layers, calculate gradients, and train the model[cite: 1].
**Deep Learning** — Sequence modeling, backpropagation, and parameter optimization using AdamW[cite: 1].
**Transformer Architecture** — Decoder-only architecture featuring causal multi-head self-attention and Pre-LN Transformer blocks[cite: 1].
**Natural Language Processing (NLP)** — Character-level tokenization, sequence encoding/decoding, and autoregressive text generation[cite: 1].
**Jupyter / Google Colab** — Interactive notebook environment for GPU-accelerated training and experimentation[cite: 1].

** THE END**
