# **Assignment 3 – Question 5: RNN vs Transformer Language Model**

## Overview

This assignment implements and compares two neural language models trained on the same dataset using **Byte Pair Encoding (BPE)** tokenization:

1. **RNN-based model** – implemented with an **LSTM** layer
2. **Transformer-based model** – using multi-head self-attention blocks

The goal is to evaluate how both models perform on a small text corpus (Shakespeare dataset) in terms of training behavior, convergence, and loss.

---

## Project Structure

```
Assignment3_Q5/
│
├── tokenizer_bpe.py                 # BPE tokenizer using tiktoken (vocab = 10,000)
├── char_transformer_language_model.py  # Trains LSTM and Transformer models
├── input.txt                        # Shakespeare text dataset
├── report.pdf                       # Short analytical report
└── README.md                        # (this file)
```

---

## Requirements

You need **Python 3.10+** and the following libraries:

```bash
pip install torch torchvision torchaudio tiktoken
```
---

## How to Run

### **Step 1 – Tokenize the Dataset**

Run the BPE tokenizer:

```bash
python tokenizer_bpe.py
```

This will:

* Load `input.txt`
* Encode it using GPT-2’s BPE vocabulary (size = 10,000)
* Save the encoded data as `bpe_tokens.pt`

---

### **Step 2 – Train Both Models**

Run the main training script:

```bash
python char_transformer_language_model.py
```

This script:

* Loads the same tokenized dataset for both models
* Trains the **LSTM** first, then the **Transformer**
* Prints training progress and final validation losses

Example output:

```
Training LSTM...
LSTM Step 0, loss: 10.83
...
Training Transformer...
Transformer Step 800, loss: 5.37
Final Validation Loss — LSTM: 5.18, Transformer: 5.26
```

If the losses drop steadily, your implementation is correct.

---

### **Step 3 – View the Report**

Read `report.txt` (or upload as PDF).
It explains:

* Model setup
* Training results
* Performance comparison
* Analysis and conclusion

---

## Summary of Findings

| Model           | Strengths                                                               | Limitations                            |
| --------------- | ----------------------------------------------------------------------- | -------------------------------------- |
| **LSTM (RNN)**  | Fast training, good for small data, lower memory use                    | Struggles with long-range dependencies |
| **Transformer** | Captures long-term context, parallel training, smoother text generation | Higher memory cost, slower on CPU      |

Final validation losses were around **5.2–5.3**, confirming successful convergence.

---

## Notes

* Both models use **AdamW optimizer** and **cross-entropy loss**.
* Batch size = 16, block size = 64, learning rate = 3e-4.
* The entire pipeline runs on CPU or GPU automatically.

