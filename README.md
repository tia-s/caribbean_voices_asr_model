# Caribbean Voices ASR – Zindi Hackathon

**Team:** diTranscribers (_Collaborated with: @Dougsworth_)

**Model:** Whisper Large-v3  

**Approach:** Parameter-efficient fine-tuning of Whisper using LoRA (Low-Rank Adaptation)

## Overview

This project implements an Automatic Speech Recognition (ASR) system developed for the **Caribbean Voices ASR Zindi Hackathon**. The goal of the task is to transcribe Caribbean-accented speech from audio recordings into text.

The system fine-tunes the **Whisper Large-v3** model using **LoRA (Low-Rank Adaptation)** to efficiently adapt the model to the dataset while reducing the number of trainable parameters.

The pipeline consists of:

- Dataset preprocessing and feature extraction
- LoRA-based fine-tuning of Whisper Large-v3
- Beam search inference for transcription generation
- Postprocessing and normalization of transcriptions

---

# Methodology

## Model

The system is based on **OpenAI Whisper Large-v3**, a transformer-based sequence-to-sequence model designed for multilingual speech recognition and translation.

Fine-tuning is performed using **LoRA (Low-Rank Adaptation)**, which injects trainable low-rank matrices into selected attention layers while keeping the majority of model parameters frozen.

This approach allows efficient training with reduced GPU memory requirements.

### LoRA Configuration

- Rank (`r`): 64  
- Alpha: 128  
- Dropout: 0.05  
- Target modules:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`

---

# Data Processing

Audio files are loaded and resampled to **16kHz**, which matches Whisper's expected input format.

For each training sample:

1. The `.wav` file is loaded using **librosa**
2. Audio features are extracted using the **Whisper feature extractor**
3. The corresponding transcription is tokenized using the Whisper tokenizer

The dataset is split into:

- **Training set:** 90%
- **Validation set:** 10%

A custom **data collator** is used to dynamically pad audio features and label sequences during training.

---

# Training

Training is performed using the **HuggingFace Seq2SeqTrainer**.

Key training parameters:

| Parameter | Value |
|--------|--------|
| Batch size | 8 |
| Gradient accumulation | 2 |
| Learning rate | 2e-5 |
| Warmup steps | 500 |
| Maximum training steps | 3500 |
| Evaluation interval | 500 steps |
| Checkpoint interval | 500 steps |

Mixed precision (**FP16**) training is used to improve GPU efficiency.

The best model checkpoint (based on validation loss) is saved at the end of training.

---

# Inference

During inference:

1. The **base Whisper model** is loaded.
2. The **LoRA adapter weights** are loaded.
3. The adapter is **merged into the base model**.
4. Beam search decoding is used to generate transcriptions.

### Decoding configuration

- Beam size: 10  
- Temperature: 0.0  
- Repetition penalty: 1.2  
- Maximum tokens: 444

Using deterministic decoding ensures reproducible predictions.

---

# Text Postprocessing Normalization

Predicted transcriptions undergo several normalization steps.

### Cleaning

The following transformations are applied:

- Removal of filler words (e.g., *uh*, *um*, *ah*)
- Removal of duplicated words
- Removal of punctuation
- Lowercasing

### Special Term Normalization

Some structured tokens are expanded into spoken form:

Examples:

| Original | Normalized |
|--------|--------|
| Y2K | year 2000 |
| G15 | G-Fifteen |
| A300 | A-three-hundred |

### Number Normalization

Numbers are converted to written words:

Examples:

| Original | Normalized |
|--------|--------|
| 42 | forty two |
| 1st | first |
| 1970s | nineteen seventies |

This step improves alignment with expected transcription formats based on prior analysis of the test data.

---

# Environment Setup

## Install Dependencies

```bash
pip install transformers peft librosa num2words accelerate tqdm datasets scikit-learn
```

---

# Dataset Setup

Audio files are extracted from a compressed archive.

Example:

```python
!mkdir -p /content/Audio
!unzip -o -q "/content/drive/MyDrive/ZindiHackathon/Audio.zip" -d "/content/Audio"
```

---

# Training

Training is executed using the HuggingFace `Seq2SeqTrainer`.

The model checkpoint and processor are saved after training:

```
trainer.save_model(CONFIG["final_model_dir"])
processor.save_pretrained(CONFIG["final_model_dir"])
```

---

# Inference

To generate predictions on the test set:

1. Load the base Whisper model
2. Load the LoRA adapter
3. Merge adapter weights into the base model
4. Run beam search decoding

Predictions are written to:

```
submission.csv
```

---

# Postprocessing Pipeline

Final transcriptions are processed using a normalization pipeline:

1. Special term normalization
2. Number-to-word conversion
3. Final formatting and whitespace cleanup

The processed results are written back to the submission file.

---

# Reproducibility

Training uses:

```
random_state = 42
```

Inference uses deterministic decoding parameters:

```
temperature = 0.0
num_beams = 10
```

These settings ensure consistent outputs across runs.

---

# Tools and Libraries

- PyTorch
- HuggingFace Transformers
- PEFT (LoRA)
- Librosa
- Num2Words
- HuggingFace Datasets
- Scikit-learn
- TQDM

---

# Output

The final output is a CSV file containing predicted transcriptions:

```
ID,Transcription
audio_001,example transcription
audio_002,example transcription
```

