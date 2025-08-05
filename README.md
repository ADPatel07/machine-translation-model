<div align="center">

# Language Translator (English ↔ Gujarati)

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Datasets-yellow)](https://huggingface.co/docs/datasets/)

</div>

A neural machine translation system that translates between English and Gujarati using a Transformer architecture. This project implements a sequence-to-sequence model with attention mechanism for accurate and efficient translation.

## Features

- **Bidirectional Translation**: Translate text between English and Gujarati
- **Transformer Architecture**: Implements the original "Attention is All You Need" paper architecture
- **Custom Tokenizer**: Word-level tokenizer with support for special tokens (SOS, EOS, PAD, UNK)
- **Training Pipeline**: Complete training and evaluation workflow
- **Hugging Face Integration**: Uses Hugging Face's datasets for training data

## Prerequisites

- Python 3.7+
- PyTorch
- Transformers library
- Datasets library
- Tokenizers library
- tqdm

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ADPatel07/machine-translation-model.git
   cd machine-translation-model
   ```

2. Install the required packages:
   ```bash
   pip install torch transformers datasets tokenizers tqdm
   ```

## Project Structure

- `main.py`: Main script for training and evaluating the translation model
- `texts.py`: Contains dataset handling, tokenizer, and training loop
- `bid_transformer.py`: Implements the Transformer model architecture
- `tokenizer_en.json`: Pre-trained English tokenizer
- `tokenizer_gu.json`: Pre-trained Gujarati tokenizer
- `checkpoint.pth`: Saved model weights

## Configuration

The model can be configured using the following parameters in `main.py`:

```python
config = {
    'src_lang': 'en',      # Source language
    'tgt_lang': 'gu',      # Target language
    'dmodel': 512,         # Dimension of the model
    'vocab_size': 0,       # Will be set automatically
    'seq_len': 0,          # Will be set based on dataset
    'num_layers': 3,       # Number of encoder/decoder layers
    'num_head': 8,         # Number of attention heads
    'dff': 2048,           # Dimension of feed-forward network
    'dropout': 0.1,        # Dropout rate
    'lr': 9.9**-4,         # Learning rate
    'batch_size': 10,      # Training batch size
    'epochs': 50           # Number of training epochs
}
```

## Usage

### Training

To train the model from scratch:

```bash
python main.py
```

The script will:
1. Load the Helsinki-NLP/opus-100 English-Gujarati dataset
2. Train the tokenizers if they don't exist
3. Train the transformer model
4. Save the model checkpoint

### Evaluation

The model automatically evaluates on the test set after training. The evaluation includes both English-to-Gujarati and Gujarati-to-English translation directions.

## Dataset

The model is trained on the Helsinki-NLP/opus-100 dataset, which contains parallel sentences in English and Gujarati. The dataset is automatically downloaded when you run the training script.

## Model Architecture

The implementation follows the original Transformer architecture from "Attention is All You Need" with:
- Multi-head self-attention
- Position-wise feed-forward networks
- Residual connections and layer normalization
- Scaled dot-product attention
- Masked self-attention in the decoder

## Performance

The model's performance can be evaluated using standard machine translation metrics like BLEU score, though these are not currently implemented in the codebase.

## Acknowledgements

- The Transformer architecture from "Attention is All You Need"
- Hugging Face for the datasets and tokenizers
- PyTorch for the deep learning framework
