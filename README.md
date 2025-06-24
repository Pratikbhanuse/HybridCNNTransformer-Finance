# HybridLOBForecast

This repository contains a hybrid CNN + Transformer deep learning model to forecast intraday price movements using Limit Order Book (LOB) snapshots combined with sentiment scores extracted from financial texts via FinBERT.

## Features

- CNN + Transformer architecture for sequence modeling of LOB data.
- Sentiment analysis integration using FinBERT.
- Multi-class classification of intraday price direction.
- Training and evaluation with metrics: accuracy, F1-score, confusion matrix, ROC-AUC.
- Visualization and logging of training progress.

## Dataset

The model uses the [FI-2010 dataset](https://raw.githubusercontent.com/seanahmad/fi2010/refs/heads/main/data/data.csv) (please download and place `fi2010.csv` in the repo root).

## Installation

Use the provided `requirements.txt` to install dependencies:

```bash
pip install -r requirements.txt
