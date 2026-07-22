# Gen-AI-IE7374

Milestone 3 project for IE7374 (Generative AI). This explores two NLP tasks on the AG News dataset: text classification and word-level language modeling (next-word prediction).

## What's in here

- `experiments/text_classification.ipynb` — the main notebook. Loads AG News, trains a TF-IDF + Logistic Regression baseline, an LSTM text classifier, and an LSTM language model.
- `docs/research_and_methods.md` — write-up covering objectives, literature review, benchmarking, and the preliminary experiments (including an overfitting issue we ran into and fixed).
- `docs/lstm_training_curves.png` — training/validation accuracy and loss plots for the classifier.
- `data/` — not used to store a static dataset; AG News is pulled directly through Hugging Face's `datasets` library at runtime.
- `requirements.txt` — Python packages needed to run the notebook.

## Setup

1. Clone the repo:
```
git clone https://github.com/clemenceu/Gen-AI-IE7374.git
cd Gen-AI-IE7374
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Open `experiments/text_classification.ipynb` in VS Code or Jupyter and run the cells top to bottom.



## What was done

Started out planning to do text classification with an LSTM, but since this is a Generative AI course, I added a language modeling task too (predicting the next word in a sequence).

Summary of results:
- A simple TF-IDF + Logistic Regression baseline actually beat the LSTM classifier on this dataset (87% vs 66% accuracy). Makes sense since we only used 8,000 training examples — LSTMs need more data to learn well from scratch.
- The first attempt at the language model overfit (validation loss kept getting worse while training loss kept improving). This was fixed by using more training data, a smaller LSTM, more dropout, and early stopping, which got validation loss actually improving alongside training loss.
- The language model ends up with a perplexity of about 664, and generates text that uses believable news-style words but isn't coherent yet. Full details and generated examples are in the write-up doc.

More detail, the literature we referenced, and full experiment results can be found in `docs/research_and_methods.md`.
