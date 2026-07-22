# Research and Selection of Methods

## 1. Define Objectives

This project explores two natural language processing task types: **text classification** and **language modeling**. Both were implemented and evaluated so the final approach could be chosen based on real evidence rather than assumption.

- **Text classification**: given a short news snippet, predict which of four topic categories it belongs to (World, Sports, Business, Sci/Tech).
- **Language modeling**: given a sequence of words, predict the next word — the foundational generative NLP task, and the one most directly aligned with this course's focus on generative models.

language modeling is treated as the primary deliverable, with text classification retained as a full secondary experiment and useful point of comparison (see Benchmarking below). This also reflects the project's original direction: the team's preliminary proposal (SupplySense) targeted a generative, retrieval-augmented assistant. Since that approach required paid LLM API access that wasn't available for this milestone, language modeling was chosen as a locally-runnable generative task that preserves the same idea: a model that produces new text.

## 2. Literature Review

**Text classification with recurrent networks.** Sequence models such as LSTMs are widely used for text classification because, unlike bag-of-words methods, they can capture word order and context. Comparative studies combining LSTM with other architectures report that LSTM-based models are consistently competitive with or better than classical baselines on standard benchmarks, particularly as dataset size grows (Wang & Zhou, 2022, in *Journal of Physics: Conference Series*; Alqahtani et al., 2022, *Frontiers in Computational Neuroscience*). Also, other work highlights that classical machine learning approaches (e.g., logistic regression, TF-IDF-based models) remain strong, fast baselines, especially on smaller datasets where deep models are more prone to overfitting (Aggarwal & Zhai, 2012, *A Survey of Text Classification Algorithms*). This tension — deep model expressiveness vs. classical model data-efficiency — directly motivated the benchmarking comparison run in this milestone.

**Recurrent neural network language models.** The use of recurrent neural networks for next-word prediction was established by Mikolov et al. (2010), who showed that RNN-based language models could achieve significantly lower perplexity than traditional n-gram statistical models by learning continuous representations of context rather than relying on fixed-order word counts. This paper is the foundational reference for the language modeling component of this project: the core idea implemented here — an Embedding layer feeding a recurrent (LSTM) layer, trained to predict the next word via cross-entropy loss — descends directly from this line of work. Perplexity, the evaluation metric used in this project's language model experiments, is the same metric introduced as the standard in this and subsequent RNN language modeling papers.

**Regularization.** Dropout (Srivastava, Hinton, Krizhevsky, Sutskever, & Salakhutdinov, 2014, *Journal of Machine Learning Research*) is the regularization technique applied throughout this project's neural models. The original paper demonstrates that randomly deactivating a fraction of neurons during training reduces co-adaptation between units and measurably reduces overfitting across vision, speech, and text tasks. This project applies the same principle directly: both the classification LSTM and the language model use dropout and recurrent dropout, and the language model's second iteration increases dropout specifically in response to an observed overfitting pattern (see Preliminary Experiments).

## 3. Benchmarking

Two approaches were directly compared on the same classification task and dataset (AG News, 8,000 training / 2,000 test samples) to evaluate accuracy, computational efficiency, and general suitability:

| Model | Accuracy | Weighted F1 | Training Time |
|---|---|---|---|
| TF-IDF + Logistic Regression | 87.3% | 87.2% | 1.5 seconds |
| Embedding + LSTM | 66.1% | 63.4% | ~2 minutes |

**Accuracy and performance:** the classical TF-IDF + Logistic Regression approach substantially outperformed the LSTM on this dataset size. This is a documented pattern in the literature reviewed above — deep sequence models typically need much more data to learn useful representations from scratch, while TF-IDF already encodes strong lexical signal that a linear classifier can exploit immediately.

**Computational efficiency:** the classical model trained roughly 80 times faster. This matters for iteration speed during development and would matter significantly at production scale.

**Scalability:** the LSTM approach is expected to scale better with more data — its accuracy gap with the classical baseline would likely narrow substantially at full dataset size (120,000 rows vs. the 8,000 used here), since neural sequence models generally continue improving with more data while TF-IDF-based linear models plateau earlier. This wasn't tested directly due to time constraints, but is a natural next step.

**Availability of pretrained models:** the LSTM classifier and language model here were trained entirely from scratch (embeddings included), which is the most data-hungry approach. Using pretrained word embeddings (e.g., GloVe, Word2Vec) instead of learning embeddings from scratch is a well-established way to substantially improve small-data performance, and is flagged as a direction for improvement in the language model experiments below.

Based on this comparison, language modeling was retained as the primary deliverable for its direct alignment with the course's generative focus, while the classification results are reported as a complete secondary experiment and a useful illustration of the classical-vs-deep-learning tradeoff for small text datasets.

## 4. Preliminary Experiments

Two rounds of experimentation were conducted, both surfacing and addressing real overfitting, which directly informed the final design choices.

### Experiment 1: Text classification (LSTM)

An Embedding → LSTM → Dense classifier was trained on 8,000 AG News samples across 4 categories (World, Sports, Business, Sci/Tech). Training accuracy climbed to 74.4% over 10 epochs while validation accuracy plateaued around 68–72%, with validation loss beginning to climb again after epoch 5 — a mild overfitting signal. The final held-out test evaluation (66.1% accuracy, 63.4% weighted F1) showed a clear per-class pattern: Sports (F1 = 0.90) and World (F1 = 0.84) were classified reliably, while Business (F1 = 0.58) and especially Sci/Tech (F1 = 0.24) were weak, with the model appearing to over-predict "Business" at the expense of "Sci/Tech" — plausibly due to shared vocabulary (companies, markets, products) between the two categories.

### Experiment 2: Language modeling (next-word prediction)

**First attempt.** A word-level language model (Embedding → LSTM(128) → Dense(8000, softmax)) was trained on 30,000 ten-word sequences sampled from the AG News corpus. This showed severe overfitting: validation loss rose from 6.64 to 8.96 over 15 epochs while training loss fell from 6.88 to 4.85, and validation accuracy flatlined around 15–16% despite training accuracy climbing to 21.9%. Given an 8,000-word vocabulary and a genuinely hard prediction task, 30,000 examples was not enough data for the model's capacity.

**Second attempt (adjustment).** Based on this finding, three changes were made together: (1) the training sample was increased to 80,000 sequences, (2) the LSTM was reduced from 128 to 64 units with dropout increased from 0.3/0.2 to 0.5/0.3, and (3) early stopping (patience = 3 epochs, monitoring validation loss) was added. This directly resolved the overfitting: validation loss tracked training loss closely and improved from 6.64 to a best of 6.50 by epoch 8, before early stopping halted training automatically. Final perplexity was 664.28 — meaningfully better than random guessing (perplexity ≈ 8,000) but still indicating a model that has learned domain-relevant word associations without achieving fluent generation. Sample generated text confirms this: outputs are not coherent sentences, but consistently draw on plausible in-domain vocabulary (e.g., a "stock market" seed generates words like "business" and "settlement"; a "football team" seed generates "growth," "attack," "giant").

### Adjustments for future work

Both experiments point to the same underlying limitation: model capacity outstripping available training data. The most promising next steps, in order of expected impact, are: (1) training on the full 302,635-sequence corpus rather than a subsample, (2) using pretrained word embeddings instead of learning them from random initialization, and (3) increasing sequence length beyond 10 words to give the model more context per prediction.

## References

Aggarwal, C. C., & Zhai, C. (2012). *A survey of text classification algorithms.* In *Mining Text Data* (pp. 163–222). Springer.

Mikolov, T., Karafiát, M., Burget, L., Černocký, J., & Khudanpur, S. (2010). Recurrent neural network based language model. In *Proceedings of INTERSPEECH 2010* (pp. 1045–1048).

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929–1958.

Wang, H., & Zhou, J. (2022). Research of text classification based on TF-IDF and CNN-LSTM. *Journal of Physics: Conference Series*, 2171.

Alqahtani, A., Khan, H. U., Alsubai, S., Sha, M., Almadhor, A., Iqbal, T., & Abbas, S. (2022). An efficient approach for textual data classification using deep learning. *Frontiers in Computational Neuroscience*, 16. https://doi.org/10.3389/fncom.2022.992296