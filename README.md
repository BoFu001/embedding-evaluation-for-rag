# Which Embedding Should a RAG System Use?

An empirical comparison of **TF-IDF**, **Sentence-BERT** (`all-mpnet-base-v2`) and **OpenAI** (`text-embedding-3-small`) on the two jobs an embedding does inside a RAG pipeline: judging whether two texts mean the same thing, and retrieving the right document for a query.

**Short answer: it depends on how many words the query and the document share.**

<p align="center">
  <img src="images/jaccard_qqp_vs_msmarco.png" width="100%">
</p>
<p align="center">
  <img src="images/qqp_vs_msmarco_r1.png" width="70%">
</p>

On Quora, duplicate questions share many words (mean Jaccard 0.417) and TF-IDF keeps up with dense models. On MS MARCO, short queries are matched against long passages with almost no shared vocabulary (Jaccard 0.069), and dense embeddings pull clearly ahead: **Recall@1 0.99 vs 0.88**. The second setting is much closer to real RAG retrieval.

## Findings

1. **Task-specific beats bigger.** Sentence-BERT, trained for semantic similarity, beat OpenAI's larger general-purpose model on duplicate detection (F1 **0.841** vs 0.815), and it runs locally for free.
2. **Lexical overlap decides the winner in retrieval.** TF-IDF had the best Recall@5/10 on Quora but the worst Recall@1 on MS MARCO.
3. **Aggregate metrics can mislead.** TF-IDF's best F1 needed a threshold of 0.30, which gave it the highest recall (0.957) and the lowest precision (0.616). Looking at score distributions, not only the metrics, showed why.

| | Quora · similarity F1 | Quora · retrieval R@1 | MS MARCO · retrieval R@1 |
|---|---|---|---|
| TF-IDF | 0.750 | 0.745 | 0.884 |
| Sentence-BERT | **0.841** | 0.750 | 0.986 |
| OpenAI | 0.815 | **0.781** | **0.990** |

Full results (Accuracy, Precision, Recall, Spearman, MAP@10, NDCG@10) are in the notebook.

## Where all three fail

On the 72 duplicate pairs with almost no shared words (Jaccard < 0.15), dense models score them much higher than TF-IDF (mean 0.72 vs 0.53), but some pairs defeat every method. *"Do I need an anti-virus for my mac?"* vs *"Does a Mac need anti-virus software? Why and why not?"* gets a **negative** score from Sentence-BERT: the yes/no question and the longer two-part question look different to the model even though they ask the same thing. Zero-shot has limits, and fine-tuning on domain data is the next step.

<p align="center">
  <img src="images/error_analysis_scores.png" width="70%">
</p>

## Design choices

- **Fair thresholds:** each method gets its own classification threshold, found by grid search to maximise F1, so no method is helped or hurt by a fixed cut-off.
- **Two datasets on purpose:** Quora (3,000 balanced pairs, short texts, high overlap) and MS MARCO (500 query-passage pairs, long texts, low overlap) to test whether the ranking of methods holds across conditions. It does not.
- **Zero-shot and reproducible:** pre-trained models only, no fine-tuning, fixed seed 42.

## What I take into RAG work

- Choose the embedding model for the task and the data, not by size or price.
- Keep a lexical component: hybrid (keyword + dense) retrieval covers the cases where exact terms matter.
- Calibrate thresholds and inspect score distributions before trusting a single metric.

These lessons fed directly into my dissertation project, [EquityMind](https://github.com/BoFu001/equitymind-core), an agentic RAG system for financial analysis.

---

Coursework for **INM434: Natural Language Processing**, MSc Artificial Intelligence, City St George's, University of London.
Notebook: [`INM434_Coursework_BoFu.ipynb`](INM434_Coursework_BoFu.ipynb) · [Open in Colab](https://colab.research.google.com/drive/1UYQaO2iJwsR07riiFP3fak2YOM2pH9qo) · Libraries: `sentence-transformers`, `openai`, `scikit-learn`, `datasets`, `scipy`
