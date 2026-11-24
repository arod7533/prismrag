# PrismRAG
**See the evidence. Measure the answer.**

A local Python, LangChain, Gradio, Ollama and Chroma RAG portfolio project.

## Run
Install [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/), start Ollama, then:

```sh
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
uv sync --locked --python 3.12
uv run python app.py
```
Open http://127.0.0.1:7860. Documents load and vectorize automatically. Ask a question in Chat; Measurements updates automatically. No uploads, metric selections, datasets or evaluation buttons are required. First startup downloads no models automatically; the two pull commands install them first. Initial indexing takes longer than reopening the persisted collection.

## Included knowledge base
`data/corpus/` contains 24 fictional Northstar Analytics documents covering product, current and archived pricing, billing, support, security, permissions, connectors, incidents and employee operations. It includes overlapping policies and historical distractors. `data/eval.json` supplies 20 curated demonstration questions with reference answers and document-level source labels. The original three tutorial documents remain in `data/tutorial/` and are not indexed by the app. All company details are synthetic and clearly marked. See `data/CORPUS.md`.

## Two tabs
**Chat** introduces the fictional company and provides six clickable suggested questions, all matching labeled reference questions. Select an example to fill the message box, then Send. A conversation and readable source passages sit side by side. Each message retrieves independently and asks the local model for a cited answer.

**Measurements** is an automatic learning page organized around three questions:

1. **Did we retrieve the right evidence?** Four benchmark metrics, a short interpretation and a real per-question walkthrough showing expected versus retrieved sources. These are means across the development set, not answer scores.
2. **Is the latest answer good?** Reference-based accuracy, faithfulness and relevance for the latest chat only. An expandable discussion presents the actual answer, available reference, judge reasons, token overlap and retrieved source rankings. The page explains what each judge receives and why scores can disagree.
3. **What should we try next?** Suggestions tied to observed results, with one-change-at-a-time experiments and evaluation limitations.

Definitions and evaluation design are available in a separate expandable explanation. Chunk boundaries, full-vector topic diagnostics and the PCA scatterplot are under **Optional diagnostics**, with the plot itself collapsed. The scatterplot is exploratory; labeled retrieval scores evaluate usefulness and actual text boundaries help assess segmentation. There are no required setup or evaluation controls.

On startup, the app embeds and retrieves the 20 labeled development questions, retains the actual source rankings and per-question scores for discussion, then computes document-level Precision@k, Recall@k, MRR and binary nDCG@k. Duplicate chunks from one document count once. These are retrieval benchmark means, not scores for an unlabeled chat question. Partial source labels can miss other useful evidence.

After a chat answer, latency and latest retrieval update immediately; faithfulness, answer relevance and context relevance are scored automatically in a structured local judge call. For an exact bundled question, a separate structured judge estimates **answer accuracy** against the curated reference. Accuracy is a 0–1 reference-based estimate for the latest question, not a measured overall percent-correct rate; unlabeled questions explicitly show not measured. Faithfulness can be high for an answer grounded in outdated context while reference-based accuracy is low. The response appears before judging finishes. The same model answers and judges, so judge scores have correlated bias and require human spot checks. Failed judge calls show unavailable rather than zero. Token F1 is calculated only when the chat question exactly matches a bundled labeled question (ignoring case and outside whitespace); other questions show no reference. F1 misses valid paraphrases and does not prove factuality. Scores range 0–1; latency is seconds for retrieval plus answer generation, excluding judging/indexing.

Reports save automatically to gitignored `runs/`. They contain the actual question, answer, retrieved evidence, scores, model names and corpus hash. Review before sharing. The benchmark set is synthetic development data, not an independent held-out test set. No scores are fabricated. Models can cite incorrectly or follow injected document instructions despite prompt separation.

## Vector database and pipeline
Documents → LangChain recursive character chunks → Ollama embeddings → **persistent local Chroma** → cosine nearest-neighbor retrieval → LangChain prompt → ChatOllama answer.

Chroma collections are stored in gitignored `.chroma/`, keyed by the corpus/chunk fingerprint, embedding model name and installed model digest. Reopening unchanged data reuses stored vectors; changed data/model identity creates a different collection. Missing embeddings in an interrupted build are resumed. Chroma performs retrieval; NumPy normalizes stored vectors for the PCA display. Collections accumulate until removed manually; do not commit the database. This embedded single-process design is intended for a small local portfolio app, not a multi-worker production service.

The learning dashboard connects metrics to retrieval, generation, segmentation and speed, explains why each matters with examples, and proposes experiments from observed scores. Recommendations use simple review heuristics (for example, judge scores below 0.8); these are not calibrated pass/fail thresholds. Guidance never changes the pipeline automatically.

Color in the PCA plot identifies one of six human-authored topics, shared across related documents (`prismrag/topics.py`). A white diamond highlights retrieved chunks and a star marks the query. PCA is lossy; the title shows variance retained. Full-vector cosine silhouette summarizes separation relative to the supplied topic labels; near 1 indicates clearer separation, near 0 overlap and negative values indicate proximity to another topic. Nearest-topic agreement excludes chunks from the same document. Broad topics can legitimately overlap, so neither score is a universal embedding-quality grade.

Segmentation diagnostics report minimum/median/maximum character counts, short chunks under 100 characters (a review heuristic), and actual overlap, which can be lower than the splitter's configured 100-character target. The prefilled pricing document inspector shows exact zero-based character offsets `[start, end)`, interval bars and complete chunk text. Page-specific offsets are captured by LangChain's `add_start_index=True`. Incomplete endings may be headings or lists, so flags request inspection rather than marking a failure. Offsets/topic metadata create a new content fingerprint and Chroma collection; old collections remain untouched. Visual clustering does not prove retrieval quality. Model tags can change; embedding digests prevent reusing a collection with changed embedding weights. Answer-model digests/hardware are not yet captured in reports, so reports are not a strict reproducibility guarantee.

The app uses `qwen3.5:9b`, `nomic-embed-text`, 700-character chunks, 100-character overlap and top-k 4. Developers can change these constants in `app.py` and pipeline defaults. Restart after corpus changes. Loading state is shared through a process cache; chat and latest measurements are browser-session state. Baseline retrieval evaluations repeat per browser load, but document embeddings persist.

## Hosting
[Repository](https://github.com/arod7533/prismrag)

The Gradio app runs locally with Ollama. The GitHub repository contains the source code and setup instructions. A public live demo requires a separately provisioned backend with authentication, resource limits and TLS. Keep Ollama private. The app binds to localhost. Document text and model inference stay local by default; package/model installation requires network access.

## Checks
```sh
uv run pytest -q
uv run python -c "from app import create_app; create_app()"
# Requires installed models and running Ollama:
uv run python tests/smoke_local.py
```
CI runs model-free metric, automation, accuracy and chunk-boundary checks plus UI construction. Integration tests require local models.
