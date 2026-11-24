# PrismRAG
**See the evidence. Measure the answer.**

A portfolio RAG lab built with Python, LangChain, Gradio, and local Ollama models.

## Run locally
Install [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/), then:

```sh
ollama serve
# In another terminal:
ollama pull qwen3.5:9b
ollama pull nomic-embed-text
uv sync --locked --python 3.12
uv run python app.py
```
Open http://127.0.0.1:7860. Open Documents in Chat and choose Use sample documents, then ask a question. Upload your own files and choose Prepare documents to replace the sample corpus. Model downloads require internet; inference stays local. You can substitute installed Ollama models in the settings. First inference can be slow.

## Features
- Two-tab interface: clean Chat beside readable context, and Measurements for evaluation, settings, experiment comparisons, latency, and vector visualization.
- TXT, Markdown, text-based PDF ingestion; configurable chunking and exact cosine retrieval.
- Measurements tab with selectable metric definitions, per-case results, judge reasons, aggregate means, and downloadable JSON runs.
- PCA embedding atlas colored by source, with retrieved chunks and projected query.
- Static GitHub Pages showcase and separate Python CI.

## Architecture
Documents → LangChain recursive character splitter → Ollama embeddings → normalized in-memory NumPy index → cosine top-k → LangChain prompt → ChatOllama → cited response.

The exact index is intentionally simple and inspectable for a small portfolio corpus. It is rebuilt per browser session, does not persist private document contents, and is not intended for millions of chunks. Evaluation exports in `runs/` contain question, reference, answer, filenames, corpus hash, model names, settings, scores and judge reasons. They are gitignored; review before sharing. Model tags are mutable: record model digests and machine details for strict reproducibility. Changing chunk or embedding settings requires rebuilding the index. Each chat question is retrieved independently.

## Evaluation practice
Upload a JSON array matching `data/eval.json`. Each case requires `question`, `reference`, and a nonempty `relevant_sources` list matching indexed filenames. Labels are source-level and repeated chunks are deduplicated for retrieval scoring. Metrics use the number of distinct retrieved sources as the precision denominator. Add chunk-level graded labels for a more demanding benchmark.

Deterministic: Precision@k, Recall@k, MRR, binary nDCG@k, answer token F1, and retrieval-plus-generation latency. Optional local LLM judge: faithfulness, answer relevance, context relevance. Judge failures remain null, with success counts in aggregates. These are custom transparent evaluators, not a claim of RAGAS equivalence.

Start with a human-labeled development set, tune chunk size/overlap/k, and compare downloaded runs with the same corpus hash. Reserve a held-out test set and use a separate stronger judge. Inspect unsupported claims and missing evidence manually. The three bundled cases are smoke data only; no performance results are fabricated. PCA colors and clusters do not prove embedding quality. No automatic citation validity or safety guarantee is implied.

## GitHub and Pages
GitHub Pages cannot run Gradio, Python, or Ollama. `docs/` is a static showcase explaining the real application; it does not pretend to provide live inference. For a live public app, provision a backend with model resources, authentication, quotas, and TLS; keep Ollama private. This is a local single-user app by default.

Create an empty GitHub repository called `prismrag`, then run:

```sh
git remote add origin https://github.com/YOUR_USERNAME/prismrag.git
git push -u origin main
```
In repository Settings → Pages → Source choose **GitHub Actions**. The included workflow publishes `docs/`. The Python workflow tests independently and does not need model downloads.

## Validation
```sh
uv run pytest -q
uv run python -c "from app import create_app; create_app()"
```
Retrieval metric unit tests run without Ollama. Full integration requires a running Ollama instance with the selected embedding, answer and judge models. Documents are untrusted evidence; prompt separation reduces but does not eliminate prompt injection. Bind locally and do not upload sensitive files to an exposed instance.
