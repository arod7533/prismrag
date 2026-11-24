from pathlib import Path
import gradio as gr
import pandas as pd
from prismrag.pipeline import build_index, answer
from prismrag.metrics import METRICS
from prismrag.evaluation import read_cases, evaluate
from prismrag.visualization import embedding_plot

ROOT = Path(__file__).parent

def explain(selected):
    return "\n\n".join(f"**{name}** — {METRICS[name]}" for name in selected)

def index_files(paths, embedding, size, overlap):
    try:
        index = build_index(paths, embedding, int(size), int(overlap))
        plot = embedding_plot(index) if len(index.chunks) > 1 else None
        return index, f"Indexed {len(index.chunks)} chunks · {index.vectors.shape[1]} dimensions · {embedding}", plot
    except Exception as exc:
        raise gr.Error(f"Indexing failed. Check Ollama is running and the embedding model is installed. {exc}")

def chat(question, history, index, model, k):
    if index is None:
        raise gr.Error("Build an index first.")
    if not question.strip():
        raise gr.Error("Enter a question.")
    try:
        response, found, query, seconds = answer(index, question, model, int(k))
        evidence = [{"citation": f"[{rank}]", "source": doc.metadata["source"], "page": doc.metadata["page"], "cosine": round(score, 4), "passage": doc.page_content} for rank, (_, doc, score) in enumerate(found, 1)]
        return "", (history or []) + [{"role":"user", "content":question}, {"role":"assistant", "content":response}], evidence, embedding_plot(index, query, found) if len(index.chunks)>1 else None, f"{seconds:.2f}s · {len(found)} retrieved chunks"
    except Exception as exc:
        raise gr.Error(str(exc))

def run_eval(index, dataset, model, judge, k, selected):
    if index is None:
        raise gr.Error("Build an index first.")
    try:
        cases = read_cases(dataset or ROOT / "data/eval.json", index)
        rows, path = evaluate(index, cases, model, judge, int(k), selected)
        frame = pd.DataFrame(rows)
        summary = []
        for name in selected:
            values = pd.to_numeric(frame[name], errors="coerce").dropna()
            summary.append({"metric":name, "mean":round(float(values.mean()),4) if len(values) else None, "successful cases":len(values), "total cases":len(frame)})
        return pd.DataFrame(summary), frame, path
    except Exception as exc:
        raise gr.Error(str(exc))

def compare_runs(paths):
    import json
    if not paths:
        raise gr.Error("Select exported evaluation reports.")
    results = []
    try:
        for path in paths:
            report = json.loads(Path(path).read_text())
            config = report["config"]
            row = {"run": Path(path).name, **config}
            for name in config["metrics"]:
                values = pd.to_numeric(pd.Series([case.get(name) for case in report["cases"]]), errors="coerce").dropna()
                row[name] = float(values.mean()) if len(values) else None
                row[name + " valid/total"] = f"{len(values)}/{len(report['cases'])}"
            results.append(row)
        return pd.DataFrame(results)
    except Exception as exc:
        raise gr.Error(f"Invalid exported report: {exc}")

def create_app():
    with gr.Blocks(title="PrismRAG") as demo:
        state = gr.State(None)
        gr.Markdown("# PrismRAG\n### See the evidence. Measure the answer.\nA local retrieval and evaluation lab · LangChain × Ollama")
        with gr.Accordion("Corpus & experiment settings", open=True):
            with gr.Row():
                files = gr.File(file_count="multiple", file_types=[".pdf", ".txt", ".md"], type="filepath", label="Your documents")
                with gr.Column():
                    embedding = gr.Textbox(value="nomic-embed-text", label="Ollama embedding model")
                    model = gr.Textbox(value="qwen3.5:9b", label="Ollama answer model")
                    judge = gr.Textbox(value="qwen3.5:9b", label="Ollama judge model")
            with gr.Row():
                size = gr.Slider(200, 2000, value=700, step=100, label="Chunk size (characters)")
                overlap = gr.Slider(0, 400, value=100, step=25, label="Chunk overlap")
                k = gr.Slider(1, 12, value=4, step=1, label="Retrieved chunks (k)")
            with gr.Row():
                sample = gr.Button("Load sample corpus")
                build = gr.Button("Build index", variant="primary")
            status = gr.Markdown("Start Ollama, load documents, then build the index. Settings apply when you rebuild.")
        with gr.Tab("Chat & evidence"):
            with gr.Row():
                with gr.Column(scale=3):
                    history = gr.Chatbot(label="Ask your documents", height=460)
                    question = gr.Textbox(label="Question", placeholder="How does retrieval work?")
                    send = gr.Button("Ask", variant="primary")
                    timing = gr.Markdown()
                    clear = gr.Button("Clear conversation")
                    gr.Markdown("Each turn retrieves independently. Ask self-contained questions. Citations identify supplied passages; model citations still need review.")
                with gr.Column(scale=2):
                    gr.Markdown("### Retrieved evidence\nCosine similarity measures proximity, not confidence or correctness.")
                    context = gr.JSON(label="Ranked passages")
        with gr.Tab("Evaluation lab"):
            gr.Markdown("## Measure before you tune\nCompare runs with different chunk sizes, k, and models on the same labeled dataset. Sample cases are a tiny smoke benchmark, not a general quality claim.")
            dataset = gr.File(label="Evaluation JSON (optional; defaults to sample)", file_types=[".json"], type="filepath")
            selected = gr.CheckboxGroup(list(METRICS), value=["Precision@k", "Recall@k", "MRR", "nDCG@k", "Answer token F1", "Latency"], label="Metrics")
            help_text = gr.Markdown(explain(["Precision@k", "Recall@k", "MRR", "nDCG@k", "Answer token F1", "Latency"]))
            gr.Markdown("Judge metrics run locally and add latency. Using the same answer and judge model can bias scores. Prefer a separate, stronger judge and human spot checks. Failed judgments remain missing and are excluded from means.")
            run = gr.Button("Run evaluation", variant="primary")
            summary = gr.Dataframe(label="Aggregate scores", interactive=False)
            details = gr.Dataframe(label="Per-question results and judge reasons", interactive=False)
            download = gr.File(label="Download reproducible run JSON")
            gr.Markdown("### Compare experiments\nUpload saved run reports to compare means and settings. Compare matching corpus hashes and identical question sets; model judges may differ between runs.")
            reports = gr.File(label="Saved run JSON reports", file_count="multiple", file_types=[".json"], type="filepath")
            compare = gr.Button("Compare runs")
            comparison = gr.Dataframe(label="Run comparison", interactive=False)
        with gr.Tab("Embedding atlas"):
            gr.Markdown("## Inspect the vector space\nColor identifies source documents; symbols identify retrieved chunks; a star marks the latest query. PCA is a lossy projection of normalized embeddings. Clusters do not establish embedding quality. Use labeled Recall@k and nDCG alongside this view. The title reports retained variance.")
            plot = gr.Plot()
        sample.click(lambda: sorted(str(p) for p in (ROOT / "data/corpus").glob("*.md")), outputs=files)
        build.click(index_files, [files, embedding, size, overlap], [state, status, plot]).then(lambda: ([], None), outputs=[history, context])
        inputs = [question, history, state, model, k]
        outputs = [question, history, context, plot, timing]
        send.click(chat, inputs, outputs)
        question.submit(chat, inputs, outputs)
        clear.click(lambda: ([], None, ""), outputs=[history, context, timing])
        compare.click(compare_runs, reports, comparison)
        selected.change(explain, selected, help_text)
        run.click(run_eval, [state, dataset, model, judge, k, selected], [summary, details, download])
    return demo

if __name__ == "__main__":
    create_app().queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", share=False)
