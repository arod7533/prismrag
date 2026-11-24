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
        raise gr.Error("Open Documents in the Chat tab and prepare your documents or use the sample documents.")
    if not question.strip():
        raise gr.Error("Enter a question.")
    try:
        response, found, query, seconds = answer(index, question, model, int(k))
        evidence = [{"citation": f"[{rank}]", "source": doc.metadata["source"], "page": doc.metadata["page"], "cosine": round(score, 4), "passage": doc.page_content} for rank, (_, doc, score) in enumerate(found, 1)]
        import html
        evidence_text = "\n\n---\n\n".join(
            f"### {item['citation']} {html.escape(item['source'])} · page {item['page']}\n\n"
            + html.escape(item['passage']) for item in evidence)
        return "", (history or []) + [{"role":"user", "content":question}, {"role":"assistant", "content":response}], evidence_text, embedding_plot(index, query, found) if len(index.chunks)>1 else None, f"{seconds:.2f}s · {len(found)} retrieved chunks"
    except Exception as exc:
        raise gr.Error(str(exc))

def run_eval(index, dataset, model, judge, k, selected):
    if index is None:
        raise gr.Error("Open Documents in the Chat tab and prepare your documents or use the sample documents.")
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
    defaults = ["Precision@k", "Recall@k", "MRR", "nDCG@k", "Answer token F1", "Latency"]
    with gr.Blocks(title="PrismRAG") as demo:
        state = gr.State(None)
        gr.Markdown("# PrismRAG\nAsk your documents. See the sources.")
        with gr.Tab("Chat"):
            with gr.Accordion("Documents", open=False):
                files = gr.File(file_count="multiple", file_types=[".pdf", ".txt", ".md"], type="filepath", label="Your documents")
                with gr.Row():
                    sample = gr.Button("Use sample documents")
                    build = gr.Button("Prepare documents", variant="primary")
                status = gr.Markdown("Add documents and select Prepare documents, or use the sample documents.")
            with gr.Row():
                with gr.Column(scale=1):
                    history = gr.Chatbot(label="Conversation", height=520)
                    question = gr.Textbox(label="Message", show_label=False, placeholder="Ask a question about your documents…")
                    with gr.Row():
                        send = gr.Button("Send", variant="primary")
                        clear = gr.Button("Clear chat")
                with gr.Column(scale=1):
                    with gr.Group(elem_id="context-panel"):
                        gr.Markdown("### Relevant context")
                        context = gr.Markdown("Passages used to answer your question will appear here.", elem_id="context-passages")
        with gr.Tab("Measurements"):
            gr.Markdown("## Evaluation\nMeasure answer quality and retrieval, compare experiments, and inspect embeddings.")
            with gr.Accordion("Models & retrieval settings", open=False):
                with gr.Row():
                    embedding = gr.Textbox(value="nomic-embed-text", label="Embedding model")
                    model = gr.Textbox(value="qwen3.5:9b", label="Answer model")
                    judge = gr.Textbox(value="qwen3.5:9b", label="Judge model")
                with gr.Row():
                    size = gr.Slider(200, 2000, value=700, step=100, label="Chunk size (characters)")
                    overlap = gr.Slider(0, 400, value=100, step=25, label="Chunk overlap")
                    k = gr.Slider(1, 12, value=4, step=1, label="Retrieved chunks (k)")
                gr.Markdown("Prepare documents again after changing chunking or embedding settings. Each chat message retrieves independently.")
            timing = gr.Markdown("Chat latency will appear here after a response.")
            dataset = gr.File(label="Evaluation dataset · JSON (optional; defaults to sample)", file_types=[".json"], type="filepath")
            selected = gr.CheckboxGroup(list(METRICS), value=defaults, label="Metrics to measure")
            with gr.Accordion("What do these metrics mean?", open=False):
                help_text = gr.Markdown(explain(defaults))
            with gr.Accordion("How to interpret results", open=False):
                gr.Markdown("The sample dataset is a smoke benchmark, not a general quality claim. Local model judges are fallible; prefer a separate stronger judge and human checks. Failed judgments are excluded from means and reported in success counts. Cosine similarity is proximity, not confidence. Model citations need review.")
            run = gr.Button("Run evaluation", variant="primary")
            summary = gr.Dataframe(label="Aggregate scores", interactive=False)
            with gr.Accordion("Per-question results & export", open=False):
                details = gr.Dataframe(label="Answers, scores and judge reasons", interactive=False)
                download = gr.File(label="Download run JSON")
            with gr.Accordion("Compare experiments", open=False):
                gr.Markdown("Compare saved runs using the same corpus hash and question set.")
                reports = gr.File(label="Saved run JSON reports", file_count="multiple", file_types=[".json"], type="filepath")
                compare = gr.Button("Compare runs")
                comparison = gr.Dataframe(label="Run comparison", interactive=False)
            with gr.Accordion("Vector visualization", open=False):
                gr.Markdown("Color = source; symbols = retrieved chunks; star = latest query. PCA is a lossy projection; retained variance appears in the title. Clusters alone do not prove quality. Check labeled recall and nDCG.")
                plot = gr.Plot()
        empty_context = "Passages used to answer your question will appear here."
        sample.click(lambda: sorted(str(p) for p in (ROOT / "data/corpus").glob("*.md")), outputs=files).then(
            index_files, [files, embedding, size, overlap], [state, status, plot]).then(
            lambda: ([], empty_context), outputs=[history, context])
        build.click(index_files, [files, embedding, size, overlap], [state, status, plot]).then(
            lambda: ([], empty_context), outputs=[history, context])
        inputs = [question, history, state, model, k]
        outputs = [question, history, context, plot, timing]
        send.click(chat, inputs, outputs)
        question.submit(chat, inputs, outputs)
        clear.click(lambda: ([], empty_context, ""), outputs=[history, context, timing])
        compare.click(compare_runs, reports, comparison)
        selected.change(explain, selected, help_text)
        run.click(run_eval, [state, dataset, model, judge, k, selected], [summary, details, download])
    return demo

if __name__ == "__main__":
    create_app().queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", share=False, footer_links=[], theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"), css="""
.gradio-container { max-width: 1480px !important; }
#context-panel { height: 520px; overflow-y: auto; padding: 20px; }
#context-panel h3 { color: #c084fc; }
#context-passages { line-height: 1.7; overflow-wrap: anywhere; }
@media (max-width: 768px) { #context-panel { height: 360px; } }
""")
