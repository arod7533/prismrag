from functools import lru_cache
from pathlib import Path
import html
import gradio as gr
from prismrag.pipeline import build_index, answer
from prismrag.automatic import benchmark, measurement_rows, waiting_rows, judge_turn, save_turn
from prismrag.visualization import embedding_plot

ROOT = Path(__file__).parent
ANSWER_MODEL = "qwen3.5:9b"
EMBEDDING_MODEL = "nomic-embed-text"
K = 4
EMPTY_CONTEXT = "Passages used to answer your question will appear here."
CSS = """
.gradio-container { max-width: 1480px !important; }
#context-panel { height: 520px; overflow-y: auto; padding: 20px; }
#context-panel h3 { color: #c084fc; }
#context-passages { line-height: 1.7; overflow-wrap: anywhere; }
@media (max-width: 768px) { #context-panel { height: 360px; } }
"""

@lru_cache(maxsize=1)
def load_corpus():
    paths = sorted(p for p in (ROOT / "data/corpus").iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"})
    index = build_index(paths, EMBEDDING_MODEL)
    return index

def initialize():
    try:
        index = load_corpus()
        atlas = embedding_plot(index) if len(index.chunks) > 1 else None
        corpus = (f"**{len({d.metadata['source'] for d in index.chunks})} documents · {len(index.chunks)} chunks · "
                  f"{index.vectors.shape[1]} dimensions** · Chroma (local, persistent) · {EMBEDDING_MODEL}")
        # Show a ready chat before calculating the bundled retrieval benchmark.
        yield index, {}, "Documents ready. Ask a question below.", corpus, waiting_rows(), atlas
        try:
            baseline = benchmark(index, ROOT / "data/eval.json", K)
            rows = measurement_rows(baseline)
        except Exception as exc:
            baseline = {"error": str(exc), "corpus": index.fingerprint}
            rows = [[name, "Benchmark unavailable" if name in {"Precision@k", "Recall@k", "MRR", "nDCG@k"} else value, description] for name, value, description in waiting_rows()]
            corpus += f"\n\nBenchmark unavailable: {exc}"
        yield index, baseline, "Documents ready. Ask a question below.", corpus, rows, atlas
    except Exception as exc:
        yield None, {}, f"Documents could not load. Check Ollama is running with {ANSWER_MODEL} and {EMBEDDING_MODEL}. {exc}", "Corpus unavailable.", waiting_rows(), None

def chat(question, history, index, baseline):
    if index is None:
        raise gr.Error("The documents are still loading. If loading failed, start Ollama and refresh this page.")
    if not question.strip():
        raise gr.Error("Enter a question.")
    try:
        response, found, query, seconds = answer(index, question, ANSWER_MODEL, K)
        context = "\n\n---\n\n".join(f"### [{rank}] {html.escape(doc.metadata['source'])} · page {doc.metadata['page']}\n\n{html.escape(doc.page_content)}"
            for rank, (_, doc, _) in enumerate(found, 1))
        turn = {"question": question, "answer": response, "seconds": seconds,
                "context": [doc.page_content for _, doc, _ in found],
                "retrieved": [{"source": doc.metadata["source"], "cosine": score} for _, doc, score in found],
                "model": ANSWER_MODEL, "embedding_model": EMBEDDING_MODEL, "k": K}
        table = [[rank, doc.metadata["source"], doc.metadata["page"], round(score, 4)] for rank, (_, doc, score) in enumerate(found, 1)]
        return ("", (history or []) + [{"role":"user", "content":question}, {"role":"assistant", "content":response}],
                context, embedding_plot(index, query, found) if len(index.chunks)>1 else None,
                turn, measurement_rows(baseline, turn), table, "Answer ready. Quality checks are running automatically…")
    except Exception as exc:
        raise gr.Error(str(exc))

def automatic_checks(turn, baseline):
    if not turn:
        return measurement_rows(baseline), "Ask a question to measure answer quality."
    try:
        judgments = judge_turn(turn, ANSWER_MODEL)
        rows = measurement_rows(baseline, turn, judgments=judgments)
        status = "Latest answer evaluated automatically. Local judge estimates need human review."
    except Exception as exc:
        rows = measurement_rows(baseline, turn, judge_error=str(exc))
        status = "Answer available. Local quality judging failed; retrieval and timing measurements remain available."
    try:
        save_turn(turn, baseline, rows)
    except Exception:
        status += " Run could not be saved to disk."
    return rows, status

def create_app():
    with gr.Blocks(title="PrismRAG") as demo:
        index = gr.State(None)
        baseline = gr.State({})
        turn = gr.State(None)
        gr.Markdown("# PrismRAG\nAsk Northstar Analytics’ knowledge base. See the sources.")
        with gr.Tab("Chat"):
            ready = gr.Markdown("Loading the included documents automatically…")
            with gr.Row():
                with gr.Column(scale=1):
                    history = gr.Chatbot(label="Conversation", height=520)
                    question = gr.Textbox(label="Message", show_label=False, placeholder="What does the Growth plan cost per month now?")
                    with gr.Row():
                        send = gr.Button("Send", variant="primary")
                        clear = gr.Button("Clear chat")
                with gr.Column(scale=1):
                    with gr.Group(elem_id="context-panel"):
                        gr.Markdown("### Relevant context")
                        context = gr.Markdown(EMPTY_CONTEXT, elem_id="context-passages")
        with gr.Tab("Measurements"):
            gr.Markdown("## Automatic measurements\nEverything updates automatically. Retrieval scores use the bundled labeled questions; answer quality and latency describe your latest chat response.")
            corpus = gr.Markdown("Loading corpus…")
            progress = gr.Markdown("Ask a question to measure answer quality.")
            measurements = gr.Dataframe(headers=["Measurement", "Result", "What it means"], value=waiting_rows(), interactive=False, wrap=True, label="Evaluations & measurements")
            gr.Markdown("### Latest retrieval\nCosine similarity is proximity, not confidence. Each message retrieves independently.")
            retrieval = gr.Dataframe(headers=["Rank", "Source", "Page", "Cosine similarity"], value=[], interactive=False, label="Retrieved passages")
            gr.Markdown("### Vector space\nColor identifies sources; symbols identify retrieved chunks; the star is your latest question. PCA loses information; retained variance appears in the title. Clusters alone do not establish quality.")
            plot = gr.Plot()
            gr.Markdown("The company and documents are fictional. Bundled questions are development examples, not a held-out benchmark. Judge scores are fallible estimates from the answer model. Token F1 requires an exact bundled question with a reference answer. Reports save automatically to `runs/` locally.")
        demo.load(initialize, outputs=[index, baseline, ready, corpus, measurements, plot])
        inputs = [question, history, index, baseline]
        outputs = [question, history, context, plot, turn, measurements, retrieval, progress]
        for event in [send.click, question.submit]:
            event(chat, inputs, outputs).success(automatic_checks, [turn, baseline], [measurements, progress])
        clear.click(lambda b: ([], EMPTY_CONTEXT, None, measurement_rows(b), [], "Ask a question to measure answer quality."),
                    inputs=baseline, outputs=[history, context, turn, measurements, retrieval, progress])
    return demo

if __name__ == "__main__":
    create_app().queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", share=False,
        footer_links=[], theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"), css=CSS)
