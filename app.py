from functools import lru_cache
from pathlib import Path
import html
import gradio as gr
from prismrag.pipeline import build_index, answer
from prismrag.automatic import benchmark, measurement_rows, waiting_rows, judge_turn, save_turn, reference_case
from prismrag.visualization import embedding_plot, boundary_view, segmentation_diagnostics
from prismrag.dashboard import simple_dashboard

ROOT = Path(__file__).parent
ANSWER_MODEL = "qwen3.5:9b"
EMBEDDING_MODEL = "nomic-embed-text"
K = 4
EXAMPLE_QUESTIONS = [
    "What does the Growth plan cost per month now?",
    "Which plan do I need for Warehouse Sync and hourly refreshes?",
    "How long does Growth retain imported data?",
    "Can a Viewer export raw data?",
    "Is scheduled PDF delivery available?",
    "What should I do after an API HTTP 429 response?",
]
EMPTY_CONTEXT = "Passages used to answer your question will appear here."
CSS = """
.gradio-container { max-width: 1480px !important; }
#context-panel { height: 520px; overflow-y: auto; padding: 20px; }
#context-panel h3 { color: #c084fc; }
#context-passages { line-height: 1.7; overflow-wrap: anywhere; }
.scorecards { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; margin:12px 0; }
.scorecard { padding:24px; border:1px solid var(--border-color-primary); border-radius:12px; background:var(--block-background-fill); }
.metric-strip { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }
.metric-item { padding:16px 20px; border:1px solid var(--border-color-primary); border-radius:10px; }
.metric-value { font-size:26px; font-weight:650; margin:8px 0; }
.metric-group-title { font-size:18px; margin:24px 0 12px; }
.scorecard, .metric-item { position:relative; }
.metric-help { display:inline-block; margin-left:8px; font-weight:400; }
.metric-help > summary { display:inline-block; cursor:help; list-style:none; font-size:17px; color:var(--body-text-color-subdued); border-radius:4px; }
.metric-help > summary::-webkit-details-marker { display:none; }
.metric-help > summary:focus-visible { outline:2px solid #a78bfa; outline-offset:3px; }
.metric-help > .metric-tooltip { display:none; position:absolute; top:48px; left:10px; right:10px; z-index:50; padding:16px; border-radius:10px; border:1px solid #64748b; background:#172033; color:#f1f5f9; box-shadow:0 8px 30px #0006; font-size:14px; line-height:1.5; }
.scorecard:hover .metric-tooltip, .metric-item:hover .metric-tooltip, .metric-help:focus-within > .metric-tooltip, .metric-help[open] > .metric-tooltip { display:block; }
.metric-tooltip p { margin:8px 0; }
.metric-tooltip small { color:#cbd5e1; }
.scorecard-label { font-size:16px; font-weight:600; }
.scorecard-value { font-size:38px; font-weight:700; line-height:1.4; margin:12px 0; }
.scorecard-note { font-size:14px; color:var(--body-text-color-subdued); }
@media (max-width:700px) { .scorecards, .metric-strip { grid-template-columns:1fr; } }
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
        diagnostics = segmentation_diagnostics(index)
        boundaries, chunk_text = boundary_view(index, "pricing-current.md")
        # Publish the ready state only after labels/benchmark are available, avoiding unlabeled early chats.
        try:
            baseline = benchmark(index, ROOT / "data/eval.json", K)
            rows = measurement_rows(baseline)
        except Exception as exc:
            baseline = {"error": str(exc), "corpus": index.fingerprint}
            rows = [[name, "Benchmark unavailable" if name in {"Precision@k", "Recall@k", "MRR", "nDCG@k"} else value, description] for name, value, description in waiting_rows()]
            corpus += f"\n\nBenchmark unavailable: {exc}"
        yield index, baseline, "Documents ready. Ask a suggested question or enter your own.", corpus, rows, atlas, diagnostics, boundaries, chunk_text, ""
    except Exception as exc:
        yield None, {}, f"Documents could not load. Check Ollama is running with {ANSWER_MODEL} and {EMBEDDING_MODEL}. {exc}", "Corpus unavailable.", waiting_rows(), None, "Diagnostics unavailable.", None, "", "Suggestions appear after the documents load."

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
                turn, measurement_rows(baseline, turn), table, "Answer ready. Quality checks are running automatically…", "")
    except Exception as exc:
        raise gr.Error(str(exc))

def automatic_checks(turn, baseline, index):
    if not turn:
        return measurement_rows(baseline), "Ask a question to measure answer quality.", ""
    judgments = None
    try:
        case = reference_case(baseline, turn["question"])
        turn = dict(turn, reference=case["reference"] if case else None)
        judgments = judge_turn(turn, ANSWER_MODEL, reference=turn["reference"])
        rows = measurement_rows(baseline, turn, judgments=judgments)
        status = "Latest answer evaluated automatically. Local judge estimates need human review."
    except Exception as exc:
        rows = measurement_rows(baseline, turn, judge_error=str(exc))
        status = "Answer available. Local quality judging failed; retrieval and timing measurements remain available."
    try:
        save_turn(turn, baseline, rows)
    except Exception:
        status += " Run could not be saved to disk."
    return rows, status, ""

def create_app():
    with gr.Blocks(title="PrismRAG") as demo:
        index = gr.State(None)
        baseline = gr.State({})
        turn = gr.State(None)
        gr.Markdown("# PrismRAG\nAsk Northstar Analytics’ knowledge base. See the sources.")
        with gr.Tab("Chat"):
            gr.Markdown("**Meet Northstar Analytics:** a fictional company that sells Harbor, an analytics tool for retail stores. Its knowledge base covers plans, billing, security, data imports and support. Try a question below—no company knowledge needed.")
            ready = gr.Markdown("Loading the included documents automatically…")
            with gr.Row():
                with gr.Column(scale=1):
                    history = gr.Chatbot(label="Conversation", height=520)
                    question = gr.Textbox(label="Message", show_label=False, placeholder="What does the Growth plan cost per month now?")
                    with gr.Row():
                        send = gr.Button("Send", variant="primary")
                        clear = gr.Button("Clear chat")
                    gr.Examples(examples=[[q] for q in EXAMPLE_QUESTIONS], inputs=question,
                        label="Try a question · select one, then Send", examples_per_page=6)
                with gr.Column(scale=1):
                    with gr.Group(elem_id="context-panel"):
                        gr.Markdown("### Relevant context")
                        context = gr.Markdown(EMPTY_CONTEXT, elem_id="context-passages")
        with gr.Tab("Measurements"):
            gr.Markdown("## RAG performance")
            measurements = gr.State(waiting_rows())
            corpus = gr.State("")
            guidance = gr.State("")
            progress = gr.State("")
            cards = gr.HTML()
            summary = gr.Markdown("Checking the documents…")
            with gr.Accordion("Score details", open=False):
                details = gr.Markdown()
                retrieval = gr.Dataframe(headers=["Rank", "Source", "Page", "Cosine similarity"], value=[], interactive=False, label="Latest retrieved sources")
            with gr.Accordion("Explore chunks", open=False):
                diagnostics = gr.Markdown()
                gr.Markdown("Topic map · for exploration, not an accuracy score.")
                plot = gr.Plot()
                source_choices = sorted(p.name for p in (ROOT / "data/corpus").iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"})
                inspect_source = gr.Dropdown(choices=source_choices, value="pricing-current.md", label="Document")
                boundaries = gr.Plot(label="Chunk boundaries")
                chunk_text = gr.HTML()
        measurements.change(simple_dashboard, [measurements, baseline, turn], [cards, summary, details])
        demo.load(initialize, outputs=[index, baseline, ready, corpus, measurements, plot, diagnostics, boundaries, chunk_text, guidance]).then(simple_dashboard, [measurements, baseline, turn], [cards, summary, details])
        inspect_source.change(lambda idx, source: boundary_view(idx, source) if idx else (None, "Documents are still loading."), [index, inspect_source], [boundaries, chunk_text])
        inputs = [question, history, index, baseline]
        outputs = [question, history, context, plot, turn, measurements, retrieval, progress, guidance]
        for event in [send.click, question.submit]:
            event(chat, inputs, outputs).success(automatic_checks, [turn, baseline, index], [measurements, progress, guidance])
        clear.click(lambda b, idx: ([], EMPTY_CONTEXT, None, measurement_rows(b), [], "Ask a question to measure answer quality.", ""),
                    inputs=[baseline,index], outputs=[history, context, turn, measurements, retrieval, progress, guidance])
    return demo

if __name__ == "__main__":
    create_app().queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", share=False,
        footer_links=[], theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"), css=CSS)
