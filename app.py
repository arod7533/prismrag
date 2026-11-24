from functools import lru_cache
from pathlib import Path
import html
import gradio as gr
from prismrag.pipeline import build_index, answer
from prismrag.automatic import benchmark, measurement_rows, waiting_rows, judge_turn, save_turn, reference_case
from prismrag.visualization import embedding_plot, boundary_view, segmentation_diagnostics
from prismrag.learning import learning_guidance, metric_glossary

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
        yield index, baseline, "Documents ready. Ask a suggested question or enter your own.", corpus, rows, atlas, diagnostics, boundaries, chunk_text, learning_guidance(index, baseline)
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
                turn, measurement_rows(baseline, turn), table, "Answer ready. Quality checks are running automatically…", learning_guidance(index, baseline, turn))
    except Exception as exc:
        raise gr.Error(str(exc))

def automatic_checks(turn, baseline, index):
    if not turn:
        return measurement_rows(baseline), "Ask a question to measure answer quality.", learning_guidance(index, baseline)
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
    return rows, status, learning_guidance(index, baseline, turn, judgments)

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
            gr.Markdown("## Automatic measurements\nEverything updates automatically. Retrieval scores use the bundled labeled questions; answer quality and latency describe your latest chat response.")
            gr.Markdown("**1 · Retrieve:** check whether the right evidence was found. **2 · Generate:** check support, relevance and accuracy. **3 · Inspect chunks:** check boundaries and topic neighborhoods. **4 · Improve:** change one thing and measure again.")
            corpus = gr.Markdown("Loading corpus…")
            progress = gr.Markdown("Ask a question to measure answer quality.")
            measurements = gr.Dataframe(headers=["Measurement", "Result", "What it means"], value=waiting_rows(), interactive=False, wrap=True, label="Evaluations & measurements")
            guidance = gr.Markdown("Suggestions appear after the documents load.")
            with gr.Accordion("Learn each metric: why it matters and what to try", open=False):
                gr.Markdown(metric_glossary())
            gr.Markdown("### Latest retrieval\nCosine similarity is proximity, not confidence. Each message retrieves independently.")
            retrieval = gr.Dataframe(headers=["Rank", "Source", "Page", "Cosine similarity"], value=[], interactive=False, label="Retrieved passages")
            gr.Markdown("## Are the chunks segmented sensibly?\nCheck two different things: **text boundaries** (did we preserve useful passages?) and **embedding neighborhoods** (are related topics close?). A scatterplot cannot establish both.")
            diagnostics = gr.Markdown("Computing chunk diagnostics…")
            gr.Markdown("### Topic neighborhoods\nThe six colors are human-labeled subjects, shared across related documents. Nearby chunks of the same color suggest topic grouping; mixed colors can be legitimate cross-topic relationships. White diamonds mark retrieved evidence; the star marks your question. Hover for the source and chunk preview. The PCA view is lossy—use the full-vector diagnostics above, not just visual spacing.")
            plot = gr.Plot()
            gr.Markdown("### Inspect actual chunk boundaries\nThe default pricing example is already populated. Optionally inspect another document. Bars show exact character ranges within each source page; the cards show the complete text of each chunk. Check that a passage keeps its meaning and necessary qualifications. Short chunks and incomplete endings are review hints, not automatic failures.")
            source_choices = sorted(p.name for p in (ROOT / "data/corpus").iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"})
            inspect_source = gr.Dropdown(choices=source_choices, value="pricing-current.md", label="Document to inspect")
            boundaries = gr.Plot(label="Chunk boundaries")
            chunk_text = gr.HTML()
            gr.Markdown("The company and documents are fictional. Bundled questions are development examples, not a held-out benchmark. Judge scores are fallible estimates from the answer model. Answer accuracy and token F1 require an exact bundled question with a reference answer. Accuracy is reference-based model judging, not an independently verified overall accuracy rate. Reports save automatically to `runs/` locally.")
        demo.load(initialize, outputs=[index, baseline, ready, corpus, measurements, plot, diagnostics, boundaries, chunk_text, guidance])
        inspect_source.change(lambda idx, source: boundary_view(idx, source) if idx else (None, "Documents are still loading."), [index, inspect_source], [boundaries, chunk_text])
        inputs = [question, history, index, baseline]
        outputs = [question, history, context, plot, turn, measurements, retrieval, progress, guidance]
        for event in [send.click, question.submit]:
            event(chat, inputs, outputs).success(automatic_checks, [turn, baseline, index], [measurements, progress, guidance])
        clear.click(lambda b, idx: ([], EMPTY_CONTEXT, None, measurement_rows(b), [], "Ask a question to measure answer quality.", learning_guidance(idx,b)),
                    inputs=[baseline,index], outputs=[history, context, turn, measurements, retrieval, progress, guidance])
    return demo

if __name__ == "__main__":
    create_app().queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", share=False,
        footer_links=[], theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"), css=CSS)
