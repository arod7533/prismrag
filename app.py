from functools import lru_cache
from pathlib import Path
import html
import gradio as gr
from prismrag.pipeline import build_index, answer
from prismrag.automatic import benchmark, measurement_rows, waiting_rows, judge_turn, save_turn, reference_case
from prismrag.visualization import embedding_plot, boundary_view, segmentation_diagnostics
from prismrag.learning import learning_guidance, metric_glossary
from prismrag.evaluation_view import evaluation_sections

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
            gr.Markdown("## Understand the evaluation\nFollow the evidence, then the answer. Everything runs automatically. **The first section covers a fixed question set; the second covers only your latest chat response.**")
            corpus = gr.Markdown("Loading corpus…")
            measurements = gr.State(waiting_rows())
            gr.Markdown("### 1. Did we retrieve the right evidence?\n**Method:** run the 20 bundled questions through retrieval with top-k 4. Compare retrieved source filenames with the curated evidence labels, then average each metric across questions. No answer model or model judge is involved in these four scores. Values range from 0 to 1.")
            retrieval_scores = gr.Markdown("Running the retrieval benchmark…")
            with gr.Accordion("Walk through a real evaluation", open=False):
                worked_example = gr.Markdown("Loading a measured example…")
            gr.Markdown("### 2. Is the latest answer good?\n**Method:** after each chat response, a local model judges support, relevance and context usefulness. For a question with a curated reference, a separate call compares the answer with a curated reference to estimate accuracy. These 0–1 scores are fallible judgments, not independently verified facts.")
            progress = gr.Markdown("Ask a question in Chat to evaluate an answer.")
            answer_scores = gr.Markdown("Answer measurements appear after you chat.")
            with gr.Accordion("Discuss the latest answer: evidence, reference and judge reasons", open=False):
                answer_discussion = gr.Markdown("Ask a question to see its evaluation discussion.")
                retrieval = gr.Dataframe(headers=["Rank", "Source", "Page", "Cosine similarity"], value=[], interactive=False, label="Passages supplied to the answer model")
                gr.Markdown("The exact passages are visible beside Chat. Cosine similarity is vector proximity, not confidence or correctness.")
            gr.Markdown("### 3. What should we try next?\nUse the discussion below to choose one experiment. Compare the same question set before and after a change; do not optimize a score without reading the answers.")
            guidance = gr.Markdown("Recommendations appear after the benchmark finishes.")
            with gr.Accordion("Evaluation design and limitations", open=False):
                gr.Markdown("**What is labeled?** The 20 demonstration questions have reference answers and relevant source filenames authored alongside the fictional documents. These are development examples, not an independent held-out test. Precision can undercount useful sources omitted from the labels.\n\n**What does the judge see?** Support and relevance judging receives the question, actual answer and retrieved context. Accuracy judging separately receives the question, actual answer and reference. Exact question matching ignores case and outer whitespace; arbitrary questions do not gain an invented reference.\n\n**What can go wrong?** The same local model answers and judges, creating correlated bias. A faithful answer can repeat outdated context; an accurate answer can use different wording from the reference. Missing or failed checks are shown as unavailable, never zero.\n\n**How would we strengthen this?** Add independently reviewed paraphrases, multi-document and unanswerable cases; reserve a held-out set; compare with a separate stronger judge and human reviews. Save the same dataset and configuration when comparing runs. Reports save automatically to the local runs folder.")
                gr.Markdown(metric_glossary())
            with gr.Accordion("Optional diagnostics: chunking and embedding neighborhoods", open=False):
                gr.Markdown("#### Is the scatterplot relevant?\nIt is an **exploration aid**, not an evaluation score. It may reveal related passages, odd neighbors or where a query lands. PCA compresses hundreds of dimensions into two, so apparent distances and clusters can mislead. Use the labeled retrieval results above to assess usefulness; inspect actual text boundaries to assess splitting.")
                diagnostics = gr.Markdown("Computing chunk diagnostics…")
                with gr.Accordion("Explore the topic scatterplot", open=False):
                    gr.Markdown("Colors are human-authored topics; white diamonds mark retrieved chunks and the star marks your question. Hover for the source and text preview. Topic separation uses full vectors, while this map is a lossy projection. Neither establishes that a chunk preserves its meaning.")
                    plot = gr.Plot()
                gr.Markdown("#### Inspect the actual splitting\nThe pricing document is preselected. Bars show character ranges; cards show the full chunk text. Read across the boundary: did a rule lose its condition or a paragraph lose its heading? Size/ending flags are review hints, not failures. Changing the document below is optional exploration, not evaluation setup.")
                source_choices = sorted(p.name for p in (ROOT / "data/corpus").iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf"})
                inspect_source = gr.Dropdown(choices=source_choices, value="pricing-current.md", label="Document to inspect")
                boundaries = gr.Plot(label="Chunk boundaries")
                chunk_text = gr.HTML()
        measurements.change(evaluation_sections, [measurements, baseline, turn], [retrieval_scores, worked_example, answer_scores, answer_discussion])
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
