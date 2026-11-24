"""Optional integration check requiring Ollama and the installed local models."""
from pathlib import Path
from unittest.mock import patch
from prismrag.pipeline import build_index, answer
from prismrag.automatic import benchmark, judge_turn, measurement_rows
from prismrag.metrics import METRICS
from prismrag.visualization import boundary_view, segmentation_diagnostics
from prismrag.visualization import embedding_plot

paths = sorted(Path('data/corpus').glob('*.md'))
index = build_index(paths, 'nomic-embed-text')
assert len(paths) == 24
print('Indexed',len(index.chunks),'chunks from',len(paths),'documents',flush=True)
with patch('prismrag.pipeline.OllamaEmbeddings.embed_documents', side_effect=AssertionError('Re-embedded unchanged corpus')):
    reused = build_index(paths, 'nomic-embed-text')
assert index.fingerprint == reused.fingerprint
baseline = benchmark(index, 'data/eval.json', 4)
assert baseline['count'] == 20
print('Persistent reuse and automatic benchmark passed:',baseline['means'],flush=True)
question = baseline['cases'][0]['question']
result, found, query, seconds = answer(index, question, 'qwen3.5:9b', 4)
assert result and found
assert 'pricing-current.md' in [d.metadata['source'] for _,d,_ in found]
assert len(embedding_plot(index, query, found).data) > 1
turn = {'question':question,'answer':result,'seconds':seconds,'context':[d.page_content for _,d,_ in found]}
judgments = judge_turn(turn, 'qwen3.5:9b', reference=baseline['cases'][0]['reference'])
assert judgments.answer_accuracy is not None
assert 'Actual overlap' in segmentation_diagnostics(index)
assert 'Chunk' in boundary_view(index, 'pricing-current.md')[1]
assert all(0 <= getattr(judgments,key).score <= 1 for key in ['faithfulness','answer_relevance','context_relevance'])
assert len(measurement_rows(baseline,turn,judgments)) == len(METRICS)
print('Local answer, vector plot and automatic quality checks passed:',result,flush=True)
