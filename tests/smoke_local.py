"""Optional integration test requiring local Ollama models."""
from pathlib import Path
from prismrag.pipeline import build_index, answer
from prismrag.evaluation import evaluate, read_cases
from prismrag.visualization import embedding_plot
index = build_index(sorted(Path('data/corpus').glob('*.md')), 'nomic-embed-text')
result, found, query, latency = answer(index, 'What does chunk overlap do?', 'qwen3.5:9b', 2)
assert result and found
assert found[0][1].metadata['source'] == 'retrieval.md'
assert len(embedding_plot(index, query, found).data) > 1
rows, path = evaluate(index, read_cases('data/eval.json', index)[:1], 'qwen3.5:9b', 'qwen3.5:9b', 2, ['Recall@k', 'Faithfulness', 'Latency'])
assert rows[0]['Recall@k'] == 1
assert rows[0]['Faithfulness'] is not None, rows
print('Local integration passed:',len(index.chunks),'chunks; answer:',result,'; report:',path)
