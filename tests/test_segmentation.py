from types import SimpleNamespace
import numpy as np
from langchain_core.documents import Document
from prismrag.visualization import chunk_spans, boundary_view, embedding_plot, segmentation_diagnostics
from prismrag.pipeline import load_documents
from prismrag.topics import TOPICS


def test_boundaries_and_actual_overlap_are_not_the_configured_target():
    chunks = [Document(page_content='Alpha beta.',metadata={'source':'pricing-current.md','page':1,'start_index':0}),
              Document(page_content='beta. Gamma.',metadata={'source':'pricing-current.md','page':1,'start_index':6})]
    idx=SimpleNamespace(chunks=chunks,vectors=np.array([[1.,0],[0.,1.]]),fingerprint='test',chunk_size=700,overlap=100)
    spans=chunk_spans(idx,'pricing-current.md')
    assert spans[1]['overlap']==5
    assert spans[1]['end']==18
    figure, text=boundary_view(idx,'pricing-current.md')
    assert figure.data[1].base[0]==6
    assert '5 overlapping characters' in text
    assert 'median 5 characters' in segmentation_diagnostics(idx)


def test_boundary_text_is_escaped_and_unknown_offsets_are_not_invented():
    idx=SimpleNamespace(chunks=[Document(page_content='<script>bad()</script>',metadata={'source':'x.md'})])
    fig, text=boundary_view(idx,'x.md')
    assert '<script>' not in text
    assert '&lt;script&gt;' in text
    assert chunk_spans(idx,'x.md')[0]['start'] is None


def test_topic_colors_are_shared_across_sources_and_retrieval_is_an_overlay():
    docs=[Document(page_content='text.',metadata={'source':name}) for name in ['pricing-current.md','pricing-archive.md','security.md']]
    idx=SimpleNamespace(chunks=docs,vectors=np.array([[1.,0,0],[.99,.1,0],[0.,0,1.]]),fingerprint='test')
    fig=embedding_plot(idx,found=[(0,docs[0],1.)])
    assert [trace.name for trace in fig.data]==['Product & plans','Security & access','Retrieved evidence']
    assert len(fig.data[0].x)==2


def test_all_corpus_sources_have_authored_topics_and_exact_offsets():
    from pathlib import Path
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    paths=sorted(Path('data/corpus').glob('*.md'))
    assert len(paths)==24
    assert {p.name for p in paths}==set(TOPICS)
    originals={doc.metadata['source']:doc for doc in load_documents(paths)}
    chunks=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=100,add_start_index=True).split_documents(list(originals.values()))
    for doc in chunks:
        start=doc.metadata['start_index']
        assert start>=0
        assert originals[doc.metadata['source']].page_content[start:start+len(doc.page_content)]==doc.page_content
