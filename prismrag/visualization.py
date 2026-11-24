import html
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from .topics import TOPIC_COLORS, topic_for


def embedding_plot(index, query=None, found=()):
    if len(index.chunks) < 2:
        raise ValueError("At least two chunks are required for a projection.")
    projection = PCA(n_components=min(2, *index.vectors.shape))
    points = projection.fit_transform(index.vectors)
    if points.shape[1] == 1:
        points = np.column_stack([points[:, 0], np.zeros(len(points))])
    fig = go.Figure()
    labels = [topic_for(d) for d in index.chunks]
    for topic, color in TOPIC_COLORS.items():
        ids = [i for i, value in enumerate(labels) if value == topic]
        if not ids:
            continue
        custom = [[i + 1, html.escape(index.chunks[i].metadata['source']), len(index.chunks[i].page_content),
                   html.escape(index.chunks[i].page_content[:160])] for i in ids]
        fig.add_scatter(x=points[ids, 0], y=points[ids, 1], mode="markers", name=topic,
            marker={"size":12, "color":color, "line":{"width":1,"color":"#101827"}}, customdata=custom,
            hovertemplate="Chunk %{customdata[0]} · %{customdata[1]}<br>%{customdata[2]} characters<br>%{customdata[3]}<extra>" + topic + "</extra>")
    hits = [i for i, _, _ in found]
    if hits:
        fig.add_scatter(x=points[hits,0], y=points[hits,1], mode="markers", name="Retrieved evidence",
            marker={"symbol":"diamond-open", "size":20, "color":"white", "line":{"width":2}}, hoverinfo="skip")
    if query is not None:
        q = projection.transform([query])[0]
        fig.add_scatter(x=[q[0]], y=[q[1] if len(q)>1 else 0], mode="markers", marker={"symbol":"star","size":20,"color":"white"}, name="Your question")
    variance = float(np.nan_to_num(projection.explained_variance_ratio_).sum())
    fig.update_layout(template="plotly_dark", title=f"Chunks by topic · PCA retains {variance:.1%} of variance",
        height=510, xaxis_title="Principal component 1", yaxis_title="Principal component 2",
        legend={"orientation":"h", "y":-0.2,"x":0}, margin={"b":130}, uirevision=index.fingerprint)
    return fig


def chunk_spans(index, source):
    selected = [(i, d) for i, d in enumerate(index.chunks) if d.metadata['source'] == source]
    selected.sort(key=lambda item: (item[1].metadata.get('page',1), item[1].metadata.get('start_index',-1),item[0]))
    rows, previous = [], {}
    for i, doc in selected:
        page = doc.metadata.get('page',1)
        start = doc.metadata.get('start_index',-1)
        end = start + len(doc.page_content) if start >= 0 else None
        prior = previous.get(page)
        overlap = max(0, prior - start) if prior is not None and start >= 0 else (0 if start >= 0 else None)
        flags = []
        if len(doc.page_content) < 100:
            flags.append("Short (<100 characters)")
        if doc.page_content.rstrip() and doc.page_content.rstrip()[-1] not in '.!?。！？':
            flags.append("Check ending: may be a heading, list or split sentence")
        rows.append({'id':i+1,'page':page,'start':start if start>=0 else None,'end':end,
                     'characters':len(doc.page_content),'overlap':overlap,'text':doc.page_content,
                     'review':'; '.join(flags) or 'No length/ending flags'})
        if end is not None:
            previous[page] = max(prior or 0,end)
    return rows


def boundary_view(index, source):
    rows = chunk_spans(index, source)
    if not rows:
        return None, "<p>No chunks available for this document.</p>"
    doc = next(d for d in index.chunks if d.metadata['source']==source)
    color = TOPIC_COLORS[topic_for(doc)]
    fig = go.Figure()
    for row in rows:
        if row['start'] is None:
            continue
        fig.add_bar(x=[row['characters']], y=[f"Chunk {row['id']}"], base=[row['start']], orientation='h',
            marker={"color":color,"line":{"width":1,"color":"white"}}, name=f"Chunk {row['id']}",
            customdata=[[row['start'],row['end'],row['overlap'],row['page']]],
            hovertemplate="Page %{customdata[3]} · characters [%{customdata[0]}, %{customdata[1]})<br>%{x} characters · %{customdata[2]} overlap with previous chunk<extra></extra>",showlegend=False)
    fig.update_layout(template="plotly_dark",title=f"Actual chunk boundaries · {source}",height=max(240,70*len(rows)+130),
        xaxis_title="Character offset within the source page (zero-based)",yaxis={"autorange":"reversed"}, barmode="overlay")
    cards = []
    for row in rows:
        interval = f"[{row['start']}, {row['end']})" if row['start'] is not None else 'not recorded'
        cards.append(f'''<article style="border-left:4px solid {color};border-radius:8px;padding:16px;margin:12px 0;background:var(--block-background-fill)">
<strong>Chunk {row['id']} · page {row['page']}</strong>
<p>{row['characters']} characters · offsets {interval} · {row['overlap'] if row['overlap'] is not None else 'unknown'} overlapping characters</p>
<p><em>{html.escape(row['review'])}</em></p>
<pre style="white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;line-height:1.6">{html.escape(row['text'])}</pre></article>''')
    return fig, ''.join(cards)


def segmentation_diagnostics(index):
    lengths = np.array([len(doc.page_content) for doc in index.chunks])
    topics = [topic_for(doc) for doc in index.chunks]
    sources = [doc.metadata['source'] for doc in index.chunks]
    similarity = np.clip(index.vectors @ index.vectors.T,-1,1)
    same_topic, same_doc = [], []
    for i in range(len(topics)):
        candidates = [j for j in range(len(topics)) if sources[j]!=sources[i]]
        if candidates:
            nearest = max(candidates,key=lambda j:similarity[i,j])
            same_topic.append(topics[i]==topics[nearest])
    for source in sorted(set(sources)):
        spans = chunk_spans(index, source)
        same_doc.extend(row['overlap'] for row in spans[1:] if row['overlap'] is not None)
    silhouette = silhouette_score(index.vectors,topics,metric='cosine') if 1<len(set(topics))<len(topics) else None
    report = [
        f"**Chunk sizes:** {lengths.min()}–{lengths.max()} characters; median {np.median(lengths):.0f}. {int((lengths<100).sum())} chunks are under 100 characters (a review flag, not an automatic failure).",
        f"**Actual overlap:** median {np.median(same_doc):.0f} characters between adjacent chunks in each document." if same_doc else "**Actual overlap:** no comparable adjacent chunks.",
        f"The splitter targets up to {index.chunk_size} characters with {index.overlap} characters of overlap; paragraph boundaries can produce less overlap, including zero.",
        f"**Topic neighbors:** {np.mean(same_topic):.0%} of chunks have a nearest chunk from another document with the same topic label." if same_topic else "**Topic neighbors:** unavailable with only one source document.",
        f"**Topic separation (cosine silhouette): {silhouette:.3f}.** Near 1 means clearer topic separation, near 0 means overlapping topics, and below 0 suggests closer neighbors in another labeled topic." if silhouette is not None else "**Topic separation:** requires multiple topics with repeated chunks.",
        "Neighbor and separation scores use the full embedding vectors, not PCA coordinates. Topics are human-authored broad labels, not discovered clusters; related policies can legitimately overlap. These diagnostics do not certify semantic chunk quality."
    ]
    return '\n\n'.join(report)
