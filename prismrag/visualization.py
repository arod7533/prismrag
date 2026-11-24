import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

def embedding_plot(index, query=None, found=()):
    if len(index.chunks) < 2:
        raise ValueError("At least two chunks are required for a projection.")
    projection = PCA(n_components=min(2, *index.vectors.shape))
    points = projection.fit_transform(index.vectors)
    if points.shape[1] == 1:
        points = np.column_stack([points[:, 0], np.zeros(len(points))])
    hits = {i for i, _, _ in found}
    frame = pd.DataFrame({"x": points[:, 0], "y": points[:, 1],
        "source": [d.metadata["source"] for d in index.chunks],
        "retrieval": ["Retrieved" if i in hits else "Other" for i in range(len(points))],
        "chunk": list(range(len(points))), "preview": [d.page_content[:120] for d in index.chunks]})
    fig = px.scatter(frame, x="x", y="y", color="source", symbol="retrieval", hover_data=["chunk", "preview"], template="plotly_dark")
    if query is not None:
        q = projection.transform([query])[0]
        fig.add_scatter(x=[q[0]], y=[q[1] if len(q)>1 else 0], mode="markers", marker={"symbol":"star", "size":20, "color":"white"}, name="Query")
    variance = float(np.nan_to_num(projection.explained_variance_ratio_).sum())
    fig.update_layout(title=f"PCA · {variance:.1%} variance retained · color = source", xaxis_title="Principal component 1", yaxis_title="Principal component 2")
    return fig
