import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLOR_PALETTE = [
    (31.0, 119.0, 180.0),
    (255.0, 127.0, 14.0),
    (44.0, 160.0, 44.0),
    (214.0, 39.0, 40.0),
    (148.0, 103.0, 189.0),
    (140.0, 86.0, 75.0),
    (227.0, 119.0, 194.0),
    (127.0, 127.0, 127.0),
    (188.0, 189.0, 34.0),
    (23.0, 190.0, 207.0),
]


def get_colors(n, alpha):
    if n == -1:
        colors = COLOR_PALETTE
    else:
        colors = COLOR_PALETTE[:n]
    return [f"rgba({color[0]}, {color[1]}, {color[2]}, {alpha})" for color in colors]


def _plot_hue(df_, name, label, color, x, y):
    rgb = f"rgba({color[0]}, {color[1]}, {color[2]}"
    fill_rgba = f"{rgb},0)"
    line_rgba = f"{rgb},1)"
    fig = px.line(
        df_,
        x=x,
        y="value",
        color="variable",
        color_discrete_map={f"{y}_lower": fill_rgba, f"{y}_upper": fill_rgba},
    )

    fig.update_traces(selector=dict(name=f"{y}_upper"), showlegend=False)
    fig.update_traces(fill="tonexty")
    if name is None:
        fig.update_traces(
            fillcolor="rgba(0,0,0,0)",
            selector=dict(name=y),
            line_color=line_rgba,
            showlegend=False,
        )
    else:
        fig.update_traces(
            fillcolor="rgba(0,0,0,0)",
            selector=dict(name=name),
            name=label,
            line_color=line_rgba,
        )
    fig.update_traces(
        fillcolor="rgba(0,0,0,0)",
        showlegend=False,
        selector=dict(name=f"{y}_lower"),
    )
    return fig


def lineplot(data, x, y, hue=None, color_palette=None):

    if hue is None:
        group = [x]
    else:
        group = [hue, x]

    n_ = data.groupby(group)[y].size()
    if (n_ == 1).all():
        return px.line(
            data.sort_values(x),
            x=x,
            y=y,
            color=hue,
            color_discrete_sequence=get_colors(-1, 1),
        )

    err = data.groupby(group)[y].std() / np.sqrt(n_)
    pdf = data.groupby(group)[y].mean().reset_index()
    pdf[f"{y}_upper"] = pdf[y] + 1.96 * pdf.set_index(group).index.map(err)
    pdf[f"{y}_lower"] = pdf[y] - 1.96 * pdf.set_index(group).index.map(err)
    pdf = pd.melt(pdf, id_vars=group)

    if hue is None:
        ascending = [True, True]
    else:
        ascending = [False, True, True]
    pdf = pdf.sort_values(group + ["variable"], ascending=ascending)

    figs = []
    if color_palette is None:
        color_palette = COLOR_PALETTE
    if hue is not None:
        for hue_, color in zip(pdf[hue].unique(), color_palette):
            df_ = pdf[pdf[hue] == hue_].copy()
            figs.append(_plot_hue(df_, y, hue_, color, x, y))
    else:
        figs.append(_plot_hue(pdf, None, None, color_palette[0], x, y))

    if not figs:
        return go.Figure()

    data = figs[0].data
    for fig in figs[1:]:
        data = data + fig.data
    figure = go.Figure(data=data)

    if hue is None:
        figure.update_layout(
            xaxis_title=x,
            yaxis_title=y,
        )
    else:
        figure.update_layout(
            xaxis_title=x,
            yaxis_title=y,
            legend_title=hue,
        )
    return figure
