import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLOR_PALETTE = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
]


def lineplot(data, x, y, hue=None, color_palette=None):
    def plot_hue(df_, name, label, color):
        rgb = f"rgba({color[0]*255}, {color[1]*255}, {color[2]*255}"
        fill_rgba = f"{rgb},0)"
        line_rgba = f"{rgb},1)"
        fig = px.line(
            df_,
            x=x,
            y="value",
            color="variable",
            color_discrete_map={f"{y}_lower": fill_rgba, f"{y}_upper": fill_rgba},
        )

        fig.update_traces(
            name="interval", selector=dict(name=f"{y}_upper"), showlegend=False
        )
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

    if hue is None:
        group = [x]
    else:
        group = [hue, x]
    err = data.groupby(group)[y].std() / np.sqrt(data.groupby(group)[y].size())
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
            figs.append(plot_hue(df_, y, hue_, color))
    else:
        figs.append(plot_hue(pdf, None, None, color_palette[0]))

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
