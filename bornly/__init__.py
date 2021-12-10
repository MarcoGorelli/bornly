import functools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from plotly.subplots import make_subplots
from bornly._seaborn.seaborn.relational import _ScatterPlotter

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


class Ax:
    def __init__(self, func, *, nrows, ncols):
        self._func = func
        self._row = func.keywords["row"]
        self._col = func.keywords["col"]
        self._nrows = nrows
        self._ncols = ncols

    def __call__(self, figure):
        return self._func(figure)

    @property
    def _figure(self):
        return self._func.keywords["figure"]

    def _find_annotation(self):
        return self._figure.layout.annotations[self._row * self._ncols + self._col]

    def set_title(self, title):
        annotation = self._find_annotation()
        if title is not None:
            annotation.update(text=title)
        else:
            annotation.update(text="")

    def set_ylabel(self, label):
        self._figure.update_yaxes(
            title_text=label, row=self._row + 1, col=self._col + 1
        )

    def set_xlabel(self, label):
        self._figure.update_xaxes(
            title_text=label, row=self._row + 1, col=self._col + 1
        )

    def set_ylim(self, ylim):
        self._figure.update_yaxes(range=ylim, row=self._row + 1, col=self._col + 1)

    def set_xlim(self, xlim):
        self._figure.update_xaxes(range=xlim, row=self._row + 1, col=self._col + 1)

    def set(self, **kwargs):
        for key, val in kwargs.items():
            getattr(self, f"set_{key}")(val)


def _add_to_fig(subplot, figure, row, col):
    for data in subplot.data:
        figure.add_trace(data, row=row + 1, col=col + 1)


def subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, **kwargs):
    fig = make_subplots(
        nrows,
        ncols,
        shared_xaxes=sharex,
        shared_yaxes=sharey,
        subplot_titles=["placeholder" for _ in range(nrows) for __ in range(ncols)],
    )
    for annotation in fig.layout.annotations:
        annotation.update(text="")
    ax = [
        [
            Ax(
                functools.partial(_add_to_fig, figure=fig, row=row, col=col),
                nrows=nrows,
                ncols=ncols,
            )
            for col in range(ncols)
        ]
        for row in range(nrows)
    ]
    if (nrows == 1) and (ncols == 1):
        return fig, ax[0][0]
    if (nrows == 1) or (ncols == 1):
        return fig, np.asarray(ax).flatten()
    return fig, np.asarray(ax)


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


def lineplot(data, x, y, hue=None, ax=None):

    if hue is None:
        group = [x]
    else:
        group = [hue, x]

    n_ = data.groupby(group)[y].size()
    if (n_ == 1).all():
        figure = px.line(
            data.sort_values(group),
            x=x,
            y=y,
            color=hue,
            color_discrete_sequence=get_colors(-1, 1),
        )
    else:
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
        if hue is not None:
            for hue_, color in zip(pdf[hue].unique(), COLOR_PALETTE):
                df_ = pdf[pdf[hue] == hue_].copy()
                figs.append(_plot_hue(df_, y, hue_, color, x, y))
        else:
            figs.append(_plot_hue(pdf, None, None, COLOR_PALETTE[0], x, y))

        if figs:
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

    if ax is None:
        return figure
    else:
        names_already_in_legend = {i.name for i in ax._figure.data if i.showlegend}
        for i in figure.data:
            if i.showlegend and i.name in names_already_in_legend:
                i.showlegend = False
        ax(figure)

def scatterplot1(data, x, y, hue=None, ax=None):

    if hue is None:
        group = [x]
    else:
        group = [hue, x]

    figure = px.scatter(
        data.sort_values(group),
        x=x,
        y=y,
        color=hue,
        color_discrete_sequence=get_colors(-1, 1),
    )

    if ax is None:
        return figure
    else:
        names_already_in_legend = {i.name for i in ax._figure.data if i.showlegend}
        for i in figure.data:
            if i.showlegend and i.name in names_already_in_legend:
                i.showlegend = False
        ax(figure)

class ScatterPlotter(_ScatterPlotter):

    @property
    def reverse_variables(self):
        return {v: k for k, v in self.variables.items()}
    
    def plot(self, ax, kws):

        # --- Determine the visual attributes of the plot

        data = self.plot_data.dropna().rename(columns=self.variables)
        if data.empty:
            return

        # Draw the scatter plot
        plotting_kwargs = {
            'data_frame': data,
            'x': self.variables['x'],
            'y': self.variables['y'],
            'color_discrete_sequence': get_colors(-1, 1),
        }
        if 'hue' in self.variables:
            plotting_kwargs = {**plotting_kwargs, **{'color': self.variables['hue']}}
        if 'size' in self.variables:
            plotting_kwargs = {**plotting_kwargs, **{'size': self.variables['size']}}
        fig = px.scatter(**plotting_kwargs)
        ax.set_xlabel(self.variables['x'])
        ax.set_ylabel(self.variables['y'])

        if not self.legend:
            for i in fig.data:
                i.showlegend = False
        
        ax(fig)

        
def scatterplot(
    *,
    x=None, y=None,
    hue=None, style=None, size=None, data=None,
    palette=None, hue_order=None, hue_norm=None,
    sizes=None, size_order=None, size_norm=None,
    markers=True, style_order=None,
    x_bins=None, y_bins=None,
    units=None, estimator=None, ci=95, n_boot=1000,
    alpha=None, x_jitter=None, y_jitter=None,
    legend="auto", ax=None,
    **kwargs
):

    variables = ScatterPlotter.get_semantics(locals())
    p = ScatterPlotter(
        data=data, variables=variables,
        x_bins=x_bins, y_bins=y_bins,
        estimator=estimator, ci=ci, n_boot=n_boot,
        alpha=alpha, x_jitter=x_jitter, y_jitter=y_jitter, legend=legend,
    )

    if ax is None:
        _, ax = subplots()

    if not p.has_xy_data:
        return ax._figure

    p.plot(ax, kwargs)
    return ax._figure

