import functools
from collections import Iterable
import matplotlib
import copy
import matplotlib as mpl
from matplotlib.pyplot import legend

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import bornly._seaborn.seaborn as _sns
from bornly._seaborn.seaborn import (
    color_palette,
    diverging_palette,
    cubehelix_palette,
    load_dataset,
)

__version__ = "0.2.4"


def _validate_pandas(*args):
    for data in args:
        if (
            isinstance(data, (pd.DataFrame, pd.Series))
            and data.index.has_duplicates
        ):
            raise ValueError(
                "Passed data with duplicate index. Please de-duplicate "
                "your index before passing it to bornly, "
                "e.g. df.reset_index(drop=True)"
            ) from None

def _dedupe_legend(fig):
    condition = lambda i: i.showlegend is not False and len(i.x) > 0
    in_legend = {i.name for i in fig.data if condition(i)}
    counts = {key: 0 for key in in_legend}
    for data_ in fig.data:
        if not condition(data_):
            continue
        counts[data_.name] += 1
        data_.showlegend = counts[data_.name] == 1

def _cartesian(x, y):
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])


def _deconvert_rgba(rgba):
    return tuple([float(i) / 255 for i in rgba.strip("rgba()").split(",")[:3]])


def _convert_color(color, alpha=1):
    if alpha is None:
        alpha = 1
    return f"rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {alpha})"


class Foo:
    update_units = lambda *_: None
    convert_units = lambda x, y: y
    get_scale = lambda *_: None  # TODO maybe, maybe, can do this
    grid = lambda x, y: None


class Line:
    def __init__(self, scatter, ax=None):
        self.scatter = scatter
        self.ax = ax

    def set_color(self, color):
        self.scatter.marker.update(color=_convert_color(color, 1))

    def get_color(self):
        return _deconvert_rgba(self.scatter.marker.color)

    def get_alpha(self):
        return float(self.scatter.marker.color.strip("rgba()").split(",")[-1])

    def get_solid_capstyle(self):
        # currently unused anyway
        pass  # huh

    def remove(self):
        # what's this for?
        pass

    def set_dashes(self, dashes):
        if dashes:
            self.scatter.line.update(dash=", ".join([str(i) for i in dashes]))

    def set_linewidth(self, width):
        # what's this for?
        pass

    def set_marker(self, marker):
        if marker == "o":
            symbol = "circle"
        elif marker == "X":
            symbol = "x"
        else:
            raise ValueError(f'Unsupported marker "{marker}", please report issue')
        self.scatter.marker.symbol = symbol

    @property
    def sticky_edges(self):
        class Foo:
            x = []
            y = []
        return Foo()

    def set_facecolors(self, facecolors):
        facecolors_series = pd.Series(facecolors)
        self.ax.figure.data = self.ax.figure.data[:-1]
        for facecolor in facecolors_series.unique():
            mask = facecolors_series == facecolor
            self.ax(
                go.Figure(
                    go.Scatter(
                        x=self.scatter.x[mask],
                        y=self.scatter.y[mask],
                        mode="markers",
                        marker=dict(color=_convert_color(facecolor, 1)),
                    )
                )
            )
        pass

    def get_sizes(self):
        return [1]

    def set_sizes(self, sizes):
        pass

    def set_linewidths(self, *args, **kwargs):
        return None


class Legend:
    def __init__(self, legend):
        self.legend = legend

    def findobj(self, obj):
        class Foo:
            def get_children(self):
                return []

        return [Foo()]


def _parse_color(color, alpha=None):
    if isinstance(color, str) and color.startswith("rgba("):
        return color
    elif isinstance(color, tuple) and len(color) == 4:
        color = _convert_color(color[:3], color[3])
    elif isinstance(color, tuple) and len(color) == 3:
        color = _convert_color(color, alpha)
    elif isinstance(color, str):
        try:
            color = _convert_color(mpl.colors.to_rgb(color), alpha)
        except ValueError:
            color = _convert_color((mpl.cm.hot(float(color))), alpha)
    return color


class Ax:
    def __init__(self, func, *, nrows, ncols):
        self._func = func
        self._row = func.keywords["row"]
        self._col = func.keywords["col"]
        self._nrows = nrows
        self._ncols = ncols
    
    def __repr__(self):
        return self.figure

    def barh(self, x, y, width, **kwargs):
        pass

    def bar(self, x, y, width, **kwargs):
        pass

    def set_xticks(self, xticks):
        pass

    def set_xticklabels(self, xticklabels):
        pass

    @property
    def yaxis(self):
        return Foo()

    @property
    def xaxis(self):
        return Foo()

    def get_xlabel(self, visible=None):
        return self.figure.layout.xaxis.title.text

    def get_ylabel(self, visible=None):
        return self.figure.layout.yaxis.title.text

    def get_legend_handles_labels(self):
        return self.figure.layout.legend, None

    def get_xticklabels(self):
        return []  # todo

    def get_yticklabels(self):
        return []  # todo

    def legend(self, title=None):
        if title is not None:
            self.figure.layout.legend.title = title
        return Legend(self.figure.layout.legend)

    def plot(self, x, y, **kwargs):
        if kwargs.get("color", None) is not None:
            color = kwargs.get("color", None)
            color = _parse_color(color)
        else:
            color = _get_colors(1, 1)[0]

        if kwargs.get("dashes") is not None:
            dash = ", ".join([str(i) for i in kwargs.get("dashes")])
            if dash == "":
                dash = None
        else:
            dash = None

        if kwargs.get("label") is not None:
            label = str(kwargs.get("label"))
        else:
            label = None

        fig = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            legendgroup=label,
            name=label,
            marker=dict(color=color),
            line=dict(dash=dash),
        )
        self.figure.add_trace(fig, row=self._row + 1, col=self._col + 1)
        return (Line(self.figure.data[-1], self),)

    def add_legend(self, legend_data, **kwargs):
        for key, val in legend_data.items():
            color = val.get_color()
            for _data in self.figure.data:
                if Line(_data).get_color() == color[:3]:
                    _data.legendgroup = key
        self.figure.layout.legend.title = kwargs.get("title")
        return Legend(self.figure.layout.legend)

    def __call__(self, figure):
        return self._func(figure)

    @property
    def figure(self):
        return self._func.keywords["figure"]

    def _find_annotation(self):
        return self.figure.layout.annotations[self._row * self._ncols + self._col]

    def set_title(self, title, *args, **kwargs):
        annotation = self._find_annotation()
        if title is not None:
            annotation.update(text=title)
        else:
            annotation.update(text="")

    def set_ylabel(self, label, visible=None):
        self.figure.update_yaxes(title_text=label, row=self._row + 1, col=self._col + 1)

    def set_xlabel(self, label, visible=None):
        self.figure.update_xaxes(title_text=label, row=self._row + 1, col=self._col + 1)

    def set_ylim(self, ylim):
        if not isinstance(ylim, Iterable):
            raise ValueError(
                "set_ylim can only be called with an iterable, "
                "e.g. set_xlim([0, 10])"
            )
        self.figure.update_yaxes(range=ylim, row=self._row + 1, col=self._col + 1)

    def set_xlim(self, xlim, *args, **kwargs):
        if not isinstance(xlim, Iterable):
            raise ValueError(
                "set_xlim can only be called with an iterable, "
                "e.g. set_xlim([0, 10])"
            )
        self.figure.update_xaxes(range=xlim, row=self._row + 1, col=self._col + 1)

    def set(self, **kwargs):
        for key, val in kwargs.items():
            getattr(self, f"set_{key}")(val)

    def fill_between(
        self,
        x,
        y1,
        y2,
        **kwargs,
    ):
        rgba = kwargs.get("rgba")
        if rgba is None:
            rgb = kwargs.get("color", None)
            alpha = kwargs.get("alpha", None)
            if rgb is not None:
                rgba = _parse_color(rgb, alpha)
            else:
                rgba = _get_colors(1, alpha)[0]
        self.figure.add_trace(
            go.Scatter(
                x=x,
                y=y2,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=self._row + 1,
            col=self._col + 1,
        )

        self.figure.add_trace(
            go.Scatter(
                x=x,
                y=y1,
                line=dict(color="rgba(0,0,0,0)"),
                fill="tonexty",
                fillcolor=rgba,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=self._row + 1,
            col=self._col + 1,
        )
        return self.figure


def _add_to_fig(subplot, figure, row, col):
    for data in subplot.data:
        figure.add_trace(data, row=row + 1, col=col + 1)


def subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, wrap=True, **kwargs):
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
    if not wrap:
        return fig, np.asarray(ax)
    if (nrows == 1) and (ncols == 1):
        return fig, ax[0][0]
    if (nrows == 1) or (ncols == 1):
        return fig, np.asarray(ax).flatten()
    return fig, np.asarray(ax)


def _get_colors(n, alpha, palette=None):
    if palette is None:
        palette = _sns.color_palette()
    if n == -1:
        colors = palette
    else:
        colors = palette[:n]
    return [_convert_color(color, alpha) for color in colors]


def scatterplot(
    *,
    x=None,
    y=None,
    hue=None,
    style=None,
    size=None,
    data=None,
    palette=None,
    hue_order=None,
    hue_norm=None,
    sizes=None,
    size_order=None,
    size_norm=None,
    markers=True,
    style_order=None,
    x_bins=None,
    y_bins=None,
    units=None,
    estimator=None,
    ci=95,
    n_boot=1000,
    alpha=None,
    x_jitter=None,
    y_jitter=None,
    legend="auto",
    ax=None,
    **kwargs,
):
    _validate_pandas(x, y, data, hue, size, style)

    variables = _sns.relational._ScatterPlotter.get_semantics(locals())
    p = _sns.relational._ScatterPlotter(
        data=data,
        variables=variables,
        x_bins=x_bins,
        y_bins=y_bins,
        estimator=estimator,
        ci=ci,
        n_boot=n_boot,
        alpha=alpha,
        x_jitter=x_jitter,
        y_jitter=y_jitter,
        legend=legend,
    )
    if palette is not None:
        p.variables["palette"] = palette
    if sizes is not None:
        p.variables["sizes"] = sizes
    if hue_order is not None:
        pass
        # raise NotImplementedError("hue_order isn't available yet")

    if ax is None:
        _, ax = subplots()

    if not p.has_xy_data:
        return ax.figure

    p.plot(ax, kwargs)
    return ax.figure


def heatmap(
    data,
    *,
    vmin=None,
    vmax=None,
    cmap=None,
    center=None,
    robust=False,
    annot=None,
    fmt=".2g",
    annot_kws=None,
    linewidths=0,
    linecolor="white",
    cbar=True,
    cbar_kws=None,
    cbar_ax=None,
    square=False,
    xticklabels="auto",
    yticklabels="auto",
    mask=None,
    ax=None,
    **kwargs,
):
    _validate_pandas(data)
    # Initialize the plotter object
    plotter = _sns.matrix._HeatMapper(
        data,
        vmin,
        vmax,
        cmap,
        center,
        robust,
        annot,
        fmt,
        annot_kws,
        cbar,
        cbar_kws,
        xticklabels,
        yticklabels,
        mask,
    )

    # Add the pcolormesh kwargs here
    kwargs["linewidths"] = linewidths
    kwargs["edgecolor"] = linecolor

    # Draw the plot and return the Axes
    if ax is not None:
        raise NotImplementedError("Passing ax is not supported")
    _, ax = subplots()
    if square:
        ax.figure["layout"]["yaxis"]["scaleanchor"] = "x"
    plotter.plot(ax, cbar_ax, kwargs)
    return ax.figure


def pairplot(
    data,
    *,
    hue=None,
    hue_order=None,
    palette=None,
    vars=None,
    x_vars=None,
    y_vars=None,
    kind="scatter",
    diag_kind="auto",
    markers=None,
    width=800,
    height=800,
    aspect=1,
    corner=False,
    dropna=False,
    plot_kws=None,
    diag_kws=None,
    grid_kws=None,
):
    _validate_pandas(data, hue)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "'data' must be pandas DataFrame object, not: {typefound}".format(
                typefound=type(data)
            )
        )

    plot_kws = {} if plot_kws is None else plot_kws.copy()
    diag_kws = {} if diag_kws is None else diag_kws.copy()
    grid_kws = {} if grid_kws is None else grid_kws.copy()

    # Resolve "auto" diag kind
    if diag_kind == "auto":
        if hue is None:
            diag_kind = "kde" if kind == "kde" else "hist"
        else:
            diag_kind = "hist" if kind == "hist" else "kde"

    numeric_cols = data.select_dtypes("number").columns
    if hue in numeric_cols:
        numeric_cols.remove(hue)
    if vars is not None:
        x_vars = list(vars)
        y_vars = list(vars)
    if x_vars is None:
        x_vars = numeric_cols
    if y_vars is None:
        y_vars = numeric_cols

    if np.isscalar(x_vars):
        x_vars = [x_vars]
    if np.isscalar(y_vars):
        y_vars = [y_vars]

    x_vars = list(x_vars)
    y_vars = list(y_vars)

    if not x_vars:
        raise ValueError("No variables found for grid columns.")
    if not y_vars:
        raise ValueError("No variables found for grid rows.")
    if not x_vars == y_vars:
        raise NotImplementedError("x_vars and y_vars need to be the same")

    fig = px.scatter_matrix(
        data,
        dimensions=x_vars,
        color=hue,
        color_discrete_sequence=_get_colors(-1, 1, palette),
        width=width or 800,
        height=height or 800,
    )
    fig.update_traces(diagonal_visible=False)
    return fig


def barplot(
    *,
    x=None,
    y=None,
    hue=None,
    data=None,
    order=None,
    hue_order=None,
    estimator=np.mean,
    ci=95,
    n_boot=1000,
    units=None,
    seed=None,
    orient=None,
    color=None,
    palette=None,
    saturation=0.75,
    errcolor=".26",
    errwidth=None,
    capsize=None,
    dodge=True,
    ax=None,
    **kwargs,
):
    _validate_pandas(x, y, data, hue)

    plotter = _sns.categorical._BarPlotter(
        x,
        y,
        hue,
        data,
        order,
        hue_order,
        estimator,
        ci,
        n_boot,
        units,
        seed,
        orient,
        color,
        palette,
        saturation,
        errcolor,
        errwidth,
        capsize,
        dodge,
    )
    plotter.data = data
    plotter.hue = hue
    plotter.x = x
    plotter.y = y
    plotter.estimator = estimator

    if ax is not None:
        raise NotImplementedError("passing ax is not supported")
    _, ax = subplots()

    plotter.plot(ax, kwargs)
    return ax.figure


def regplot(
    *,
    x=None,
    y=None,
    data=None,
    x_estimator=None,
    x_bins=None,
    x_ci="ci",
    scatter=True,
    fit_reg=True,
    ci=95,
    n_boot=1000,
    units=None,
    seed=None,
    order=1,
    logistic=False,
    lowess=False,
    robust=False,
    logx=False,
    x_partial=None,
    y_partial=None,
    truncate=True,
    dropna=True,
    x_jitter=None,
    y_jitter=None,
    label=None,
    color=None,
    marker="o",
    scatter_kws=None,
    line_kws=None,
    ax=None,
):
    _validate_pandas(x, y, data)
    plotter = _sns.regression._RegressionPlotter(
        x,
        y,
        data,
        x_estimator,
        x_bins,
        x_ci,
        scatter,
        fit_reg,
        ci,
        n_boot,
        units,
        seed,
        order,
        logistic,
        lowess,
        robust,
        logx,
        x_partial,
        y_partial,
        truncate,
        dropna,
        x_jitter,
        y_jitter,
        color,
        label,
    )

    if ax is None:
        _, ax = subplots()

    scatter_kws = {} if scatter_kws is None else copy.copy(scatter_kws)
    scatter_kws["marker"] = marker
    line_kws = {} if line_kws is None else copy.copy(line_kws)
    plotter.plot(ax, scatter_kws, line_kws)
    return ax.figure


def displot(*args, **kwargs):
    raise NotImplementedError("displot not available, use histplot instead")


def histplot(
    data=None,
    *,
    # Vector variables
    x=None,
    y=None,
    hue=None,
    weights=None,
    # Histogram computation parameters
    stat="count",
    bins="auto",
    binwidth=None,
    binrange=None,
    discrete=None,
    cumulative=False,
    common_bins=True,
    common_norm=True,
    # Histogram appearance parameters
    multiple="layer",
    element="bars",
    fill=True,
    shrink=1,
    # Histogram smoothing with a kernel density estimate
    kde=False,
    kde_kws=None,
    line_kws=None,
    # Bivariate histogram parameters
    thresh=0,
    pthresh=None,
    pmax=None,
    cbar=False,
    cbar_ax=None,
    cbar_kws=None,
    # Hue mapping parameters
    palette=None,
    hue_order=None,
    hue_norm=None,
    color=None,
    # Axes information
    log_scale=None,
    legend=True,
    ax=None,
    # Other appearance keywords
    **kwargs,
):
    _validate_pandas(x, y, data, hue)

    p = _sns.distributions._DistributionPlotter(
        data=data,
        variables=_sns.distributions._DistributionPlotter.get_semantics(locals()),
    )

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    if ax is not None:
        raise NotImplementedError("passing `ax` to `histogram` is not supported")

    if stat != "count":
        raise NotImplementedError("only count stat is currently supported")

    if not p.has_xy_data:
        return ax

    # Default to discrete bins for categorical variables
    if discrete is None:
        discrete = p._default_discrete()

    if p.univariate:
        plotting_kwargs = dict(
            data_frame=p.plot_data.rename(columns=p.variables),
            x=p.variables["x"],
            barmode="overlay",
        )
        if bins != "auto":
            plotting_kwargs["nbins"] = bins
        if hue is not None:
            plotting_kwargs["color"] = hue
        plotting_kwargs["color_discrete_sequence"] = _get_colors(-1, 1)
        if kde:
            plotting_kwargs["marginal"] = "violin"
        if plotting_kwargs.get("x") is None:
            raise ValueError("Please specify a value of `x`")
        fig = px.histogram(**plotting_kwargs)
        return fig

    else:
        raise NotImplementedError("bivariate histogram not yet supported")


class FacetGrid(_sns.axisgrid.FacetGrid):
    def update_hover(self, plot_variables, variables):
        for data in self.figure.data:
            if data.hovertemplate is None:
                continue
            for key, val in variables.items():
                if val is not None:
                    data.hovertemplate = data.hovertemplate.replace(
                        f"{plot_variables[key]}=", f"{val}="
                    )

    def tight_layout(self, *args, **kwargs):
        pass

    def despine(self, *args, **kwargs):
        pass

    @property
    def _not_bottom_axes(self):
        """Return a flat array of axes that aren't on the bottom row."""
        if self._col_wrap is None:
            return iter(self.axes[:-1, :].flatten())
        else:
            axes = []
            n_empty = self._nrow * self._ncol - self._n_facets
            for i, ax in enumerate(self.axes):
                append = i < (self._ncol * (self._nrow - 1)) and i < (
                    self._ncol * (self._nrow - 1) - n_empty
                )
                if append:
                    axes.append(ax)
            return np.array(axes, object).flat

    def _update_legend_data(self, ax):
        pass

    def add_legend(self, *args, **kwargs):
        pass

    def __init__(
        self,
        data,
        *,
        row=None,
        col=None,
        hue=None,
        col_wrap=None,
        sharex=True,
        sharey=True,
        height=3,
        aspect=1,
        palette=None,
        row_order=None,
        col_order=None,
        hue_order=None,
        hue_kws=None,
        dropna=False,
        legend_out=True,
        despine=True,
        margin_titles=False,
        xlim=None,
        ylim=None,
        subplot_kws=None,
        gridspec_kws=None,
    ):

        super(_sns.axisgrid.FacetGrid, self).__init__()

        # Determine the hue facet layer information
        hue_var = hue
        if hue is None:
            hue_names = None
        else:
            hue_names = _sns.axisgrid.categorical_order(data[hue], hue_order)

        colors = self._get_palette(data, hue, hue_order, palette)

        # Set up the lists of names for the row and column facet variables
        if row is None:
            row_names = []
        else:
            row_names = _sns.axisgrid.categorical_order(data[row], row_order)

        if col is None:
            col_names = []
        else:
            col_names = _sns.axisgrid.categorical_order(data[col], col_order)

        # Additional dict of kwarg -> list of values for mapping the hue var
        hue_kws = hue_kws if hue_kws is not None else {}

        # Make a boolean mask that is True anywhere there is an NA
        # value in one of the faceting variables, but only if dropna is True
        none_na = np.zeros(len(data), bool)
        if dropna:
            row_na = none_na if row is None else data[row].isnull()
            col_na = none_na if col is None else data[col].isnull()
            hue_na = none_na if hue is None else data[hue].isnull()
            not_na = ~(row_na | col_na | hue_na)
        else:
            not_na = ~none_na

        # Compute the grid shape
        ncol = 1 if col is None else len(col_names)
        nrow = 1 if row is None else len(row_names)
        self._n_facets = ncol * nrow

        self._col_wrap = col_wrap
        if col_wrap is not None:
            if row is not None:
                err = "Cannot use `row` and `col_wrap` together."
                raise ValueError(err)
            ncol = col_wrap
            nrow = int(np.ceil(len(col_names) / col_wrap))
        self._ncol = ncol
        self._nrow = nrow

        # Calculate the base figure size
        # This can get stretched later by a legend
        # TODO this doesn't account for axis labels
        figsize = (ncol * height * aspect, nrow * height)

        # Validate some inputs
        if col_wrap is not None:
            margin_titles = False

        # Build the subplot keyword dictionary
        subplot_kws = {} if subplot_kws is None else subplot_kws.copy()
        gridspec_kws = {} if gridspec_kws is None else gridspec_kws.copy()
        if xlim is not None:
            subplot_kws["xlim"] = xlim
        if ylim is not None:
            subplot_kws["ylim"] = ylim

        # --- Initialize the subplot grid

        # Disable autolayout so legend_out works properly

        if col_wrap is None:

            kwargs = dict(
                squeeze=False,
                sharex=sharex,
                sharey=sharey,
                subplot_kw=subplot_kws,
                gridspec_kw=gridspec_kws,
            )

            fig, axes = subplots(nrow, ncol, wrap=False, **kwargs)

            if col is None and row is None:
                axes_dict = {}
            elif col is None:
                axes_dict = dict(zip(row_names, axes.flat))
            elif row is None:
                axes_dict = dict(zip(col_names, axes.flat))
            else:
                facet_product = _sns.axisgrid.product(row_names, col_names)
                axes_dict = dict(zip(facet_product, axes.flat))

        else:
            raise NotImplementedError()
            # # If wrapping the col variable we need to make the grid ourselves
            # if gridspec_kws:
            #     warnings.warn("`gridspec_kws` ignored when using `col_wrap`")

            # n_axes = len(col_names)
            # axes = np.empty(n_axes, object)
            # axes[0] = fig.add_subplot(nrow, ncol, 1, **subplot_kws)
            # if sharex:
            #     subplot_kws["sharex"] = axes[0]
            # if sharey:
            #     subplot_kws["sharey"] = axes[0]
            # for i in range(1, n_axes):
            #     axes[i] = fig.add_subplot(nrow, ncol, i + 1, **subplot_kws)

            # axes_dict = dict(zip(col_names, axes))

        # --- Set up the class attributes

        # Attributes that are part of the public API but accessed through
        # a  property so that Sphinx adds them to the auto class doc
        self._figure = fig
        self._axes = axes
        self._axes_dict = axes_dict
        self._legend = None

        # Public attributes that aren't explicitly documented
        # (It's not obvious that having them be public was a good idea)
        self.data = data
        self.row_names = row_names
        self.col_names = col_names
        self.hue_names = hue_names
        self.hue_kws = hue_kws

        # Next the private variables
        self._nrow = nrow
        self._row_var = row
        self._ncol = ncol
        self._col_var = col

        self._margin_titles = margin_titles
        self._margin_titles_texts = []
        self._col_wrap = col_wrap
        self._hue_var = hue_var
        self._colors = colors
        self._legend_out = legend_out
        self._legend_data = {}
        self._x_var = None
        self._y_var = None
        self._sharex = sharex
        self._sharey = sharey
        self._dropna = dropna
        self._not_na = not_na

        # --- Make the axes look good

        self.set_titles()
        self.tight_layout()

        if despine:
            self.despine()

        # if sharex in [True, 'col']:
        #     for ax in self._not_bottom_axes:
        #         for label in ax.get_xticklabels():
        #             label.set_visible(False)
        #         ax.xaxis.offsetText.set_visible(False)
        #         ax.xaxis.label.set_visible(False)

        # if sharey in [True, 'row']:
        #     for ax in self._not_left_axes:
        #         for label in ax.get_yticklabels():
        #             label.set_visible(False)
        #         ax.yaxis.offsetText.set_visible(False)
        #         ax.yaxis.label.set_visible(False)


def relplot(
    *,
    x=None,
    y=None,
    hue=None,
    size=None,
    style=None,
    data=None,
    row=None,
    col=None,
    col_wrap=None,
    row_order=None,
    col_order=None,
    palette=None,
    hue_order=None,
    hue_norm=None,
    sizes=None,
    size_order=None,
    size_norm=None,
    markers=None,
    dashes=None,
    style_order=None,
    legend=True,
    kind="scatter",
    height=5,
    aspect=1,
    facet_kws=None,
    units=None,
    **kwargs,
):
    _validate_pandas(x, y, data, hue, style)

    if kind == "scatter":

        plotter = _sns.relational._ScatterPlotter
        func = scatterplot
        markers = True if markers is None else markers

    elif kind == "line":

        plotter = _sns.relational._LinePlotter
        func = lineplot
        dashes = True if dashes is None else dashes

    else:
        err = "Plot kind {} not recognized".format(kind)
        raise ValueError(err)

    # Check for attempt to plot onto specific axes and warn
    if "ax" in kwargs:
        msg = (
            "relplot is a figure-level function and does not accept "
            "the `ax` parameter. You may wish to try {}".format(kind + "plot")
        )
        import warnings

        warnings.warn(msg, UserWarning)
        kwargs.pop("ax")

    # Use the full dataset to map the semantics
    p = plotter(
        data=data,
        variables=plotter.get_semantics(locals()),
        legend=legend,
    )
    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)
    p.map_size(sizes=sizes, order=size_order, norm=size_norm)
    p.map_style(markers=markers, dashes=dashes, order=style_order)

    # Extract the semantic mappings
    if "hue" in p.variables:
        palette = p._hue_map.lookup_table
        hue_order = p._hue_map.levels
        hue_norm = p._hue_map.norm
    else:
        palette = hue_order = hue_norm = None

    if "size" in p.variables:
        sizes = p._size_map.lookup_table
        size_order = p._size_map.levels
        size_norm = p._size_map.norm

    if "style" in p.variables:
        style_order = p._style_map.levels
        if markers:
            markers = {k: p._style_map(k, "marker") for k in style_order}
        else:
            markers = None
        if dashes:
            dashes = {k: p._style_map(k, "dashes") for k in style_order}
        else:
            dashes = None
    else:
        markers = dashes = style_order = None

    # Now extract the data that would be used to draw a single plot
    variables = p.variables
    plot_data = p.plot_data
    plot_semantics = p.semantics

    # Define the common plotting parameters
    plot_kws = dict(
        palette=palette,
        hue_order=hue_order,
        hue_norm=hue_norm,
        sizes=sizes,
        size_order=size_order,
        size_norm=size_norm,
        markers=markers,
        dashes=dashes,
        style_order=style_order,
        legend=True,
    )
    plot_kws.update(kwargs)
    if kind == "scatter":
        plot_kws.pop("dashes")

    # Add the grid semantics onto the plotter
    grid_semantics = "row", "col"
    p.semantics = plot_semantics + grid_semantics
    p.assign_variables(
        data=data,
        variables=dict(
            x=x,
            y=y,
            hue=hue,
            size=size,
            style=style,
            units=units,
            row=row,
            col=col,
        ),
    )

    # Define the named variables for plotting on each facet
    # Rename the variables with a leading underscore to avoid
    # collisions with faceting variable names
    plot_variables = {v: f"_{v}" for v in variables}
    plot_kws.update(plot_variables)

    # Pass the row/col variables to FacetGrid with their original
    # names so that the axes titles render correctly
    grid_kws = {v: p.variables.get(v, None) for v in grid_semantics}

    # Rename the columns of the plot_data structure appropriately
    new_cols = plot_variables.copy()
    new_cols.update(grid_kws)
    full_data = p.plot_data.rename(columns=new_cols)

    # Set up the FacetGrid object
    facet_kws = {} if facet_kws is None else facet_kws.copy()
    g = FacetGrid(
        data=full_data.dropna(axis=1, how="all"),
        **grid_kws,
        col_wrap=col_wrap,
        row_order=row_order,
        col_order=col_order,
        height=height,
        aspect=aspect,
        dropna=False,
        **facet_kws,
    )

    # Draw the plot
    g.map_dataframe(func, **plot_kws)

    # Label the axes
    g.set_axis_labels(variables.get("x", None), variables.get("y", None))

    g.update_hover(plot_variables, variables)

    # Show the legend
    if legend:
        # Replace the original plot data so the legend uses
        # numeric data with the correct type
        p.plot_data = plot_data
        p.add_legend_data(g.axes.flat[0])

        _dedupe_legend(g.figure)
        if p.legend_data:
            g._figure.layout.legend.title = p.legend_title

    # Rename the columns of the FacetGrid's `data` attribute
    # to match the original column names
    orig_cols = {f"_{k}": f"_{k}_" if v is None else v for k, v in variables.items()}
    grid_data = g.data.rename(columns=orig_cols)
    if data is not None and (x is not None or y is not None):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        g.data = pd.merge(
            data,
            grid_data[grid_data.columns.difference(data.columns)],
            left_index=True,
            right_index=True,
        )
    else:
        g.data = grid_data

    return g._figure


def lmplot(
    *,
    x=None,
    y=None,
    data=None,
    hue=None,
    col=None,
    row=None,  # TODO move before data once * is enforced
    palette=None,
    col_wrap=None,
    height=5,
    aspect=1,
    markers="o",
    hue_order=None,
    col_order=None,
    row_order=None,
    legend=True,
    x_estimator=None,
    x_bins=None,
    x_ci="ci",
    scatter=True,
    fit_reg=True,
    ci=95,
    n_boot=1000,
    units=None,
    seed=None,
    order=1,
    logistic=False,
    lowess=False,
    robust=False,
    logx=False,
    x_partial=None,
    y_partial=None,
    truncate=True,
    x_jitter=None,
    y_jitter=None,
    scatter_kws=None,
    line_kws=None,
    facet_kws=None,
):
    _validate_pandas(x, y, data, hue)

    if facet_kws is None:
        facet_kws = {}

    if data is None:
        raise TypeError("Missing required keyword argument `data`.")

    # Reduce the dataframe to only needed columns
    need_cols = [x, y, hue, col, row, units, x_partial, y_partial]
    cols = np.unique([a for a in need_cols if a is not None]).tolist()
    data = data[cols]

    # Initialize the grid
    facets = FacetGrid(
        data,
        row=row,
        col=col,
        hue=hue,
        palette=palette,
        row_order=row_order,
        col_order=col_order,
        hue_order=hue_order,
        height=height,
        aspect=aspect,
        col_wrap=col_wrap,
        **facet_kws,
    )

    # Add the markers here as FacetGrid has figured out how many levels of the
    # hue variable are needed and we don't want to duplicate that process
    if facets.hue_names is None:
        n_markers = 1
    else:
        n_markers = len(facets.hue_names)
    if not isinstance(markers, list):
        markers = [markers] * n_markers
    if len(markers) != n_markers:
        raise ValueError(
            (
                "markers must be a singleton or a list of markers "
                "for each level of the hue variable"
            )
        )
    facets.hue_kws = {"marker": markers}

    def update_datalim(data, x, y, ax, **kws):
        pass
        # xys = data[[x, y]].to_numpy().astype(float)
        # ax.update_datalim(xys, updatey=False)
        # ax.autoscale_view(scaley=False)

    facets.map_dataframe(update_datalim, x=x, y=y)

    # Draw the regression plot on each facet
    regplot_kws = dict(
        x_estimator=x_estimator,
        x_bins=x_bins,
        x_ci=x_ci,
        scatter=scatter,
        fit_reg=fit_reg,
        ci=ci,
        n_boot=n_boot,
        units=units,
        seed=seed,
        order=order,
        logistic=logistic,
        lowess=lowess,
        robust=robust,
        logx=logx,
        x_partial=x_partial,
        y_partial=y_partial,
        truncate=truncate,
        x_jitter=x_jitter,
        y_jitter=y_jitter,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
    )
    facets.map_dataframe(regplot, x=x, y=y, **regplot_kws)
    facets.set_axis_labels(x, y)

    # Add a legend
    if legend and (hue is not None) and (hue not in [col, row]):
        facets._figure.layout.legend.title = hue
    facets.update_hover({"x": "x", "y": "y"}, {"x": x, "y": y})
    return facets._figure


def lineplot(
    *,
    x=None,
    y=None,
    hue=None,
    size=None,
    style=None,
    data=None,
    palette=None,
    hue_order=None,
    hue_norm=None,
    sizes=None,
    size_order=None,
    size_norm=None,
    dashes=True,
    markers=None,
    style_order=None,
    units=None,
    estimator="mean",
    ci="deprecated",
    n_boot=1000,
    seed=None,
    sort=True,
    err_style="band",
    err_kws=None,
    legend="auto",
    errorbar=("ci", 95),
    ax=None,
    **kwargs,
):
    _validate_pandas(x, y, data)
    if ax is None:
        _, ax = subplots()

    if "size" in kwargs:
        raise NotImplementedError("size isn't supported yet. Use relplot instead?")
    if kwargs.get("err_style") == "bars":
        raise NotImplementedError(
            'err_style = "bars" is not supported yet, use "band" instead.'
        )
    if kwargs.get("dashes") is False:
        raise NotImplementedError("passing dashes=False is not supported yet")
    if kwargs.get("markers") is False:
        raise NotImplementedError("passing markers=False is not supported yet")

    fig = _sns.lineplot(
        ax=ax,
        x=x,
        y=y,
        hue=hue,
        size=size,
        style=style,
        data=data,
        palette=palette,
        hue_order=hue_order,
        hue_norm=hue_norm,
        sizes=sizes,
        size_order=size_order,
        size_norm=size_norm,
        dashes=dashes,
        markers=markers,
        style_order=style_order,
        units=units,
        estimator=estimator,
        ci=ci,
        n_boot=n_boot,
        seed=seed,
        sort=sort,
        err_style=err_style,
        err_kws=err_kws,
        legend=legend,
        errorbar=errorbar,
        **kwargs,
    ).figure
    color_legend_map = {
        ",".join(i.marker.color.split(",")[:3]): i.legendgroup
        for i in fig.data
        if i.legendgroup and i.marker.color is not None
    }
    dash_legend_map = {i.line.dash: i.legendgroup for i in fig.data if i.legendgroup}
    if not color_legend_map and not dash_legend_map:
        fig.update_layout(showlegend=False)

    x_label = ax.get_xlabel()
    y_label = ax.get_ylabel()

    for _data in fig.data:
        if _data.marker.color is not None:
            _color = _data.marker.color
        elif _data.fillcolor is not None:
            _color = _data.fillcolor
        else:
            _color = None
        if _data.line.dash is not None:
            _dash = _data.line.dash
        else:
            _dash = None

        legendgroup = set()

        # I think the solution is to always plot the rgba color!

        if _color is not None:
            name = color_legend_map.get(",".join(_color.split(",")[:3]))
            if name is not None:
                legendgroup.add(name)

        dashname = dash_legend_map.get(_dash)
        if style is not None and dashname is not None:
            legendgroup.add(dashname)

        if legendgroup and not _data.hoverinfo == "skip":
            _data.legendgroup = ", ".join(legendgroup)
            _data.name = ", ".join(legendgroup)
        if not _data.hoverinfo == "skip":
            _data.hovertemplate = (
                f'{x_label or "x"}=%{{x}}<br>{y_label or "y"}=%{{y}}<extra></extra>'
            )

    return fig


def kdeplot(
    x=None,
    *,
    y=None,
    cut=3,
    clip=None,
    legend=True,
    cumulative=False,
    cbar=False,
    cbar_ax=None,
    cbar_kws=None,
    ax=None,
    # New params
    weights=None,  # TODO note that weights is grouped with semantics
    hue=None,
    palette=None,
    hue_order=None,
    hue_norm=None,
    multiple="layer",
    common_norm=True,
    common_grid=False,
    levels=10,
    thresh=0.05,
    bw_method="scott",
    bw_adjust=1,
    log_scale=None,
    color=None,
    fill=None,
    # Renamed params
    data=None,
    data2=None,
    # New in v0.12
    warn_singular=True,
    **kwargs,
):
    _validate_pandas(x, y, data, data2)

    # need to set args here
    if kwargs.get("ax") is None:
        _, ax = subplots()
    else:
        ax = kwargs.pop("ax")
    fig = _sns.kdeplot(
        ax=ax,
        x=x,
        y=y,
        cut=cut,
        clip=clip,
        legend=legend,
        cumulative=cumulative,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        weights=weights,
        hue=hue,
        palette=palette,
        hue_order=hue_order,
        hue_norm=hue_norm,
        multiple=multiple,
        common_norm=common_norm,
        common_grid=common_grid,
        levels=levels,
        thresh=thresh,
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        log_scale=log_scale,
        color=color,
        fill=fill,
        data=data,
        data2=data2,
        warn_singular=warn_singular,
        **kwargs,
    ).figure
    legend_map = {
        ",".join(i.marker.color.split(",")[:3]): i.legendgroup
        for i in fig.data
        if i.legendgroup
    }
    if not legend_map:
        fig.update_layout(showlegend=False)

    x_label = ax.get_xlabel()
    y_label = ax.get_ylabel()

    for _data in fig.data:
        if _data.marker.color is not None:
            _color = _data.marker.color
        elif _data.fillcolor is not None:
            _color = _data.fillcolor
        else:
            continue
        _data.legendgroup = legend_map.get(",".join(_color.split(",")[:3]))
        _data.name = legend_map.get(",".join(_color.split(",")[:3]))
        if not _data.hoverinfo == "skip":
            _data.hovertemplate = f"{x_label}=%{{x}}<br>{y_label}=%{{y}}<extra></extra>"
    return fig
