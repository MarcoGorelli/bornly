import functools
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


def _cartesian(x, y):
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

def _deconvert_rgba(rgba):
    return tuple([float(i)/255 for i in rgba.strip('rgba()').split(',')[:3]])

def _convert_color(color, alpha):
    if alpha is not None:
        return f"rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {alpha})"
    return f"rgb({color[0]*255}, {color[1]*255}, {color[2]*255})"

class Foo:
    update_units = lambda *_: None
    convert_units = lambda x, y: y
    get_scale = lambda *_: None
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
        return float(self.scatter.marker.color.strip('rgba()').split(',')[-1])
    def get_solid_capstyle(self):
        pass  # huh
    def remove(self):
        pass
    def set_dashes(self, dashes):
        if dashes:
            self.scatter.line.update(dash=', '.join([str(i) for i in dashes]))
    def set_linewidth(self, width):
        pass
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
            self.ax(go.Figure(go.Scatter(x=self.scatter.x[mask], y=self.scatter.y[mask], mode='markers', marker=dict(color=_convert_color(facecolor, 1)))))
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
    
def _parse_color(color):
    if isinstance(color, str) and color.startswith('rgba('):
        pass
    elif isinstance(color, tuple) and len(color) == 4:
        color = _convert_color(color[:3], color[3])
    elif isinstance(color, tuple) and len(color) == 3:
        color = _convert_color(color, 1)
    elif isinstance(color, str):
        try:
            color = _convert_color(mpl.colors.to_rgb(color), 1)
        except ValueError:
            color = _convert_color((mpl.cm.hot(float(color))), 1)
    return color
class Ax:
    def __init__(self, func, *, nrows, ncols):
        self._func = func
        self._row = func.keywords["row"]
        self._col = func.keywords["col"]
        self._nrows = nrows
        self._ncols = ncols

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

    def legend(self, title):
        self.figure.layout.legend.title = title
        return Legend(self.figure.layout.legend)
    
    def plot(self, x, y, **kwargs):
        if kwargs.get('color', None) is not None:
            color = kwargs.get('color', None)
            color = _parse_color(color)
        else:
            color = _get_colors(1, 1)[0]

        fig = go.Scatter(x=x, y=y, mode='lines', legendgroup=kwargs.get('label', None), name=kwargs.get('label', None), marker=dict(color=color))
        self.figure.add_trace(fig, row=self._row+1, col=self._col+1)
        return Line(self.figure.data[-1], self),
    
    def add_legend(self, legend_data, **kwargs):
        for key, val in legend_data.items():
            color = val.get_color()
            for _data in self.figure.data:
                if Line(_data).get_color() == color[:3]:
                    _data.legendgroup = key
        self.figure.layout.legend.title = kwargs.get('title')
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
        self.figure.update_yaxes(range=ylim, row=self._row + 1, col=self._col + 1)

    def set_xlim(self, xlim, *args, **kwargs):
        if isinstance(xlim, tuple):
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
        rgb = kwargs.get('color', None)
        alpha = kwargs.get('alpha', None)
        if rgb is not None:
            rgba = _convert_color(rgb, alpha)
        self.figure.add_trace(
            go.Scatter(
                x=x,
                y=y2,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                # legendgroup=legendgroup,
                hoverinfo='skip',
            ),
            row=self._row+1,
            col=self._col+1,
        )

        self.figure.add_trace(
            go.Scatter(
                x=x,
                y=y1,
                line=dict(color="rgba(0,0,0,0)"),
                fill="tonexty",
                fillcolor=rgba,
                showlegend=False,
                # name=label,
                # legendgroup=legendgroup,
                hoverinfo='skip',
            ),
            row=self._row+1,
            col=self._col+1,
        )


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




# not easy to override `ax.scatter`, so got to do it
# this way, unfortunately
class ScatterPlotter(_sns.relational._ScatterPlotter):
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
            "data_frame": data,
            "x": self.variables["x"],
            "y": self.variables["y"],
        }
        if "palette" in self.variables:
            if isinstance(self.variables["palette"], dict):
                plotting_kwargs["color_discrete_map"] = { key: _convert_color(_sns.color_palette([val])[0], 1) for key, val in self.variables["palette"].items() }
            else:
                plotting_kwargs["color_discrete_sequence"] = _get_colors(
                    -1, 1, _sns.color_palette(self.variables["palette"])
                )
        elif 'color' in kws:
            plotting_kwargs["color_discrete_sequence"] = [_parse_color(kws['color'])]
        else:
            plotting_kwargs["color_discrete_sequence"] = _get_colors(-1, 1)
        if "hue" in self.variables:
            plotting_kwargs["color"] = self.variables["hue"]
        if "size" in self.variables:
            plotting_kwargs["size"] = self.variables["size"]
        if "sizes" in self.variables:
            plotting_kwargs["size_max"] = self.variables["sizes"][1]

        fig = px.scatter(**plotting_kwargs)
        ax.set_xlabel(self.variables["x"])
        ax.set_ylabel(self.variables["y"])

        if not self.legend:
            for i in fig.data:
                i.showlegend = False

        ax(fig)


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

    variables = ScatterPlotter.get_semantics(locals())
    p = ScatterPlotter(
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





# also not one to be overridden easily...
class HeatMapper(_sns.matrix._HeatMapper):
    def plot(self, ax, cax, kws):
        data_ = self.data.copy()
        data_[self.plot_data.mask] = np.nan
        plotting_kwargs = {
            "img": data_,
            "zmin": self.vmin,
            "zmax": self.vmax,
        }
        palette = self.cmap(np.linspace(0, 1, self.cmap.N))

        plotting_kwargs["color_continuous_scale"] = _get_colors(-1, 1, palette)
        fig = px.imshow(**plotting_kwargs)
        fig.update_layout(
            xaxis_showgrid=False, yaxis_showgrid=False, template="plotly_white"
        )
        ax(fig)
        ax.figure.update_layout(fig.layout)


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
    """Plot rectangular data as a color-encoded matrix.

    This is an Axes-level function and will draw the heatmap into the
    currently-active Axes if none is provided to the ``ax`` argument.  Part of
    this Axes space will be taken and used to plot a colormap, unless ``cbar``
    is False or a separate Axes is provided to ``cbar_ax``.

    Parameters
    ----------
    data : rectangular dataset
        2D dataset that can be coerced into an ndarray. If a Pandas DataFrame
        is provided, the index/column information will be used to label the
        columns and rows.
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments.
    cmap : matplotlib colormap name or object, or list of colors, optional
        The mapping from data values to color space. If not provided, the
        default will depend on whether ``center`` is set.
    center : float, optional
        The value at which to center the colormap when plotting divergant data.
        Using this parameter will change the default ``cmap`` if none is
        specified.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with robust quantiles instead of the extreme values.
    annot : bool or rectangular dataset, optional
        If True, write the data value in each cell. If an array-like with the
        same shape as ``data``, then use this to annotate the heatmap instead
        of the data. Note that DataFrames will match on position, not index.
    fmt : str, optional
        String formatting code to use when adding annotations.
    annot_kws : dict of key, value mappings, optional
        Keyword arguments for :meth:`matplotlib.axes.Axes.text` when ``annot``
        is True.
    linewidths : float, optional
        Width of the lines that will divide each cell.
    linecolor : color, optional
        Color of the lines that will divide each cell.
    cbar : bool, optional
        Whether to draw a colorbar.
    cbar_kws : dict of key, value mappings, optional
        Keyword arguments for :meth:`matplotlib.figure.Figure.colorbar`.
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar, otherwise take space from the
        main Axes.
    square : bool, optional
        If True, set the Axes aspect to "equal" so each cell will be
        square-shaped.
    xticklabels, yticklabels : "auto", bool, list-like, or int, optional
        If True, plot the column names of the dataframe. If False, don't plot
        the column names. If list-like, plot these alternate labels as the
        xticklabels. If an integer, use the column names but plot only every
        n label. If "auto", try to densely plot non-overlapping labels.
    mask : bool array or DataFrame, optional
        If passed, data will not be shown in cells where ``mask`` is True.
        Cells with missing values are automatically masked.
    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise use the currently-active
        Axes.
    kwargs : other keyword arguments
        All other keyword arguments are passed to
        :meth:`matplotlib.axes.Axes.pcolormesh`.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the heatmap.

    See Also
    --------
    clustermap : Plot a matrix using hierarchical clustering to arrange the
                 rows and columns.

    Examples
    --------

    Plot a heatmap for a numpy array:

    .. plot::
        :context: close-figs

        >>> import numpy as np; np.random.seed(0)
        >>> import seaborn as sns; sns.set_theme()
        >>> uniform_data = np.random.rand(10, 12)
        >>> ax = sns.heatmap(uniform_data)

    Change the limits of the colormap:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(uniform_data, vmin=0, vmax=1)

    Plot a heatmap for data centered on 0 with a diverging colormap:

    .. plot::
        :context: close-figs

        >>> normal_data = np.random.randn(10, 12)
        >>> ax = sns.heatmap(normal_data, center=0)

    Plot a dataframe with meaningful row and column labels:

    .. plot::
        :context: close-figs

        >>> flights = sns.load_dataset("flights")
        >>> flights = flights.pivot("month", "year", "passengers")
        >>> ax = sns.heatmap(flights)

    Annotate each cell with the numeric value using integer formatting:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, annot=True, fmt="d")

    Add lines between each cell:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, linewidths=.5)

    Use a different colormap:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, cmap="YlGnBu")

    Center the colormap at a specific value:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, center=flights.loc["Jan", 1955])

    Plot every other column label and don't plot row labels:

    .. plot::
        :context: close-figs

        >>> data = np.random.randn(50, 20)
        >>> ax = sns.heatmap(data, xticklabels=2, yticklabels=False)

    Don't draw a colorbar:

    .. plot::
        :context: close-figs

        >>> ax = sns.heatmap(flights, cbar=False)

    Use different axes for the colorbar:

    .. plot::
        :context: close-figs

        >>> grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
        >>> f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
        >>> ax = sns.heatmap(flights, ax=ax,
        ...                  cbar_ax=cbar_ax,
        ...                  cbar_kws={"orientation": "horizontal"})

    Use a mask to plot only part of a matrix

    .. plot::
        :context: close-figs

        >>> corr = np.corrcoef(np.random.randn(10, 200))
        >>> mask = np.zeros_like(corr)
        >>> mask[np.triu_indices_from(mask)] = True
        >>> with sns.axes_style("white"):
        ...     f, ax = plt.subplots(figsize=(7, 5))
        ...     ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True)
    """
    # Initialize the plotter object
    plotter = HeatMapper(
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


# just redirect to plotly equivalent
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
    """Plot pairwise relationships in a dataset.

    By default, this function will create a grid of Axes such that each numeric
    variable in ``data`` will by shared across the y-axes across a single row and
    the x-axes across a single column. The diagonal plots are treated
    differently: a univariate distribution plot is drawn to show the marginal
    distribution of the data in each column.

    It is also possible to show a subset of variables or plot different
    variables on the rows and columns.

    This is a high-level interface for :class:`PairGrid` that is intended to
    make it easy to draw a few common styles. You should use :class:`PairGrid`
    directly if you need more flexibility.

    Parameters
    ----------
    data : `pandas.DataFrame`
        Tidy (long-form) dataframe where each column is a variable and
        each row is an observation.
    hue : name of variable in ``data``
        Variable in ``data`` to map plot aspects to different colors.
    hue_order : list of strings
        Order for the levels of the hue variable in the palette
    palette : dict or seaborn color palette
        Set of colors for mapping the ``hue`` variable. If a dict, keys
        should be values  in the ``hue`` variable.
    vars : list of variable names
        Variables within ``data`` to use, otherwise use every column with
        a numeric datatype.
    {x, y}_vars : lists of variable names
        Variables within ``data`` to use separately for the rows and
        columns of the figure; i.e. to make a non-square plot.
    kind : {'scatter', 'kde', 'hist', 'reg'}
        Kind of plot to make.
    diag_kind : {'auto', 'hist', 'kde', None}
        Kind of plot for the diagonal subplots. If 'auto', choose based on
        whether or not ``hue`` is used.
    markers : single matplotlib marker code or list
        Either the marker to use for all scatterplot points or a list of markers
        with a length the same as the number of levels in the hue variable so that
        differently colored points will also have different scatterplot
        markers.
    height : scalar
        Height (in inches) of each facet.
    aspect : scalar
        Aspect * height gives the width (in inches) of each facet.
    corner : bool
        If True, don't add axes to the upper (off-diagonal) triangle of the
        grid, making this a "corner" plot.
    dropna : boolean
        Drop missing values from the data before plotting.
    {plot, diag, grid}_kws : dicts
        Dictionaries of keyword arguments. ``plot_kws`` are passed to the
        bivariate plotting function, ``diag_kws`` are passed to the univariate
        plotting function, and ``grid_kws`` are passed to the :class:`PairGrid`
        constructor.

    Returns
    -------
    grid : :class:`PairGrid`
        Returns the underlying :class:`PairGrid` instance for further tweaking.

    See Also
    --------
    PairGrid : Subplot grid for more flexible plotting of pairwise relationships.
    JointGrid : Grid for plotting joint and marginal distributions of two variables.

    Examples
    --------

    .. include:: ../docstrings/pairplot.rst

    """

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


class BarPlotter(_sns.categorical._BarPlotter):
    def plot(self, ax, bar_kws):
        """Make the plot."""
        self.draw_bars(ax, bar_kws)

    def draw_bars(self, ax, kws):
        """Draw the bars onto `ax`."""
        # Get the right matplotlib function depending on the orientation

        data = pd.DataFrame({"y": self.statistic.flatten()})
        plotting_kwargs = dict(
            data_frame=data,
            x="x",
            y="y",
            color_discrete_sequence=_get_colors(-1, 1, self.colors),
        )
        if self.plot_hues is None:
            data["x"] = self.group_names
            data["err"] = self.confint[:, 1] - data["y"]
            plotting_kwargs["color"] = "x"
        else:
            data[["hue", "x"]] = _cartesian(self.hue_names, self.group_names)
            data["err"] = self.confint[:, :, 1].flatten() - data["y"]
            plotting_kwargs["color"] = "hue"
            plotting_kwargs["barmode"] = "group"

        plotting_kwargs["category_orders"] = {"x": self.group_names}

        if not np.isnan(data["err"]).all():
            plotting_kwargs["error_y"] = "err"

        if self.orient == "h":
            plotting_kwargs["x"], plotting_kwargs["y"] = (
                plotting_kwargs["y"],
                plotting_kwargs["x"],
            )
            plotting_kwargs["orientation"] = "h"
            if "error_y" in plotting_kwargs:
                plotting_kwargs["error_x"] = plotting_kwargs.pop("error_y")

        fig = px.bar(**plotting_kwargs)
        ax(fig)
        if self.orient == "v":
            xlabel, ylabel = self.group_label, self.value_label
        else:
            xlabel, ylabel = self.value_label, self.group_label
        ax.figure.update_layout(fig.layout)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if self.hue_names is not None:
            ax.figure.layout.legend.title = self.hue_title
        else:
            ax.figure.update_layout(showlegend=False)


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

    plotter = BarPlotter(
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


class RegressionPlotter(_sns.regression._RegressionPlotter):
    def plot(self, ax, scatter_kws, line_kws):
        """Draw the full plot."""
        # Insert the plot label into the correct set of keyword arguments
        if self.scatter:
            scatter_kws["label"] = self.label
        else:
            line_kws["label"] = self.label

        # Use the current color cycle state as a default
        if self.color is None:
            color = _sns.color_palette()[0]
            # lines, = ax.plot([], [])
            # color = lines.get_color()
            # lines.remove()
        else:
            color = self.color

        # Ensure that color is hex to avoid matplotlib weirdness
        color = mpl.colors.rgb2hex(mpl.colors.colorConverter.to_rgb(color))

        # Let color in keyword arguments override overall plot color
        scatter_kws.setdefault("color", color)
        line_kws.setdefault("color", color)

        # Draw the constituent plots
        if self.scatter:
            self.scatterplot(ax, scatter_kws)

        if self.fit_reg:
            self.lineplot(ax, line_kws)

        # Label the axes
        if hasattr(self.x, "name"):
            ax.set_xlabel(self.x.name)
        if hasattr(self.y, "name"):
            ax.set_ylabel(self.y.name)

    def scatterplot(self, ax, kws):
        """Draw the data."""
        # Treat the line-based markers specially, explicitly setting larger
        # linewidth than is provided by the seaborn style defaults.
        # This would ideally be handled better in matplotlib (i.e., distinguish
        # between edgewidth for solid glyphs and linewidth for line glyphs
        # but this should do for now.
        line_markers = ["1", "2", "3", "4", "+", "x", "|", "_"]
        if self.x_estimator is None:
            if "marker" in kws and kws["marker"] in line_markers:
                lw = mpl.rcParams["lines.linewidth"]
            else:
                lw = mpl.rcParams["lines.markeredgewidth"]
            kws.setdefault("linewidths", lw)

            if not hasattr(kws["color"], "shape") or kws["color"].shape[1] < 4:
                kws.setdefault("alpha", 0.8)

            x, y = self.scatter_data
            data = pd.DataFrame({"x": x, "y": y})
            fig = px.scatter(
                data,
                x="x",
                y="y",
                color_discrete_sequence=[
                    _convert_color(mpl.colors.to_rgb(kws["color"]), kws["alpha"])
                ],
            )
            fig.data[0].legendgroup = str(self.label)
            fig.data[0].name = self.label
            fig.data[0].showlegend=True
            ax(fig)
            ax.figure.update_layout(fig.layout)
        else:
            # TODO abstraction
            ci_kws = {"color": kws["color"]}
            ci_kws["linewidth"] = mpl.rcParams["lines.linewidth"] * 1.75
            kws.setdefault("s", 50)

            xs, ys, cis = self.estimate_data
            if [ci for ci in cis if ci is not None]:
                for x, ci in zip(xs, cis):
                    ax.plot([x, x], ci, **ci_kws)
            ax.scatter(xs, ys, **kws)

    def lineplot(self, ax, kws):
        """Draw the model."""
        # Fit the regression model
        grid, yhat, err_bands = self.fit_regression(ax)
        edges = grid[0], grid[-1]

        # Get set default aesthetics
        fill_color = kws["color"]
        lw = kws.pop("lw", mpl.rcParams["lines.linewidth"] * 1.5)
        kws.setdefault("linewidth", lw)

        # Draw the regression line and confidence interval
        data = pd.DataFrame({"x": grid, "y": yhat})

        
        
        fig = px.line(
            data,
            x="x",
            y="y",
            color_discrete_sequence=[
                _convert_color(mpl.colors.to_rgb(kws["color"]), 1)
            ],
        )
        if err_bands is not None:
            fig.data[0].error_y = dict(
                type="data",
                array=err_bands[1] - data['y'],
                color="rgba(0, 0, 0, 0)",
            )
        assert(len(fig.data) == 1)
        legendgroup = str(self.label)
        fig.data[0].legendgroup = legendgroup
        ax(fig)
        ax.figure.update_layout(fig.layout)

        # line, = ax.plot(grid, yhat, **kws)
        # if not self.truncate:
        #     line.sticky_edges.x[:] = edges  # Prevent mpl from adding margin
        if err_bands is not None:
            ax.fill_between(
                grid,
                *err_bands,
                rgba=_convert_color(mpl.colors.to_rgb(fill_color), 0.15),
                legend=False,
                legendgroup=legendgroup,
                hoverinfo="skip",
            )


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
    plotter = RegressionPlotter(
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

    p = DistributionPlotter(
        data=data, variables=DistributionPlotter.get_semantics(locals())
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
                    data.hovertemplate = data.hovertemplate.replace(f'{plot_variables[key]}=', f'{val}=')
    
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

    if kind == "scatter":

        plotter = ScatterPlotter
        func = scatterplot
        markers = True if markers is None else markers

    elif kind == "line":

        plotter = LinePlotter
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

        in_legend = {i.name for i in g.figure.data if i.showlegend}
        counts = {key: 0 for key in in_legend}
        for data_ in g.figure.data:
            if not data_.showlegend:
                continue
            counts[data_.name] += 1
            if counts[data_.name] > 1:
                data_.showlegend = False
        # if p.legend_data:
        #     g.add_legend(
        #         legend_data=p.legend_data,
        #         label_order=p.legend_order,
        #         title=p.legend_title,
        #         adjust_subtitles=True,
        #     )

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
    x=None, y=None,
    data=None,
    hue=None, col=None, row=None,  # TODO move before data once * is enforced
    palette=None, col_wrap=None, height=5, aspect=1, markers="o",
    hue_order=None, col_order=None, row_order=None,
    legend=True, x_estimator=None, x_bins=None,
    x_ci="ci", scatter=True, fit_reg=True, ci=95, n_boot=1000,
    units=None, seed=None, order=1, logistic=False, lowess=False,
    robust=False, logx=False, x_partial=None, y_partial=None,
    truncate=True, x_jitter=None, y_jitter=None, scatter_kws=None,
    line_kws=None, facet_kws=None
):


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
        data, row=row, col=col, hue=hue,
        palette=palette,
        row_order=row_order, col_order=col_order, hue_order=hue_order,
        height=height, aspect=aspect, col_wrap=col_wrap,
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
        raise ValueError(("markers must be a singleton or a list of markers "
                          "for each level of the hue variable"))
    facets.hue_kws = {"marker": markers}

    def update_datalim(data, x, y, ax, **kws):
        pass
        # xys = data[[x, y]].to_numpy().astype(float)
        # ax.update_datalim(xys, updatey=False)
        # ax.autoscale_view(scaley=False)

    facets.map_dataframe(update_datalim, x=x, y=y)

    # Draw the regression plot on each facet
    regplot_kws = dict(
        x_estimator=x_estimator, x_bins=x_bins, x_ci=x_ci,
        scatter=scatter, fit_reg=fit_reg, ci=ci, n_boot=n_boot, units=units,
        seed=seed, order=order, logistic=logistic, lowess=lowess,
        robust=robust, logx=logx, x_partial=x_partial, y_partial=y_partial,
        truncate=truncate, x_jitter=x_jitter, y_jitter=y_jitter,
        scatter_kws=scatter_kws, line_kws=line_kws,
    )
    facets.map_dataframe(regplot, x=x, y=y, **regplot_kws)
    facets.set_axis_labels(x, y)

    # Add a legend
    if legend and (hue is not None) and (hue not in [col, row]):
        facets._figure.layout.legend.title = hue
    facets.update_hover({'x': 'x', 'y': 'y'}, {'x': x, 'y': y})
    return facets._figure


def lineplot(**kwargs):
    if kwargs.get('ax') is None:
        _, ax = subplots() 
    else:
        ax = kwargs.pop('ax')

    if 'style' in kwargs:
        raise NotImplementedError('size isn\'t supported yet. Use relplot instead?')
    if 'size' in kwargs:
        raise NotImplementedError('size isn\'t supported yet. Use relplot instead?')
    if kwargs.get('err_style') == 'bars':
        raise NotImplementedError('err_style = "bars" is not supported yet, use "bands" instead.')
        
    fig = _sns.lineplot(ax=ax, **kwargs).figure
    legend_map = {','.join(i.marker.color.split(',')[:3]): i.legendgroup for i in fig.data if i.legendgroup}
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
        _data.legendgroup = legend_map.get(','.join(_color.split(',')[:3]))
        _data.name = legend_map.get(','.join(_color.split(',')[:3]))
        if not _data.hoverinfo == 'skip':
            _data.hovertemplate = f'{x_label}=%{{x}}<br>{y_label}=%{{y}}<extra></extra>'

    return fig

def kdeplot(**kwargs):
    if kwargs.get('ax') is None:
        _, ax = subplots() 
    else:
        ax = kwargs.pop('ax')
    fig = _sns.kdeplot(ax=ax, **kwargs).figure
    legend_map = {','.join(i.marker.color.split(',')[:3]): i.legendgroup for i in fig.data if i.legendgroup}
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
        _data.legendgroup = legend_map.get(','.join(_color.split(',')[:3]))
        _data.name = legend_map.get(','.join(_color.split(',')[:3]))
        if not _data.hoverinfo == 'skip':
            _data.hovertemplate = f'{x_label}=%{{x}}<br>{y_label}=%{{y}}<extra></extra>'
    return fig
