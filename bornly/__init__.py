import functools
import copy
import matplotlib as mpl

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

    def fill_between(
        self, x, y1, y2, alpha=None, color=None, rgba=None, legend=None, label=None
    ):
        # need to figure this out if I want to make progress...
        if rgba is None and color is not None and alpha is not None:
            rgba = _convert_color(color, alpha)
        self._figure.add_traces(
            go.Scatter(x=x, y=y2, line=dict(color="rgba(0,0,0,0)"), showlegend=False)
        )

        self._figure.add_traces(
            go.Scatter(
                x=x,
                y=y1,
                line=dict(color="rgba(0,0,0,0)"),
                fill="tonexty",
                fillcolor=rgba,
                showlegend=legend,
                name=label,
            )
        )


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


def _convert_color(color, alpha):
    return f"rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {alpha})"


def _deconvert_rgba(rgba, alpha):
    import re

    re.findall(r"\d+", rgba)


def _get_colors(n, alpha, palette=None):
    if palette is None:
        palette = _sns.color_palette()
    if n == -1:
        colors = palette
    else:
        colors = palette[:n]
    return [_convert_color(color, alpha) for color in colors]


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
            plotting_kwargs["color_discrete_sequence"] = _get_colors(
                -1, 1, _sns.color_palette(self.variables["palette"])
            )
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
        raise NotImplementedError("hue_order isn't available yet")

    if ax is None:
        _, ax = subplots()

    if not p.has_xy_data:
        return ax._figure

    p.plot(ax, kwargs)
    return ax._figure


class LinePlotter(_sns.relational._LinePlotter):
    def plot(self, ax, kws):
        """Draw the plot onto an axes, passing matplotlib kwargs."""

        # Draw a test plot, using the passed in kwargs. The goal here is to
        # honor both (a) the current state of the plot cycler and (b) the
        # specified kwargs on all the lines we will draw, overriding when
        # relevant with the data semantics. Note that we won't cycle
        # internally; in other words, if ``hue`` is not used, all elements will
        # have the same color, but they will have the color that you would have
        # gotten from the corresponding matplotlib function, and calling the
        # function will advance the axes property cycle.

        kws.setdefault("markeredgewidth", kws.pop("mew", 0.75))
        kws.setdefault("markeredgecolor", kws.pop("mec", "w"))

        # Set default error kwargs
        err_kws = self.err_kws.copy()
        if self.err_style == "band":
            err_kws.setdefault("alpha", 0.2)
        elif self.err_style == "bars":
            pass
        elif self.err_style is not None:
            err = "`err_style` must be 'band' or 'bars', not {}"
            raise ValueError(err.format(self.err_style))

        # Initialize the aggregation object
        agg = _sns._statistics.EstimateAggregator(
            self.estimator,
            self.errorbar,
            n_boot=self.n_boot,
            seed=self.seed,
        )

        # TODO abstract variable to aggregate over here-ish. Better name?
        agg_var = "y"
        grouper = ["x"]

        # TODO How to handle NA? We don't want NA to propagate through to the
        # estimate/CI when some values are present, but we would also like
        # matplotlib to show "gaps" in the line when all values are missing.
        # This is straightforward absent aggregation, but complicated with it.
        # If we want to use nas, we need to conditionalize dropna in iter_data.

        # Loop over the semantic subsets and add to the plot
        grouping_vars = "hue"  # , "size", "style"
        for sub_vars, sub_data in self.iter_data(grouping_vars, from_comp_data=True):

            if self.sort:
                sort_vars = ["units", "x", "y"]
                sort_cols = [var for var in sort_vars if var in self.variables]
                sub_data = sub_data.sort_values(sort_cols)

            if self.estimator is not None:
                if "units" in self.variables:
                    # TODO eventually relax this constraint
                    err = "estimator must be None when specifying units"
                    raise ValueError(err)
                grouped = sub_data.groupby(grouper, sort=self.sort)
                # Could pass as_index=False instead of reset_index,
                # but that fails on a corner case with older pandas.
                sub_data = grouped.apply(agg, agg_var).reset_index()

            # TODO this is pretty ad hoc ; see GH2409
            for var in "xy":
                if self._log_scaled(var):
                    for col in sub_data.filter(regex=f"^{var}"):
                        sub_data[col] = np.power(10, sub_data[col])

            # --- Draw the main line(s)
            if "hue" in sub_vars:
                sub_data["hue"] = sub_vars["hue"]
            plotting_kwargs = {
                "data_frame": sub_data.rename(columns=self.variables),
                "x": self.variables["x"],
                "y": self.variables["y"],
            }
            if "hue" in sub_vars:
                plotting_kwargs["color"] = self.variables["hue"]
                line_color = _convert_color(self._hue_map(sub_vars["hue"]), 1)
                fill_color = _convert_color(self._hue_map(sub_vars["hue"]), 0.2)
            else:
                line_color = _get_colors(1, 1)[0]
                fill_color = _get_colors(1, 0.2)[0]
            plotting_kwargs["color_discrete_sequence"] = [line_color]

            fig = px.line(**plotting_kwargs)
            ax(fig)
            ax.set_xlabel(self.variables["x"])
            ax.set_ylabel(self.variables["y"])

            # --- Draw the confidence intervals

            if self.estimator is not None and self.errorbar is not None:

                #     # TODO handling of orientation will need to happen here

                ax.fill_between(
                    sub_data["x"],
                    sub_data["ymin"],
                    sub_data["ymax"],
                    rgba=fill_color,
                    legend=False,
                )

                if self.err_style == "bars":
                    raise NotImplementedError("Can't use bars are err_style")

        if "hue" in sub_vars:
            ax._figure.update_layout(legend_title=self.variables["hue"])


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

    variables = LinePlotter.get_semantics(locals())
    p = LinePlotter(
        data=data,
        variables=variables,
        estimator=estimator,
        n_boot=n_boot,
        seed=seed,
        sort=sort,
        err_style=err_style,
        err_kws=err_kws,
        legend=legend,
        errorbar=errorbar,
    )

    if ax is None:
        _, ax = subplots()

    if not p.has_xy_data:
        return ax._figure

    p.plot(ax, kwargs)
    return ax._figure


class DistributionPlotter(_sns.distributions._DistributionPlotter):
    def plot_univariate_density(
        self,
        multiple,
        common_norm,
        common_grid,
        warn_singular,
        fill,
        color,
        legend,
        estimate_kws,
        ax,
        **plot_kws,
    ):

        # Handle conditional defaults
        if fill is None:
            fill = multiple in ("stack", "fill")

        # plot_kws = _normalize_kwargs(plot_kws, artist)

        # Input checking
        _sns.distributions._check_argument(
            "multiple", ["layer", "stack", "fill"], multiple
        )

        # Always share the evaluation grid when stacking
        subsets = bool(set(self.variables) - {"x", "y"})
        if subsets and multiple in ("stack", "fill"):
            common_grid = True

        # Check if the data axis is log scaled
        log_scale = self._log_scaled(self.data_variable)

        # Do the computation
        densities = self._compute_univariate_density(
            self.data_variable,
            common_norm,
            common_grid,
            estimate_kws,
            log_scale,
            warn_singular,
        )

        # Adjust densities based on the `multiple` rule
        densities, baselines = self._resolve_multiple(densities, multiple)

        # Control the interaction with autoscaling by defining sticky_edges
        # i.e. we don't want autoscale margins below the density curve
        sticky_density = (0, 1) if multiple == "fill" else (0, np.inf)

        if multiple == "fill":
            # Filled plots should not have any margins
            sticky_support = densities.index.min(), densities.index.max()
        else:
            sticky_support = []

        if fill:
            if multiple == "layer":
                default_alpha = 0.25
            else:
                default_alpha = 0.75
        else:
            default_alpha = 1
        alpha = plot_kws.pop("alpha", default_alpha)  # TODO make parameter?

        # Now iterate through the subsets and draw the densities
        # We go backwards so stacked densities read from top-to-bottom
        for sub_vars, _ in self.iter_data("hue", reverse=True):

            # Extract the support grid and density curve for this level
            key = tuple(sub_vars.items())
            try:
                density = densities[key]
            except KeyError:
                continue
            support = density.index
            fill_from = baselines[key]

            # ax = self._get_axes(sub_vars)

            if "hue" in self.variables:
                sub_color = self._hue_map(sub_vars["hue"])
            else:
                sub_color = color

            # artist_kws = self._artist_kws(
            #     plot_kws, fill, False, multiple, sub_color, alpha
            # )

            # Either plot a curve with observation values on the x axis
            if "x" in self.variables:

                if fill:
                    raise NotImplementedError("fill isn't supported yet")

                else:
                    sub_data = (
                        density.to_frame(name="support")
                        .reset_index()
                        .rename(columns={"index": "density"})
                    )
                    if "hue" in sub_vars:
                        sub_data["hue"] = sub_vars["hue"]
                    plotting_kwargs = {
                        "data_frame": sub_data,
                        "x": "density",
                        "y": "support",
                    }
                    if "hue" in sub_vars:
                        plotting_kwargs["color"] = "hue"
                        color = _convert_color(self._hue_map(sub_vars["hue"]), 1)
                    else:
                        color = _get_colors(1, 1)[0]
                    plotting_kwargs["color_discrete_sequence"] = [color]
                    fig = px.line(**plotting_kwargs)
                    ax(fig)
                    ax.set_xlabel(self.variables["x"])
                    ax.set_ylabel("Density")

        # --- Finalize the plot ----


def kdeplot(
    x=None,  # Allow positional x, because behavior will not change with reorg
    *,
    y=None,
    gridsize=200,  # TODO maybe depend on uni/bivariate?
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
    # New in v0.12
    warn_singular=True,
    **kwargs,
):
    # Handle `n_levels`
    # This was never in the formal API but it was processed, and appeared in an
    # example. We can treat as an alias for `levels` now and deprecate later.
    levels = kwargs.pop("n_levels", levels)

    # Handle "soft" deprecation of shade `shade` is not really the right
    # terminology here, but unlike some of the other deprecated parameters it
    # is probably very commonly used and much hard to remove. This is therefore
    # going to be a longer process where, first, `fill` will be introduced and
    # be used throughout the documentation. In 0.12, when kwarg-only
    # enforcement hits, we can remove the shade/shade_lowest out of the
    # function signature all together and pull them out of the kwargs. Then we
    # can actually fire a FutureWarning, and eventually remove.

    p = DistributionPlotter(
        data=data,
        variables=DistributionPlotter.get_semantics(locals()),
    )

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    if ax is None:
        _, ax = subplots()

    # p._attach(ax, allowed_types=["numeric", "datetime"], log_scale=log_scale)

    # method = ax.fill_between if fill else ax.plot
    # color = _sns.distributions._default_color(method, hue, color, kwargs)

    if not p.has_xy_data:
        return ax

    # Pack the kwargs for statistics.KDE
    estimate_kws = dict(
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        gridsize=gridsize,
        cut=cut,
        clip=clip,
        cumulative=cumulative,
    )

    if p.univariate:

        plot_kws = kwargs.copy()

        p.plot_univariate_density(
            multiple=multiple,
            common_norm=common_norm,
            common_grid=common_grid,
            fill=fill,
            color=color,
            legend=legend,
            warn_singular=warn_singular,
            estimate_kws=estimate_kws,
            ax=ax,
            **plot_kws,
        )

    else:

        raise NotImplementedError("bivariate kde coming soon!")
    #     p.plot_bivariate_density(
    #         common_norm=common_norm,
    #         fill=fill,
    #         levels=levels,
    #         thresh=thresh,
    #         legend=legend,
    #         color=color,
    #         warn_singular=warn_singular,
    #         cbar=cbar,
    #         cbar_ax=cbar_ax,
    #         cbar_kws=cbar_kws,
    #         estimate_kws=estimate_kws,
    #         **kwargs,
    #     )

    return ax._figure


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
        ax._figure.update_layout(fig.layout)


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
    if ax is None:
        _, ax = subplots()
    # if square:
    #     ax.set_aspect("equal")
    plotter.plot(ax, cbar_ax, kwargs)
    return ax._figure


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

        data = pd.DataFrame({'y': self.statistic.flatten()})
        plotting_kwargs = dict(
            data_frame=data,
            x='x',
            y='y',
            color_discrete_sequence=_get_colors(-1, 1, self.colors),
        )
        if self.plot_hues is None:
            data['x'] = self.group_names
            data["err"] = self.confint[:, 1] - data['y']
            plotting_kwargs["color"] = 'x'
        else:
            data[['hue', 'x']] = _cartesian(self.hue_names, self.group_names)
            data["err"] = self.confint[:, :, 1].flatten() - data['y']
            plotting_kwargs["color"] = 'hue'
            plotting_kwargs["barmode"] = "group"

        plotting_kwargs["category_orders"] = {'x': self.group_names}

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
        ax._figure.update_layout(fig.layout)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if self.hue_names is not None:
            ax._figure.layout.legend.title = self.hue_title
        else:
            ax._figure.update_layout(showlegend=False)


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

    if ax is None:
        _, ax = subplots()

    plotter.plot(ax, kwargs)
    return ax._figure

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

            if not hasattr(kws['color'], 'shape') or kws['color'].shape[1] < 4:
                kws.setdefault("alpha", .8)

            x, y = self.scatter_data
            data = pd.DataFrame({'x': x, 'y': y})
            fig = px.scatter(data, x='x', y='y', color_discrete_sequence=[_convert_color(mpl.colors.to_rgb(kws['color']), kws['alpha'])])
            ax(fig)
            # ax.scatter(x, y, **kws)
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
        data = pd.DataFrame({'x': grid, 'y': yhat})
        fig = px.line(data, x='x', y='y', color_discrete_sequence=[_convert_color(mpl.colors.to_rgb(kws['color']), 1)])
        ax(fig)

        # line, = ax.plot(grid, yhat, **kws)
        # if not self.truncate:
        #     line.sticky_edges.x[:] = edges  # Prevent mpl from adding margin
        if err_bands is not None:
            ax.fill_between(grid, *err_bands, rgba=_convert_color(mpl.colors.to_rgb(fill_color), .15), legend=False)

def regplot(
    *,
    x=None, y=None,
    data=None,
    x_estimator=None, x_bins=None, x_ci="ci",
    scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None,
    seed=None, order=1, logistic=False, lowess=False, robust=False,
    logx=False, x_partial=None, y_partial=None,
    truncate=True, dropna=True, x_jitter=None, y_jitter=None,
    label=None, color=None, marker="o",
    scatter_kws=None, line_kws=None, ax=None
):

    plotter = RegressionPlotter(x, y, data, x_estimator, x_bins, x_ci,
                                 scatter, fit_reg, ci, n_boot, units, seed,
                                 order, logistic, lowess, robust, logx,
                                 x_partial, y_partial, truncate, dropna,
                                 x_jitter, y_jitter, color, label)

    if ax is None:
        _, ax = subplots()

    scatter_kws = {} if scatter_kws is None else copy.copy(scatter_kws)
    scatter_kws["marker"] = marker
    line_kws = {} if line_kws is None else copy.copy(line_kws)
    plotter.plot(ax, scatter_kws, line_kws)
    return ax._figure

