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

    def scatter(self, *args, **kwargs):
        pass

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

    def set_ylabel(self, label):
        self.figure.update_yaxes(title_text=label, row=self._row + 1, col=self._col + 1)

    def set_xlabel(self, label):
        self.figure.update_xaxes(title_text=label, row=self._row + 1, col=self._col + 1)

    def set_ylim(self, ylim):
        self.figure.update_yaxes(range=ylim, row=self._row + 1, col=self._col + 1)

    def set_xlim(self, xlim):
        self.figure.update_xaxes(range=xlim, row=self._row + 1, col=self._col + 1)

    def set(self, **kwargs):
        for key, val in kwargs.items():
            getattr(self, f"set_{key}")(val)

    def bar(self, *args, **kwargs):
        pass

    def fill_between(
        self,
        x,
        y1,
        y2,
        alpha=None,
        color=None,
        rgba=None,
        legend=None,
        label=None,
        legendgroup=None,
        hoverinfo=None,
    ):
        # need to figure this out if I want to make progress...
        if rgba is None and color is not None and alpha is not None:
            rgba = _convert_color(color, alpha)
        self.figure.add_trace(
            go.Scatter(
                x=x,
                y=y2,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                legendgroup=legendgroup,
                hoverinfo=hoverinfo,
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
                showlegend=legend,
                name=label,
                legendgroup=legendgroup,
                hoverinfo=hoverinfo,
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


def _convert_color(color, alpha):
    if alpha is not None:
        return f"rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {alpha})"
    return f"rgb({color[0]*255}, {color[1]*255}, {color[2]*255})"


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
                plotting_kwargs["color_discrete_map"] = { key: _convert_color(val, 1) for key, val in self.variables["palette"].items() }
            else:
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
        pass
        # raise NotImplementedError("hue_order isn't available yet")

    if ax is None:
        _, ax = subplots()

    if not p.has_xy_data:
        return ax.figure

    p.plot(ax, kwargs)
    return ax.figure


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

            draw_ci = (
                (self.estimator is not None)
                and (self.errorbar is not None)
                and (not sub_data["ymax"].isna().all())
            )
            if draw_ci:
                fig.data[0].error_y = dict(
                    type="data",
                    array=sub_data["ymax"] - sub_data["y"],
                    color="rgba(0, 0, 0, 0)",
                )
            ax(fig)
            ax.set_xlabel(self.variables["x"])
            ax.set_ylabel(self.variables["y"])

            # --- Draw the confidence intervals

            if draw_ci:

                #     # TODO handling of orientation will need to happen here

                ax.fill_between(
                    sub_data["x"],
                    sub_data["ymin"],
                    sub_data["ymax"],
                    rgba=fill_color,
                    legend=False,
                    legendgroup=sub_vars.get("hue", ""),
                    hoverinfo="skip",
                )

                if self.err_style == "bars":
                    raise NotImplementedError("Can't use bars are err_style")

        if "hue" in sub_vars:
            ax.figure.update_layout(legend_title=self.variables["hue"])


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
        return ax.figure

    p.plot(ax, kwargs)
    return ax.figure


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

    if ax is not None:
        raise NotImplementedError("passing ax is not supported")
    _, ax = subplots()

    # p._attach(ax, allowed_types=["numeric", "datetime"], log_scale=log_scale)

    # method = ax.fill_between if fill else ax.plot
    # color = _sns.distributions._default_color(method, hue, color, kwargs)

    if not p.has_xy_data:
        return ax

    if fill is not None:
        raise NotImplementedError("fill is not yet available")

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

    return ax.figure


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
        data = pd.DataFrame({"x": grid, "y": yhat})
        fig = px.line(
            data,
            x="x",
            y="y",
            color_discrete_sequence=[
                _convert_color(mpl.colors.to_rgb(kws["color"]), 1)
            ],
        )
        ax(fig)

        # line, = ax.plot(grid, yhat, **kws)
        # if not self.truncate:
        #     line.sticky_edges.x[:] = edges  # Prevent mpl from adding margin
        if err_bands is not None:
            ax.fill_between(
                grid,
                *err_bands,
                rgba=_convert_color(mpl.colors.to_rgb(fill_color), 0.15),
                legend=False,
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


def distplot(*args, **kwargs):
    raise NotImplementedError("distplot not available, use histplot instead")


def displot(*args, **kwargs):
    raise NotImplementedError("displot not available yet, use histplot and/or kdeplot")


def lmplot(*args, **kwargs):
    raise NotImplementedError("lmplot not available yet, use regplot instead")


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
        legend=False,
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

    # Show the legend
    if legend:
        # Replace the original plot data so the legend uses
        # numeric data with the correct type
        p.plot_data = plot_data
        p.add_legend_data(g.axes.flat[0])
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

    return g
