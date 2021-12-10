import functools

import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from plotly.subplots import make_subplots
import bornly._seaborn.seaborn as _sns

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

    def fill_between(self, x, y1, y2, alpha=None, color=None, rgba=None, legend=None, label=None):
        # need to figure this out if I want to make progress...
        if rgba is None and color is not None and alpha is not None:
            rgba = _convert_color(color, alpha)
        self._figure.add_traces(go.Scatter(x=x, y = y2,
                                line = dict(color='rgba(0,0,0,0)'), showlegend=False))

        self._figure.add_traces(go.Scatter(x=x, y = y1,
                                line = dict(color='rgba(0,0,0,0)'),
                                fill='tonexty', 
                                fillcolor=rgba,
                                showlegend=legend,
                                name=label,
                                ))


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
    return f'rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {alpha})'
def _deconvert_rgba(rgba, alpha):
    import re
    re.findall(r'\d+', rgba)
    breakpoint()

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
            'data_frame': data,
            'x': self.variables['x'],
            'y': self.variables['y'],
        }
        if 'palette' in self.variables:
            plotting_kwargs['color_discrete_sequence'] = _get_colors(-1, 1, _sns.color_palette(self.variables['palette']))
        else:
            plotting_kwargs['color_discrete_sequence'] = _get_colors(-1, 1)
        if 'hue' in self.variables:
            plotting_kwargs['color'] = self.variables['hue']
        if 'size' in self.variables:
            plotting_kwargs['size'] = self.variables['size']
        if 'sizes' in self.variables:
            plotting_kwargs['size_max'] = self.variables['sizes'][1]
        
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
    if palette is not None:
        p.variables['palette'] = palette
    if sizes is not None:
        p.variables['sizes'] = sizes
    if hue_order is not None:
        raise NotImplementedError('hue_order isn\'t available yet')

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

        kws.setdefault("markeredgewidth", kws.pop("mew", .75))
        kws.setdefault("markeredgecolor", kws.pop("mec", "w"))

        # Set default error kwargs
        err_kws = self.err_kws.copy()
        if self.err_style == "band":
            err_kws.setdefault("alpha", .2)
        elif self.err_style == "bars":
            pass
        elif self.err_style is not None:
            err = "`err_style` must be 'band' or 'bars', not {}"
            raise ValueError(err.format(self.err_style))

        # Initialize the aggregation object
        agg = _sns._statistics.EstimateAggregator(
            self.estimator, self.errorbar, n_boot=self.n_boot, seed=self.seed,
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
        grouping_vars = "hue"#, "size", "style"
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

            plotting_kwargs['color_discrete_sequence'] = [line_color]
                
            fig = px.line(**plotting_kwargs)
            ax(fig)
            ax.set_xlabel(self.variables['x'])
            ax.set_ylabel(self.variables['y'])

            # --- Draw the confidence intervals

            if self.estimator is not None and self.errorbar is not None:

            #     # TODO handling of orientation will need to happen here

                ax.fill_between(
                    sub_data["x"], sub_data["ymin"], sub_data["ymax"],
                    rgba=fill_color,
                    legend=False,
                )

                if self.err_style == "bars":
                    raise NotImplementedError('Can\'t use bars are err_style')

        if 'hue' in sub_vars:
            ax._figure.update_layout(legend_title=self.variables['hue'])



def lineplot(
    *,
    x=None, y=None,
    hue=None, size=None, style=None,
    data=None,
    palette=None, hue_order=None, hue_norm=None,
    sizes=None, size_order=None, size_norm=None,
    dashes=True, markers=None, style_order=None,
    units=None, estimator="mean", n_boot=1000, seed=None,
    sort=True, err_style="band", err_kws=None,
    legend="auto",
    errorbar=("ci", 95),
    ax=None, **kwargs
):

    variables = LinePlotter.get_semantics(locals())
    p = LinePlotter(
        data=data, variables=variables,
        estimator=estimator, n_boot=n_boot, seed=seed,
        sort=sort, err_style=err_style, err_kws=err_kws, legend=legend,
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
        _sns.distributions._check_argument("multiple", ["layer", "stack", "fill"], multiple)

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
                default_alpha = .25
            else:
                default_alpha = .75
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
                    raise NotImplementedError('fill isn\'t supported yet')

                else:
                    sub_data = density.to_frame(name='support').reset_index().rename(columns={'index': 'density'})
                    if 'hue' in sub_vars:
                        sub_data['hue'] = sub_vars['hue']
                    plotting_kwargs = {
                        'data_frame': sub_data,
                        'x': 'density',
                        'y': 'support',
                    }
                    if 'hue' in sub_vars:
                        plotting_kwargs['color'] = 'hue'
                        color = _convert_color(self._hue_map(sub_vars['hue']), 1)
                    else:
                        color = _get_colors(1, 1)[0]
                    plotting_kwargs['color_discrete_sequence'] = [color]
                    fig = px.line(**plotting_kwargs)
                    ax(fig)
                    ax.set_xlabel(self.variables['x'])
                    ax.set_ylabel('Density')

        # --- Finalize the plot ----



def kdeplot(
    x=None,  # Allow positional x, because behavior will not change with reorg
    *,
    gridsize=200,  # TODO maybe depend on uni/bivariate?
    cut=3, clip=None, legend=True, cumulative=False,
    cbar=False, cbar_ax=None, cbar_kws=None,
    ax=None,

    # New params
    weights=None,  # TODO note that weights is grouped with semantics
    hue=None, palette=None, hue_order=None, hue_norm=None,
    multiple="layer", common_norm=True, common_grid=False,
    levels=10, thresh=.05,
    bw_method="scott", bw_adjust=1, log_scale=None,
    color=None, fill=None,

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

    # else:

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