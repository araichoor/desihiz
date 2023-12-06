#!/usr/bin/env python

import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.transforms as mtransforms


def custom_hexbin_getvals(
    ax,
    xs,
    ys,
    C,
    reduce_C_function,
    gridsize,
    extent,
    cmap,
    clim,
    mincnt=None,
    weights=None,
    alpha_mean=0.5,
    alpha_log=True,
):
    if np.nanmax(np.abs(ys)) >= 1000000:
        print(np.nanmax(np.abs(ys)))
        sys.exit("np.nanmax(np.abs(ys)) >= 1000000; exiting")
    # first get stats on C (only where data...)
    hb = ax.hexbin(
        xs,
        ys,
        C=C,
        reduce_C_function=reduce_C_function,
        gridsize=gridsize,
        extent=extent,
        mincnt=mincnt,
        visible=False,
    )
    hb_xs = hb.get_offsets()[:, 0]
    hb_ys = hb.get_offsets()[:, 1]
    hb_cs = hb.get_array()
    # then get the counts (for all points of the grid)
    if weights is not None:
        hb = ax.hexbin(
            xs,
            ys,
            C=weights,
            reduce_C_function=np.sum,
            gridsize=gridsize,
            extent=extent,
            visible=False,
        )
    else:
        hb = ax.hexbin(xs, ys, C=None, gridsize=gridsize, extent=extent, visible=False)
    tmpxs = hb.get_offsets()[:, 0]
    tmpys = hb.get_offsets()[:, 1]
    tmpns = hb.get_array()

    # twisted way to match...
    hb_unqs = 1000000 * hb_xs + hb_ys
    tmpunqs = 1000000 * tmpxs + tmpys
    sel = np.in1d(tmpunqs, hb_unqs)
    assert sel.sum() == hb_unqs.size
    assert (tmpxs[sel] != hb_xs).sum() == 0
    assert (tmpys[sel] != hb_ys).sum() == 0
    hb_ns = tmpns[sel]
    # transparency = f(nb of parent obj)
    # normalizing to mean, then setting mean to alpha_mean
    if alpha_log:
        hb_as = 1 + np.log10(hb_ns / hb_ns.mean())
    else:
        hb_as = hb_ns / hb_ns.mean()
    hb_as = np.clip(alpha_mean * hb_as, 0, 1)
    return hb_xs, hb_ys, hb_cs, hb_as


def custom_hexbin_polygon(gridsize, extent):
    # hexbin-like hexagon
    # https://matplotlib.org/stable/_modules/matplotlib/axes/_axes.html#Axes.hexbin
    xmin, xmax, ymin, ymax = extent
    padding = 1.0e-9 * (xmax - xmin)
    xmin -= padding
    xmax += padding
    nx = gridsize
    ny = int(nx / np.sqrt(3))
    sx = (xmax - xmin) / nx
    sy = (ymax - ymin) / ny
    return [sx, sy / 3] * np.array(
        [[0.5, -0.5], [0.5, 0.5], [0.0, 1.0], [-0.5, 0.5], [-0.5, -0.5], [0.0, -1.0]]
    )


def custom_hexbin_plotcol(
    ax, hb_xs, hb_ys, hb_cs, hb_as, gridsize, extent, clim, cmap, zorder=0
):
    # hexbin-like hexagon
    polygon = custom_hexbin_polygon(gridsize, extent)
    offsets = np.zeros((len(hb_xs), 2), float)
    offsets[:, 0], offsets[:, 1] = hb_xs, hb_ys
    collection = mcoll.PolyCollection(
        [polygon],
        edgecolors="face",
        offsets=offsets,
        transOffset=mtransforms.AffineDeltaTransform(ax.transData),
    )
    collection.set_array(hb_cs)
    collection.set_cmap(cmap)
    collection.set_norm(None)
    collection.set_alpha(hb_as)
    collection.set_zorder(zorder)
    collection._scale_norm(None, clim[0], clim[1])
    ax.add_collection(collection, autolim=False)


def custom_hexbin(
    ax,
    xs,
    ys,
    C,
    reduce_C_function,
    gridsize,
    extent,
    cmap,
    clim,
    mincnt=None,
    weights=None,
    alpha_mean=0.5,
    alpha_log=True,
    zorder=0,
):
    hb_xs, hb_ys, hb_cs, hb_as = custom_hexbin_getvals(
        ax,
        xs,
        ys,
        C,
        reduce_C_function,
        gridsize,
        extent,
        cmap,
        clim,
        mincnt=mincnt,
        weights=weights,
        alpha_mean=0.5,
        alpha_log=True,
    )
    custom_hexbin_plotcol(
        ax, hb_xs, hb_ys, hb_cs, hb_as, gridsize, extent, clim, cmap, zorder=zorder
    )
    sc = ax.scatter(None, None, c=0.0, cmap=cmap, vmin=clim[0], vmax=clim[1])
    return sc, hb_xs, hb_ys, hb_cs, hb_as


def plot_star_contours(ax, xlim, ylim, star_xs, star_ys):

    # stars (68.3% contour)
    h, xedges, yedges = np.histogram2d(
        star_xs,
        star_ys,
        bins=[
            np.linspace(xlim[0], xlim[1], 50),
            np.linspace(ylim[0], ylim[1], 50),
        ],
    )
    xcens = 0.5 * (xedges[1:] + xedges[:-1])
    ycens = 0.5 * (yedges[1:] + yedges[:-1])
    tmph = h.flatten()
    tmph = tmph[tmph.argsort()[::-1]]
    tmpsum = tmph.cumsum()
    fracs = [0.683]
    levels = [tmph[tmpsum > frac * h.sum()][0] for frac in fracs]
    colors = np.array(["k" for level in levels])
    ax.contour(xcens, ycens, h.T, levels, colors=colors)
    ax.plot(np.nan, np.nan, color="k", label="Stellar locus (68.3% contour)")
