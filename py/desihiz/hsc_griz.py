#!/usr/bin/env python

import os
import sys
import numpy as np
import fitsio
from astropy.table import Table
from astropy.io import fits

from desitarget.randoms import randoms_in_a_brick_from_edges
from desitarget.geomask import hp_in_box
from desitarget.brightmask import make_bright_star_mask_in_hp, is_in_bright_mask
from astropy.coordinates import SkyCoord
from astropy import units

from desihiz.hizmerge_io import get_cosmos2020_fn, match_coord
from desihiz.plot_utils import custom_hexbin, plot_star_contours

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from desispec.tile_qa_plot import get_quantz_cmap

from desiutil.log import get_logger

log = get_logger()


def get_hscwide_pz_fns_bounds(field, release):

    assert field in ["cosmos"]
    assert release in ["pdr2", "pdr3"]

    pdir = os.path.join(os.getenv("DESI_ROOT"), "users", "raichoor", "hsc", release)

    if field == "cosmos":

        zfn = get_cosmos2020_fn("cosmos_yr1")
        zkeys = {
            "ID": "ID",
            "RA": "ALPHA_J2000",
            "DEC": "DELTA_J2000",
            "ZPHOT": "lp_zBEST",
        }

        # /global/cfs/cdirs/desi/users/raichoor/hsc/pdr3/hsc-pdr3-wide-photometry-cosmos.query
        pfn = os.path.join(
            pdir, "hsc-{}-wide-photometry-{}.fits".format(release, field)
        )
        ramin, ramax, decmin, decmax = 148, 152, 0, 4

    log.info("field\t: {}".format(field))
    log.info("pfn\t: {}".format(pfn))
    log.info("zfn\t: {}".format(zfn))
    log.info(
        "ramin, ramax, decmin, decmax\t: {}, {}, {}, {}".format(
            ramin, ramax, decmin, decmax
        )
    )
    log.info("zkeys\t: {}".format(zkeys))

    return pfn, zfn, ramin, ramax, decmin, decmax, zkeys


# from Arjun odin/cosmos...
def get_bs_radii_deg(bs_mags):
    return 0.07 * (6.3 / bs_mags) ** 2


def get_bsm(maglim, ramin, ramax, decmin, decmax):
    nside, nest = 64, True
    radecbox = [ramin - 2, ramax + 2, decmin - 2, decmax + 2]  # safe..
    pixs = hp_in_box(nside, radecbox)
    mask = list()
    for pix in pixs:
        mask.append(make_bright_star_mask_in_hp(nside, pix, maglim=maglim))
    bsm = Table(np.concatenate(mask))
    bsm["IN_RADIUS"] = 3600 * get_bs_radii_deg(bsm["REF_MAG"])
    return bsm


def get_is_bsm(d, bsm):
    ismask, _ = is_in_bright_mask(d, bsm, inonly=True)
    return ismask[0]


def get_area(ramin, ramax, decmin, decmax, rdens=100000, maglim=12):
    bsm = get_bsm(maglim, ramin, ramax, decmin, decmax)
    r = Table()
    r["RA"], r["DEC"] = randoms_in_a_brick_from_edges(
        ramin, ramax, decmin, decmax, density=rdens
    )
    r["MASK"] = get_is_bsm(r, bsm)
    area = len(r) / rdens
    unmsk_area = (~r["MASK"]).sum() / rdens
    log.info("area = {:.2f} deg2, unmsk_area = {:.2f} deg2".format(area, unmsk_area))
    return area, unmsk_area


# rough eye-balled calibration on suprime data
def recalibrate_c20zphot(c20zphots):
    sel = c20zphots > 2
    c20zphots[sel] += 0.02 * (1 + c20zphots[sel])
    return c20zphots


def get_match_pz(
    field,
    hscrelease,
    rmin=23.0,
    rmax=24.5,
    bsm_maglim=12,
    search_radius=1.0,
):

    # HSC wide file + boundaries
    pfn, zfn, ramin, ramax, decmin, decmax, zkeys = get_hscwide_pz_fns_bounds(
        field, hscrelease
    )

    # photometry
    p = Table(fitsio.read(pfn))
    for key in p.colnames:
        p[key].name = key.upper()
    keys = ["OBJECT_ID", "RA", "DEC", "I_EXTENDEDNESS_VALUE"]
    for band in ["G", "R", "I", "Z"]:
        p["MAG_{}".format(band)] = (
            p["{}_CMODEL_MAG".format(band)] - p["A_{}".format(band)]
        )
        p["{}_CMODEL_MAGERR".format(band)].name = "MAGERR_{}".format(band)
        keys += ["MAG_{}".format(band), "MAGERR_{}".format(band)]
    p.keep_columns(keys)
    sel = (p["MAG_R"] > rmin) & (p["MAG_R"] < rmax)
    for band in ["G", "R", "I", "Z"]:
        sel &= np.isfinite(p["MAG_{}".format(band)])
    p = p[sel]

    # bsm
    bsm = get_bsm(bsm_maglim, ramin, ramax, decmin, decmax)
    p["MASK_BSM"] = get_is_bsm(p, bsm)

    # area
    area, unmsk_area = get_area(ramin, ramax, decmin, decmax, rdens=100000, maglim=12)
    p.meta["AREA"], p.meta["BSMAREA"] = area, unmsk_area

    # zphot
    z = Table(
        fitsio.read(
            zfn, columns=[zkeys["ID"], zkeys["RA"], zkeys["DEC"], zkeys["ZPHOT"]]
        )
    )
    z["ZPHOT"] = z[zkeys["ZPHOT"]].copy()
    if field == "cosmos":
        z["ZPHOT"] = recalibrate_c20zphot(z["ZPHOT"])
    # safe..
    sel = (np.isfinite(z["ZPHOT"])) & (z["ZPHOT"] >= 0)
    z = z[sel]

    # match
    iip, iiz, _, _, _ = match_coord(
        p["RA"], p["DEC"], z[zkeys["RA"]], z[zkeys["DEC"]], search_radius=search_radius
    )
    p["ISZPHOT"] = np.zeros(len(p), dtype=bool)
    p["ISZPHOT"][iip] = True
    p["ZPHOT"] = 0.0
    p["ZPHOT"][iip] = z["ZPHOT"][iiz]
    for key in ["ID"]:
        p["ZPHOT_{}".format(key)] = np.zeros_like(z[zkeys[key]], shape=(len(p),))
        p["ZPHOT_{}".format(key)][iip] = z[zkeys[key]][iiz]

    # set HSC/stars to zphot=0
    sel = (p["ISZPHOT"]) & (p["I_EXTENDEDNESS_VALUE"] == 0)
    p["ZPHOT"][sel] = 0

    p.meta["PHOTFN"] = pfn
    p.meta["ZPHOTFN"] = zfn
    p.meta["RAMIN"], p.meta["RAMAX"], p.meta["DECMIN"], p.meta["DECMAX"] = (
        ramin,
        ramax,
        decmin,
        decmax,
    )
    p.meta["RMIN"], p.meta["RMAX"] = rmin, rmax
    p.meta["BSMMAG"] = bsm_maglim

    return p


# v20231206: https://desisurvey.slack.com/archives/C0351RV8CBE/p1701383913231779
def get_mysel(d, selection, rmin, rmax):

    assert selection in ["v20231206"]

    mydict = {}

    if selection == "v20231206":

        # rmag
        mydict["R_SEL"] = (d["MAG_R"] > rmin) & (d["MAG_R"] < rmax)

        # (r, ri)
        mydict["R_IZ_SEL"] = d["MAG_R"] > 23.5 + 5 * (d["MAG_I"] - d["MAG_Z"])

        # (gr, ri)
        mydict["GR_RI_SEL"] = d["MAG_R"] - d["MAG_I"] > -0.3
        mydict["GR_RI_SEL"] &= (d["MAG_G"] - d["MAG_R"] > 0.2) & (
            d["MAG_G"] - d["MAG_R"] < 0.6
        )
        mydict["GR_RI_SEL"] &= d["MAG_G"] - d["MAG_R"] > 0.15 + 2 * (
            d["MAG_R"] - d["MAG_I"]
        )
        mydict["GR_RI_SEL"] &= d["MAG_G"] - d["MAG_R"] < 0.8 - 2 * (
            d["MAG_R"] - d["MAG_I"]
        )

        # all
        mydict["SELECTION"] = (~d["MASK_BSM"]).copy()
        mydict["SELECTION"] &= mydict["R_SEL"]
        mydict["SELECTION"] &= mydict["GR_RI_SEL"]
        mydict["SELECTION"] &= mydict["R_IZ_SEL"]

    return mydict


def get_density(d, rmin, rmax, rounding=10):

    area = d.meta["BSMAREA"]

    sel = (d["SELECTION"]) & (d["MAG_R"] > rmin) & (d["MAG_R"] < rmax)

    density = int(rounding * np.round(sel.sum() / area / rounding, 0))

    log.info("approx. density for {} < rmag < {}: {} deg2".format(rmin, rmax, density))

    return density


def create_pdf(outpdf, d, rmins, rmaxs, densities, zmin, zmax, gridsize=50, cmap=None):

    if cmap is None:
        cmap = get_quantz_cmap(matplotlib.cm.jet, 11, 0, 1)

    log.info(densities)

    # Table() with zphot
    z_d = d[d["ISZPHOT"]]

    # Table() with stars
    star_d = d[d["I_EXTENDEDNESS_VALUE"] == 0]

    zs = z_d["ZPHOT"]
    selz = (zs > zmin) & (zs < zmax)
    zbins = np.arange(0, 4.1, 0.1)

    mysel = z_d["SELECTION"]
    if "SELECT" in d.meta:
        selname = d.meta["SELECT"]
    else:
        selname = "-"

    log.info(
        "{} selection : {:.0f}/deg2, {:.0f}% with {}<z<{}".format(
            selname,
            densities[0],
            100 * ((mysel) & (selz)).sum() / mysel.sum(),
            zmin,
            zmax,
        )
    )

    # what we ll plot
    myfuncs = [np.mean]
    css = [selz]
    clabs = ["Fraction with {} < zphot < {}".format(zmin, zmax)]
    clims = [(0.0, 0.5)]
    extends = ["max"]

    with PdfPages(outpdf) as pdf:

        for rmin, rmax, density in zip(rmins, rmaxs, densities):

            # parent sample in rmin < r < rmax
            myall_selr = (d["SELECTION"]) & (d["MAG_R"] >= rmin) & (d["MAG_R"] < rmax)

            # parent sample with zphot in rmin < r < rmax
            selr = (z_d["MAG_R"] >= rmin) & (z_d["MAG_R"] < rmax)

            # selected stars in rmin < r < rmax
            star_selr = (star_d["MAG_R"] >= rmin) & (star_d["MAG_R"] < rmax)

            # selection in rmin < r < rmax
            myselr = (mysel) & (selr)

            title = "{} < MAG_R < {}: ~{}/deg2".format(rmin, rmax, density)
            log.info(
                "{}\t{}\t{:.0f}/deg2\t{:.0f}%".format(
                    rmin,
                    rmax,
                    density,
                    100 * ((myselr) & (selz)).sum() / myselr.sum(),
                )
            )

            fig = plt.figure(figsize=(25, 5))
            gs = gridspec.GridSpec(1, 4, wspace=0.25)

            ip = 0
            for xs, ys, star_xs, star_ys, xlim, ylim, xlab, ylab in zip(
                [z_d["MAG_R"] - z_d["MAG_I"], z_d["MAG_I"] - z_d["MAG_Z"]],
                [z_d["MAG_G"] - z_d["MAG_R"], z_d["MAG_R"]],
                [star_d["MAG_R"] - star_d["MAG_I"], star_d["MAG_I"] - star_d["MAG_Z"]],
                [star_d["MAG_G"] - star_d["MAG_R"], star_d["MAG_R"]],
                [np.array([-0.5, 2.0]), np.array([-0.5, 1.0])],
                [np.array([-0.5, 2.0]), np.array([23.0, 24.75])],
                ["R - I [AB]", "I - Z [AB]"],
                ["G - R [AB]", "R [AB]"],
            ):

                extent = (xlim[0], xlim[1], ylim[0], ylim[1])

                for myfunc, cs, clab, clim, extend in zip(
                    myfuncs, css, clabs, clims, extends
                ):
                    ax = fig.add_subplot(gs[ip])
                    ax.set_title(title)
                    sc, _, _, _, _ = custom_hexbin(
                        ax,
                        xs[selr],
                        ys[selr],
                        cs[selr],
                        myfunc,
                        gridsize,
                        extent,
                        cmap,
                        clim,
                        alpha_mean=0.5,
                    )
                    if star_selr.sum() > 0:
                        plot_star_contours(
                            ax, xlim, ylim, star_xs[star_selr], star_ys[star_selr]
                        )
                    # ax.scatter(xs[myselr], ys[myselr], c="k", s=5, alpha=1, zorder=2)
                    cbar = plt.colorbar(sc, extend=extend)
                    cbar.set_label(clab)
                    cbar.mappable.set_clim(clim)
                    ax.set_xlabel(xlab)
                    ax.set_ylabel(ylab)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.grid()
                    ax.legend(loc=1)

                    # rmin, rmax cuts
                    if ip == 1:
                        ax.axhline(rmin, color="k", ls="--", lw=2, zorder=2)
                        ax.axhline(rmax, color="k", ls="--", lw=2, zorder=2)
                    if ip == 2:
                        ax.axhline(rmin, color="k", ls="--", lw=2, zorder=2)
                        ax.axhline(rmax, color="k", ls="--", lw=2, zorder=2)

                    # v20231206 cuts
                    if selname == "v20231206":
                        if ip == 0:
                            tmpxs = [-0.3, 0.0, 0.15, 0.1, -0.3, -0.3]
                            tmpys = [0.2, 0.2, 0.5, 0.6, 0.6, 0.2]
                            ax.plot(tmpxs, tmpys, color="k", ls="--", lw=2, zorder=2)
                        if ip == 1:
                            ax.plot(
                                xlim,
                                5 * xlim + 23.5,
                                color="k",
                                ls="--",
                                lw=2,
                                zorder=2,
                            )
                        if ip == 2:
                            ax.plot(
                                xlim,
                                2 * xlim + 23.2,
                                color="k",
                                ls="--",
                                lw=2,
                                zorder=2,
                            )
                            ax.axvline(0.2, color="k", ls="--", lw=2, zorder=2)
                            ax.axvline(0.6, color="k", ls="--", lw=2, zorder=2)

                    ip += 1

            # maghist
            ax = fig.add_subplot(gs[ip])
            mbins = np.arange(23, 26.1, 0.1)
            for band, histtype, alpha in zip(
                ["G", "R", "I", "Z"],
                ["step", "stepfilled", "step", "step"],
                [1, 0.5, 1, 1],
            ):
                _ = ax.hist(
                    d["MAG_{}".format(band)][myall_selr],
                    bins=mbins,
                    density=True,
                    histtype=histtype,
                    alpha=alpha,
                    label="{}-BAND".format(band),
                )
            ax.set_title(title)
            ax.set_xlabel("MAG [AB]")
            ax.set_ylabel("Normalized counts")
            ax.set_xlim(mbins[0], mbins[-1])
            ax.set_ylim(0, 4)
            ax.grid()
            ax.legend(loc=2)
            ip += 1

            # zhist
            txt = "{:.0f}% with {} < zphot < {}".format(
                100 * ((myselr) & (selz)).sum() / myselr.sum(), zmin, zmax
            )
            ax = fig.add_subplot(gs[ip])
            ax.text(0.05, 0.9, txt, transform=ax.transAxes)
            _ = ax.hist(
                z_d["ZPHOT"][myselr],
                bins=zbins,
                histtype="stepfilled",
                density=True,
                alpha=0.5,
            )
            ax.axvline(zmin, color="k", ls="--", zorder=2)
            ax.axvline(zmax, color="k", ls="--", zorder=2)
            ax.set_title(title)
            ax.set_xlabel("ZPHOT")
            ax.set_ylabel("Normalized counts")
            ax.set_xlim(zbins[0], zbins[-1])
            ax.set_ylim(0, 2)
            ax.grid()
            ip += 1

            pdf.savefig(fig, bbox_inches="tight")
            plt.close()
