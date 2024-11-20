#!/usr/bin/env python

import os
import sys
import numpy as np
import fitsio
from astropy.table import Table
from astropy.io import fits
from astropy import units
from desitarget.geomask import match
from desihiz.hizmerge_io import get_img_bands, get_cosmos2020_fn, match_coord
from desihiz.hsc_griz import recalibrate_c20zphot
from desihiz.specphot_utils import get_filts, get_smooth
import matplotlib
from matplotlib.path import Path
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from desiutil.log import get_logger

log = get_logger()

# isbug = False # use the non-bugged photometry

allowed_selections = [
    "djs",
    "v20231208",
]

suprime_bands = get_img_bands("suprime")
suprime_cols = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(suprime_bands)]

# djs cuts (maybe slightly agressive on ramax and decmax)
# approx. area
def get_radecbox_area():

    ramin, ramax, decmin, decmax = 149.4, 150.8, 1.48, 2.90
    area = 1.8
    log.info(
        "(ramin, ramax, decmin, decmax, area) = ({}, {}, {}, {})".format(
            ramin, ramax, decmin, decmax, area
        )
    )

    return ramin, ramax, decmin, decmax, area


def get_tractor_band(band):

    if band in suprime_bands:

        return band.replace("I", "I_A_L")

    else:

        return band


def add_band_quants(d, band, fmin=0.01, depthmin=1.0, rands=False):

    assert band in ["G", "R", "I"] + suprime_bands
    tband = get_tractor_band(band)

    if band in ["G", "R", "I"]:

        bb = True
        fkey = "FORCED_FLUX_{}".format(tband)
        ivkey = "FORCED_FLUX_IVAR_{}".format(tband)
        depthkey = "FORCED_PSFDEPTH_{}".format(tband)

    else:

        bb = False
        fkey = "FLUX_{}".format(tband)
        ivkey = "FLUX_IVAR_{}".format(tband)
        depthkey = "PSFDEPTH_{}".format(tband)

    # depths (data + rands)
    depths = 22.5 - 2.5 * np.log10(5.0 / np.sqrt(d[depthkey]))
    depths[d[depthkey] <= depthmin] = 22.5 - 2.5 * np.log10(5.0 / np.sqrt(depthmin))
    d["DEPTH_{}".format(band)] = depths

    # mags, snrs (data only)
    if not rands:

        fs = d[fkey]
        ms = 22.5 - 2.5 * np.log10(fs)
        ms[fs <= fmin] = 22.5 - 2.5 * np.log10(fmin)

        if bb:

            mfibs = None

        else:

            fibfs = d["FIBERFLUX_{}".format(tband)]
            mfibs = 22.5 - 2.5 * np.log10(fibfs)
            mfibs[fibfs <= fmin] = 22.5 - 2.5 * np.log10(fmin)

        snrs = d[fkey] * np.sqrt(d[ivkey])
        d["MAG_{}".format(band)] = ms

        if mfibs is not None:

            d["FIBMAG_{}".format(band)] = mfibs

        d["SNR_{}".format(band)] = snrs

    return d


# mag offsets bugged->ok computed from std stars
# bugged = correct + offset
def get_bug2ok_mags():

    return {
        "I427": +0.70,
        "I464": +0.02,
        "I484": -0.05,
        "I505": -0.06,
        "I527": -0.26,
    }


def get_selection_selbands(selection):

    assert selection in allowed_selections

    if selection == "djs":

        return ["I464", "I484", "I505", "I527"]

    if selection == "v20231208":

        return ["I464", "I484", "I505", "I527"]


def get_selection_maglims(selection, isbug):

    assert selection in allowed_selections

    suprime_bands = get_img_bands("suprime")

    if selection in ["djs", "v20231208"]:

        maglims = {
            "I427": 24.75,
            "I464": 24.75,
            "I484": 24.75,
            "I505": 24.75,
            "I527": 25.25,
        }

        # if not bug: we move the cuts to th
        if not isbug:

            bug2ok_mags = get_bug2ok_mags()

            for band in suprime_bands:

                maglims[band] += bug2ok_mags[band]

    return maglims


# templates=True => add an ad hoc 0.3 mag offset..
# DJS selection:
# templates=True => add an ad hoc 0.3 mag offset..
def get_selection_keys_thresholds(selection, selband, templates=False):

    assert selection in ["djs", "v20231208"]
    assert selband in get_selection_selbands(selection)

    # djs
    if selection == "djs":

        if selband == "I464":

            colskeys = ["DJS_COL2"]
            thresholds = [0.60]

        if selband == "I484":

            colskeys = ["DJS_COL3", "DJS_COL3M2"]
            thresholds = [0.60, 0.60]

        if selband == "I505":

            colskeys = ["DJS_COL4", "DJS_COL4M3", "DJS_COL4M2"]
            thresholds = [0.50, 0.50, 0.50]

        if selband == "I527":

            colskeys = ["DJS_COL5", "DJS_COL5M4", "DJS_COL5M3", "DJS_COL5M2"]
            thresholds = [0.40, 0.45, 0.45, 0.45]

    # v20231208
    if selection == "v20231208":

        if selband == "I464":

            colskeys = ["DJS_COL2"]
            thresholds = [0.60]

        if selband == "I484":

            colskeys = ["DJS_COL3"]
            thresholds = [0.60]

        if selband == "I505":

            colskeys = ["DJS_COL4"]
            thresholds = [0.65]  # +0.15

        if selband == "I527":

            colskeys = ["DJS_COL5"]
            thresholds = [0.80]  # +0.4

    if templates:

        thresholds = [_ - 0.3 for _ in thresholds]

    return colskeys, thresholds


def add_v20231208_sels(d, isbug, mrmin=23.0, snrmin=8.0, maxsize=1.0):

    for band in get_selection_selbands("v20231208"):

        # selection
        keys, thresholds = get_selection_keys_thresholds("v20231208", band)
        d["SEL_{}".format(band)] = d["PARENT_{}".format(band)].copy()

        for key, threshold in zip(keys, thresholds):

            d["SEL_{}".format(band)] &= d[key] > threshold

        if band == "I464":

            d["SEL_{}".format(band)] &= d["RI_COL"] < -0.1 + 0.4 * d["DJS_COL2"]

        log.info(
            "isbug={}\tSEL_{}={}".format(
                isbug,
                band,
                d["SEL_{}".format(band)].sum(),
            )
        )

    return d


def add_vi_efftime(d, sfn, isbug):

    d.meta["SP_FN"] = sfn

    if isbug:

        pextname = "PHOTINFO"

    else:

        pextname = "PHOTV2INFO"

    assert pextname in ["PHOTINFO", "PHOTV2INFO"]
    d["UNQID"] = ["{}-{}".format(b, o) for (b, o) in zip(d["BRICKNAME"], d["OBJID"])]
    s = Table.read(sfn, "SPECINFO")
    p = Table.read(sfn, pextname)
    p["UNQID"] = ["{}-{}".format(b, o) for (b, o) in zip(p["BRICKNAME"], p["OBJID"])]

    # handle for PHOTV2INFO non-matched rows...
    sel = p["RA"] != 0
    s, p = s[sel], p[sel]
    assert np.unique(d["UNQID"]).size == len(d)
    assert np.unique(p["UNQID"]).size == len(p)
    ii_d, ii_p = match(d["UNQID"], p["UNQID"])

    for key in [
        "TARGETID",
        "VI",
        "VI_QUALITY",
        "VI_Z",
        "EFFTIME_SPEC",
    ]:

        d[key] = np.zeros_like(s[key], shape=(len(d),))
        d[key][ii_d] = s[key][ii_p]

    return d


def add_c20zphot(d, search_radius=1.0, recalib=True):

    fn = get_cosmos2020_fn("cosmos_yr1")
    idkey, rakey, deckey, zkey = "ID", "ALPHA_J2000", "DELTA_J2000", "lp_zBEST"
    d.meta["ZPHFN"] = fn

    z = Table(fitsio.read(fn, columns=[idkey, rakey, deckey, zkey]))

    if recalib:

        z["ZPHOT"] = recalibrate_c20zphot(z["lp_zBEST"])
        d.meta["ZPHRECAL"] = recalib

    iid, iiz, _, _, _ = match_coord(
        d["RA"],
        d["DEC"],
        z[rakey],
        z[deckey],
        search_radius=search_radius,
    )
    d["ISZPHOT"] = np.zeros(len(d), dtype=bool)
    d["ZPHOT_ID"] = np.zeros_like(z[idkey], shape=(len(d),))
    d["ZPHOT"] = np.zeros(len(d))
    d["ISZPHOT"][iid] = True
    d["ZPHOT_ID"][iid] = z[idkey][iiz]
    d["ZPHOT"][iid] = z[zkey][iiz]

    return d


def get_ccdnames():

    return np.array(["det{}".format(_) for _ in range(10)])


def get_suprime_dir(isbug):

    if isbug:

        return os.path.join(os.getenv("DESI_ROOT"), "users", "dstn", "suprime")

    else:

        return os.path.join(os.getenv("DESI_ROOT"), "users", "dstn", "suprime-rerun-2")


def get_suprime_annotated_ccdfn(isbug):

    suprime_dir = get_suprime_dir(isbug)

    return os.path.join(suprime_dir, "ccds-annotated-suprime-IA.fits")


def get_suprime_brn_ccdfn(brn, isbug):

    suprime_dir = get_suprime_dir(isbug)

    return os.path.join(
        suprime_dir,
        "catalogs",
        "coadd",
        brn[:3],
        brn,
        "legacysurvey-{}-ccds.fits".format(brn),
    )


def get_brn_band_nccd(brn_d, band, isbug, brn, rakey="ra", deckey="dec"):

    # ccds for the brick
    brn_ccdfn = get_suprime_brn_ccdfn(brn, isbug)
    brn_ccd_d = Table.read(brn_ccdfn)
    sel = brn_ccd_d["filter"] == band.replace("I", "I-A-L")
    brn_ccd_d = brn_ccd_d[sel]
    brn_unqids = np.array(
        [
            "{}-{}".format(e, c)
            for e, c in zip(brn_ccd_d["expnum"], brn_ccd_d["ccdname"])
        ]
    )

    # all ccds (to grab ra0, ra1, etc)
    annotated_ccdfn = get_suprime_annotated_ccdfn(isbug)
    ccd_d = Table.read(annotated_ccdfn)
    sel = ccd_d["filter"] == band.replace("I", "I-A-L")
    ccd_d = ccd_d[sel]
    unqids = np.array(
        ["{}-{}".format(e, c) for e, c in zip(ccd_d["expnum"], ccd_d["ccdname"])]
    )

    # cut on ccds used in the brick
    sel = np.in1d(unqids, brn_unqids)
    ccd_d = ccd_d[sel]

    # test all ccds
    ccdnames = get_ccdnames()
    nccds = np.zeros((len(brn_d), len(ccdnames)), dtype=int)

    for i in range(len(ccdnames)):

        jj = np.where(ccd_d["ccdname"] == ccdnames[i])[0]

        for j in jj:

            # TODO handle good_region..
            # but ~ok to ignore, at worse it s ~50 pixels rejected
            ra0, ra1, ra2, ra3 = (
                ccd_d["ra0"][j],
                ccd_d["ra1"][j],
                ccd_d["ra2"][j],
                ccd_d["ra3"][j],
            )
            dec0, dec1, dec2, dec3 = (
                ccd_d["dec0"][j],
                ccd_d["dec1"][j],
                ccd_d["dec2"][j],
                ccd_d["dec3"][j],
            )
            ccd_ras = [ra0, ra1, ra2, ra3, ra0]
            ccd_decs = [dec0, dec1, dec2, dec3, dec0]
            p = Path([(ra, dec) for ra, dec in zip(ccd_ras, ccd_decs)])
            isccd_j = p.contains_points(
                [(ra, dec) for ra, dec in zip(brn_d[rakey], brn_d[deckey])]
            )
            nccds[isccd_j, i] += 1

    return nccds


def get_band_nccd(d, band, isbug, brnkey="BRICKNAME", rakey="RA", deckey="DEC"):

    ccdnames = get_ccdnames()
    nccds = np.zeros((len(d), len(ccdnames)), dtype=float)

    log.info("# BAND BRICKNAME NOBJ")
    for brn in np.unique(d[brnkey]):

        sel = d[brnkey] == brn
        log.info("{}\t{}\t{}".format(band, brn, sel.sum()))
        nccds[sel] = get_brn_band_nccd(
            d[sel], band, isbug, brn, rakey=rakey, deckey=deckey
        )

    return nccds


def add_band_nccd(d, isbug):

    for band in suprime_bands:

        d["NCCD_{}".format(band)] = get_band_nccd(d, band, isbug)

    return d


# TODO: handle nodet0 here; so far, density is wrong for that case...
# rounding=10 -> rounding to 10/deg2..
# margin_check [deg] -> to make a loose sanity check
def get_density(d, selbands, rcut=None, rounding=None, margin_check=0.025):

    ramin, ramax, decmin, decmax, area = get_radecbox_area()

    sel = np.zeros(len(d), dtype=bool)
    for selband in selbands:
        sel |= d["SEL_{}".format(selband)]

    # loose sanity check
    assert np.abs(d["RA"][sel] - ramin).min() < margin_check
    assert np.abs(d["RA"][sel] - ramax).min() < margin_check
    assert np.abs(d["DEC"][sel] - decmin).min() < margin_check
    assert np.abs(d["DEC"][sel] - decmax).min() < margin_check

    if rcut is not None:

        sel &= d["MAG_R"] < rcut

    density = sel.sum() / area

    if rounding is not None:

        density = rounding * np.round(density / rounding, 0)

    log.info(
        "selbands={}, area={} deg2 => density = {:.0f} deg2".format(
            selbands, area, density
        )
    )

    return density


def plot_selection_spectra(
    outpng,
    selection,
    d,
    band,
    zmin,
    zmax,
    mergefn=None,
    viqualcut=2.0,
    efftimemin_hrs=3.5,
    efftimemax_hrs=5.0,
    **kwargs
):

    # targetids
    sel = (d["SEL_{}".format(band)]) & (d["VI_QUALITY"] >= viqualcut)

    if efftimemin_hrs is not None:

        sel &= d["EFFTIME_SPEC"] / 3600.0 > efftimemin_hrs

    if efftimemax_hrs is not None:

        sel &= d["EFFTIME_SPEC"] / 3600.0 < efftimemax_hrs
    tids = d["TARGETID"][sel]

    s = fitsio.read(mergefn, "SPECINFO", columns=["TARGETID", "VI_Z", "VI_QUALITY"])
    sel = np.in1d(s["TARGETID"], tids)
    assert sel.sum() == len(tids)
    assert np.all(s["VI_QUALITY"][sel] >= viqualcut)
    sel &= (s["VI_Z"] > zmin) & (s["VI_Z"] < zmax)
    ii = np.where(sel)[0]

    s = s[ii]

    iislice = tuple(slice(i) for i in ii)
    h = fits.open(mergefn)
    ws = h["BRZ_WAVE"].data
    fs = h["BRZ_FLUX"].data[ii]
    ivs = h["BRZ_IVAR"].data[ii]

    fig, ax = plt.subplots(figsize=(15, 5))

    for i in range(ii.size):

        wis = ws / (1 + s["VI_Z"][i])
        smfs, _ = get_smooth(fs[i], ivs[i], 3)
        ax.plot(wis, smfs, **kwargs)

    ax.grid()
    ax.set_title(
        "{}-band {} selection : VI_QUALITY >={} and {} < VI_Z < {} ({} spectra)".format(
            band, selection, viqualcut, zmin, zmax, ii.size
        )
    )
    ax.set_xlabel("Rest-frame wavelength [A]")
    ax.set_ylabel("Flux [U.A.]")
    ax.set_xlim(1000, 1400)
    ax.set_ylim(-0.5, 2)

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def plot_selection_diagnosis(
    outpng,
    d,
    selection,
    isbug,
    selband,
    zgoalmin=2.2,
    zgoalmax=3.6,
    nodet0=False,
    rcut=None,
    viqualcut=2.0,
    efftimemin_hrs=3.5,
    efftimemax_hrs=5.0,
):

    assert selection in allowed_selections

    # selband settings
    mag_key = "MAG_{}".format(selband)
    maglim = get_selection_maglims(selection, isbug)[selband]
    keys, thresholds = get_selection_keys_thresholds(selection, selband)
    col_key, threshold = keys[0], thresholds[0]
    tmplist = d.meta[col_key].replace("MAG_", "").split()
    blue, red = tmplist[0], tmplist[2]
    hist_bands = [blue, red, "R"]

    # density
    rounding = 10

    # title
    title = "SUPRIME Medium-bands {} selections".format(selection)

    mags, cols = d[mag_key], d[col_key]

    parent_sel = d["PARENT_{}".format(selband)].copy()
    sel = d["SEL_{}".format(selband)].copy()

    # det0
    if nodet0:

        det0_sel = np.ones(len(d), dtype=bool)

        for band in suprime_bands:

            if band in used_bands:

                det0_sel &= d["NCCD_{}".format(band)][:, 0] == 0

        title = "{} (no DET0)".format(title)
        parent_sel &= det0_sel
        sel &= det0_sel

    vi_sel = (sel) & (d["VI"])

    if efftimemin_hrs is not None:

        vi_sel &= d["EFFTIME_SPEC"] / 3600.0 > efftimemin_hrs

    if efftimemax_hrs is not None:

        vi_sel &= d["EFFTIME_SPEC"] / 3600.0 < efftimemax_hrs

    viok_sel = (vi_sel) & (d["VI_QUALITY"] >= viqualcut)
    viokgoal_sel = (viok_sel) & (d["VI_Z"] > zgoalmin) & (d["VI_Z"] < zgoalmax)
    viokcont_sel = (viok_sel) & (~viokgoal_sel)
    zph_sel = (sel) & (d["ISZPHOT"]) & (d["ZPHOT"] > -1)
    zphgoal_sel = (zph_sel) & (d["ZPHOT"] > zgoalmin) & (d["ZPHOT"] < zgoalmax)

    if rcut is not None:

        rcut_sel = (sel) & (d["MAG_R"] < rcut)
        rcutviok_sel = (viok_sel) & (rcut_sel)
        rcutzph_sel = (zph_sel) & (rcut_sel)

    # density
    # TODO: handle nodet0 here; so far, density is wrong for that case...
    # rounding to 10/deg2..
    density = get_density(d, [selband], rounding=rounding)
    log.info(
        "selection={}, selband={}, Nsel={}, density={:.0f}/deg2".format(
            selection, selband, sel.sum(), density
        )
    )

    if rcut is not None:

        rcut_density = get_density(d, [selband], rcut=rcut, rounding=rounding)

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(len(hist_bands), 3, wspace=0.3, hspace=0.1)

    # mag hist
    bins = np.arange(22, 27.2, 0.2)

    for ix, band in enumerate(hist_bands):

        ax = fig.add_subplot(gs[ix, 0])
        _ = ax.hist(
            d["MAG_{}".format(band)][sel],
            bins=bins,
            density=True,
            histtype="stepfilled",
            color="orange",
            alpha=0.5,
            label="{}-band".format(band),
        )

        if rcut is not None:

            _ = ax.hist(
                d["MAG_{}".format(band)][rcut_sel],
                bins=bins,
                density=True,
                histtype="step",
                color="g",
                alpha=0.5,
            )

        if band == selband:

            ax.axvline(maglim, color="orange", ls="--", label=str(maglim))

        if (band == "R") & (rcut is not None):

            ax.axvline(rcut, color="g", ls="--", label=str(rcut))

        if ix == len(hist_bands) - 1:

            ax.set_xlabel("Magnitude [AB]")

        else:

            ax.set_xticklabels([])

        if ix == 1:

            ax.set_ylabel("Normalized counts")

        ax.set_xlim(22, 27)
        ax.set_ylim(0, 1.99)
        ax.grid()
        ax.legend(loc=2)

    # col-mag
    ax = fig.add_subplot(gs[:, 1])
    xlim = np.array([22.0, 25.5])
    ylim = np.array([-1, 4])

    mysels = [parent_sel, sel, (vi_sel) & (~viokgoal_sel), viokgoal_sel]
    mylabs = [
        None,
        "Selection (~{:.0f}/deg2)".format(density),
        "Observed, non tracer ({})".format(((vi_sel) & (~viokgoal_sel)).sum()),
        "Observed, {}<z<{} tracer ({})".format(zgoalmin, zgoalmax, viokgoal_sel.sum()),
    ]
    markers = ["o", "o", "o", "o"]
    facecolors = ["k", "orange", "none", "none"]
    edgecolors = ["k", "orange", "r", "b"]
    lws = [None, None, 1.0, 0.5]
    ss = [1, 10, 10, 10]
    alphas = [0.1, 0.5, 1.0, 0.5]
    zorders = [0, 1, 3, 2]

    if rcut is not None:

        mysels.insert(2, rcut_sel)
        mylabs.insert(2, "mag_r<{} selection (~{:.0f}/deg2)".format(rcut, rcut_density))
        markers.insert(2, "x")
        facecolors.insert(2, "g")
        edgecolors.insert(2, "g")
        lws.insert(2, 1)
        ss.insert(2, 25)
        alphas.insert(2, 0.5)
        zorders.insert(2, 3)

    for mysel, mylab, marker, facecolor, edgecolor, lw, s, alpha, zorder in zip(
        mysels, mylabs, markers, facecolors, edgecolors, lws, ss, alphas, zorders
    ):

        ax.scatter(
            mags[mysel],
            cols[mysel],
            marker=marker,
            facecolor=facecolor,
            edgecolor=edgecolor,
            lw=lw,
            s=s,
            alpha=alpha,
            label=mylab,
        )

    ax.axvline(maglim, color="k", ls="--")
    _, col_mins = get_selection_keys_thresholds(selection, selband, templates=False)
    col_min = col_mins[0]
    ax.axhline(col_min, color="k", ls="--")

    ax.set_title(title)
    ax.set_xlabel(selband)
    ylabel = d.meta[col_key].replace("MAG_", "")
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid()
    ax.legend(loc=2)

    # n(z)
    ax = fig.add_subplot(gs[:, 2])
    bins = np.arange(0, 4.05, 0.05)
    # desi zspec
    label = "DESI zspec ({:.0f}% observed)".format(100.0 * vi_sel.sum() / sel.sum())
    weights = np.zeros(len(d), dtype=float) + density / viok_sel.sum()

    _ = ax.hist(
        d["VI_Z"][viok_sel],
        bins=bins,
        density=False,
        weights=weights[viok_sel],
        histtype="stepfilled",
        color="k",
        alpha=0.75,
        zorder=0,
        label=label,
    )

    txt = "{}<zspec<{} tracers = {}/{} = {:.0f}%".format(
        zgoalmin,
        zgoalmax,
        viokgoal_sel.sum(),
        vi_sel.sum(),
        100 * viokgoal_sel.sum() / vi_sel.sum(),
    )
    ax.text(0.05, 0.8, txt, color="k", transform=ax.transAxes)

    # cosmos2020 zphot
    label = "zphot"
    weights = np.zeros(len(d), dtype=float) + density / zph_sel.sum()
    _ = ax.hist(
        d["ZPHOT"][zph_sel],
        bins=bins,
        density=False,
        weights=weights[zph_sel],
        histtype="step",
        color="r",
        lw=0.5,
        alpha=1,
        zorder=1,
        label=label,
    )

    txt = "{}<zphot<{} tracers = {}/{} = {:.0f}%".format(
        zgoalmin,
        zgoalmax,
        zphgoal_sel.sum(),
        zph_sel.sum(),
        100 * zphgoal_sel.sum() / zph_sel.sum(),
    )
    ax.text(0.05, 0.75, txt, color="r", transform=ax.transAxes)

    # if rcut is not None:
    #    _ = ax.hist(d["VI_Z"][rcutviok_sel], bins=bins, density=True, histtype="stepfilled", color="g", alpha=0.5, zorder=0)
    #    _ = ax.hist(d["COSMOS2020_ZPHOT"][rcutc20_sel], bins=bins, density=True, histtype="step", color="g", lw=0.5, alpha=1, zorder=1)
    # ax.set_title(title)

    ax.set_xlabel("Redshift (z)")
    ax.set_ylabel("N / deg2 / (dz={:.3f})".format(bins[1] - bins[0]))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 80)
    ax.grid()
    ax.set_axisbelow(True)
    ax.legend(loc=2)

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def plot_selection_nz(
    outpng,
    selection,
    zkey,
    d,
    nodet0=False,
    rcut=None,
    viqualcut=2.0,
    efftimemin_hrs=3.5,
    efftimemax_hrs=5.0,
):

    assert zkey in ["VI_Z", "ZPHOT"]

    filts = get_filts()

    xlim = (2.0, 4.0)
    zbins = np.arange(xlim[0], xlim[1] + 0.025, 0.025)

    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 1, hspace=0.05, height_ratios=[1, 0.2])
    ax = fig.add_subplot(gs[0])
    axf = fig.add_subplot(gs[1])

    for selband, col in zip(suprime_bands, suprime_cols):

        # filter
        filt = filts["SUPRIME_{}".format(selband)]
        ws, rs = filt["WAVE"], filt["RESP"]
        axf.fill_between(
            ws / 1215.67 - 1, rs * 0, rs, color=col, alpha=0.15
        )  # , label="{} filter".format(selband))

        if selband not in get_selection_selbands(selection):

            continue

        # density
        density = get_density(d, [selband], rounding=10)

        if rcut is not None:

            rcut_density = get_density(d, [selband], rcut=rcut, rounding=10)

        # label
        lab = "({}) selection (targets ~ {:.0f}/deg2)".format(selband, density)

        # selection
        sel = d["SEL_{}".format(selband)].copy()

        if zkey == "VI_Z":

            sel &= (d["VI"]) & (d["VI_QUALITY"] >= viqualcut)

            if efftimemin_hrs is not None:

                sel &= d["EFFTIME_SPEC"] / 3600.0 > efftimemin_hrs

            if efftimemax_hrs is not None:

                sel &= d["EFFTIME_SPEC"] / 3600.0 < efftimemax_hrs

        if zkey == "ZPHOT":

            sel &= d["ISZPHOT"]

        if nodet0:

            for band in used_suprime_bands:

                sel &= d["NCCD_{}".format(band)][:, 0] == 0

        if rcut is not None:

            rcut_sel = (sel) & (d["MAG_R"] < rcut)

        weights = np.zeros(len(d), dtype=float) + density / sel.sum()
        _ = ax.hist(
            d[zkey][sel],
            bins=zbins,
            density=False,
            weights=weights[sel],
            histtype="stepfilled",
            color=col,
            alpha=0.25,
            zorder=1,
            label=lab,
        )

        if rcut is not None:

            weights = np.zeros(len(d), dtype=float) + rcut_density / rcut_sel.sum()
            _ = ax.hist(
                d[zkey][rcut_sel],
                bins=zbins,
                density=False,
                weights=weights[rcut_sel],
                histtype="step",
                color=col,
                lw=1,
                zorder=1,
            )

    ax.set_title("SUPRIME Medium-bands {} selections".format(selection))
    ax.set_xticklabels([])
    ax.set_ylabel("N / deg2 / (dz={:.3f})".format(zbins[1] - zbins[0]))
    ax.set_xlim(xlim)
    ax.set_ylim(0, 40)
    ax.grid()
    ax.legend(loc=2)

    axf.set_xlim(xlim)
    axf.set_xlabel("Redshift (z)")
    axf.set_yticklabels([])
    axf.grid()

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()
