#!/usr/bin/env python


import os
from glob import glob
import fitsio
import numpy as np

from astropy.io import fits
from astropy.table import Table, vstack

from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib

from desihizmerge.hizmerge_io import get_img_bands, get_cosmos2020_fn, match_coord

from desiutil.log import get_logger

log = get_logger()


def get_suprime_dir():

    return os.path.join(os.getenv("DESI_ROOT"), "users", "dstn", "suprime")


def get_get_suprime_annotated_ccdfn():

    suprime_dir = get_suprime_dir()
    return os.path.join(suprime_dir, "ccds-annotated-suprime-IA.fits")


def get_suprime_brn_ccdfn(brn):

    suprime_dir = get_suprime_dir()
    return os.path.join(
        suprime_dir,
        "catalogs",
        "coadd",
        brn[:3],
        brn,
        "legacysurvey-{}-ccds.fits".format(brn),
    )


def get_ccdnames():

    return np.array(["det{}".format(_) for _ in range(10)])


def get_tractor_band(band, uppercase=True):
    if uppercase:
        return band.replace("I", "I_A_L")
    else:
        return band.replace("I", "i_a_l")


def get_c20_band(band):
    if band in ["I484", "I527"]:
        return band.replace("I", "SC_IA")
    else:
        return band.replace("I", "SC_IB")


def read_photspecfn(photspecfn):

    bands = get_img_bands("suprime")
    d = Table(fitsio.read(photspecfn, "CUSTOM"))
    t = Table(fitsio.read(photspecfn, "PHOTINFO"))
    for key in t.colnames:
        newkey = key.upper()
        for band in bands:
            newkey = newkey.replace("I_A_L{}".format(band[1:]), band)
        if newkey not in d.colnames:
            d[newkey] = t[key]
    return d


# TODO: first match to e.g. ls-dr9.1.1., and use FLUX_G
def get_clean_for_offsets(d, band, fmin=1, fmax=1000):

    sel = d["BRICK_PRIMARY"].copy()
    sel &= d["TYPE"] == "PSF"
    sel &= d["SPECTRO_FIBERTOTFLUX_{}".format(band)] > fmin
    sel &= d["SPECTRO_FIBERTOTFLUX_{}".format(band)] < fmax
    sel &= d["SPECTRO_FIBERTOTFLUX_{}".format(band)] > (1000 / (12.15 * d["TSNR2_LRG"]))
    return sel


def get_ratios(d, band, fkey, offsets=None):

    fratios = d["{}_{}".format(fkey, band)] / d["SPECTRO_FIBERTOTFLUX_{}".format(band)]

    if offsets is not None:
        fratios /= offsets

    return fratios


# gausspsfdepth: 5-sigma PSF detection depth in AB mag
# depth_mag = -2.5 * log10(5 / sqrt(ivar))
def ivar2mag(ivars):
    return 22.5 - 2.5 * np.log10(5 / np.sqrt(ivars))


def mag2ivar(mags):
    # return 10 ** (+0.4 * 2 * (mags - 22.5)) / 5.
    return (5 * 10 ** (+0.4 * (mags - 22.5))) ** 2


# depths are (5sigma) mags
# the ivar of an ivar-weighted mean is the sum of the ivars
# https://en.wikipedia.org/wiki/Inverse-variance_weighting
# so we first convert to ivar and sum 1/ivar
# then convert back to mag
#
# brn_d : d cut on a brick
def get_brn_perccd_gausspsfdepths(
    brn_d, band, brn, rakey="TARGET_RA", deckey="TARGET_DEC"
):

    # ccds for the brick
    brn_ccdfn = get_suprime_brn_ccdfn(brn)
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
    annotated_ccdfn = get_get_suprime_annotated_ccdfn()
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
    nccd = len(ccdnames)
    depths = np.zeros((len(brn_d), nccd), dtype=float)

    for i in range(nccd):

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

            depths[isccd_j, i] += mag2ivar(ccd_d["gausspsfdepth"][j])

        sel = depths[:, i] != 0
        depths[sel, i] = ivar2mag(depths[sel, i])

    return depths


def get_perccd_gausspsfdepths(
    d, band, brnkey="BRICKNAME", rakey="TARGET_RA", deckey="TARGET_DEC"
):

    ccdnames = get_ccdnames()
    depths = np.zeros((len(d), len(ccdnames)), dtype=float)

    for brn in np.unique(d[brnkey]):

        sel = d[brnkey] == brn
        depths[sel] = get_brn_perccd_gausspsfdepths(
            d[sel], band, brn, rakey=rakey, deckey=deckey
        )

    return depths


def add_offsets_keys(d, band, perccd_gausspsfdepths, perobj_offsets, clean_sel=None):

    nccd = len(get_ccdnames())

    # used in the process?
    if clean_sel is not None:
        key = "ISCLEAN_FOR_OFFSET_{}".format(band)
        assert key not in d.colnames
        d[key] = clean_sel

    # per-ccd gausspsfdepth
    key = "PERCCD_GAUSSPSFDEPTH_{}".format(band)
    assert key not in d.colnames
    d[key] = perccd_gausspsfdepths

    # total gausspsfdepth
    totdepths = np.zeros(len(d))
    for i in range(nccd):
        totdepths += mag2ivar(d["PERCCD_GAUSSPSFDEPTH_{}".format(band)][:, i])
    sel = totdepths != 0
    totdepths[sel] = ivar2mag(totdepths[sel])
    key = "ALLCCD_GAUSSPSFDEPTH_{}".format(band)
    assert key not in d.colnames
    d[key] = totdepths

    # per-object estimated offset
    key = "OFFSET_{}".format(band)
    assert key not in d.colnames
    d[key] = perobj_offsets

    return d


def get_band_offsets(d, band, fkey, fmin=1, fmax=1000):

    ccdnames = get_ccdnames()
    nccd = len(ccdnames)

    perccd_gausspsfdepths = get_perccd_gausspsfdepths(d, band)
    assert perccd_gausspsfdepths.shape[1] == nccd

    iv_ccds = mag2ivar(perccd_gausspsfdepths)
    phot_fs = d["{}_{}".format(fkey, band)]
    spec_fs = d["SPECTRO_FIBERTOTFLUX_{}".format(band)]

    # test flux ratios within 0, 3
    vals = np.arange(0, 3, 0.01).round(2)

    # we initialize each ccd with the median offset for a clean sample
    clean_sel = get_clean_for_offsets(d, band, fmin, fmax)
    med_offset = np.nanmedian((spec_fs / phot_fs)[clean_sel])
    ccd_offsets = med_offset + np.zeros(nccd)
    log.info("initialize with ccd_offsets = {} for all ccds".format(med_offset))

    # 10 steps should be sufficient to converge...
    # in each loop, we iterate over the ccds one by one... simpler to code..
    log.info("# BAND ITERATION {}".format("\t".join(ccdnames)))

    for i_iter in range(10):

        prev_ccd_offsets = ccd_offsets.copy()

        for ccdnum in range(nccd):

            ress = -99 + 0.0 * vals

            for i in range(len(vals)):

                ccd_offsets[ccdnum] = vals[i]

                # per-obj. offset
                perobj_offsets = (ccd_offsets * iv_ccds).sum(axis=1) / iv_ccds.sum(
                    axis=1
                )

                # corrected phot_fs
                corr_fs = spec_fs * perobj_offsets

                # quantity to minimize
                xs = np.abs(corr_fs / phot_fs - 1)

                # taking the median for the clean sample
                ress[i] = np.nanmedian(xs[clean_sel])

            ccd_offsets[ccdnum] = vals[ress.argmin()]

        log.info("{}\t{}\t{}".format(band, i_iter, "\t".join(ccd_offsets.astype(str))))

        if np.all(ccd_offsets == prev_ccd_offsets):

            log.info(
                "reached convergence after {} iterations; stop here".format(i_iter + 1)
            )
            break

        prev_ccd_offsets = ccd_offsets.copy()

    # add few convenience columns to d
    perobj_offsets = (ccd_offsets * iv_ccds).sum(axis=1) / iv_ccds.sum(axis=1)
    d = add_offsets_keys(
        d, band, perccd_gausspsfdepths, perobj_offsets, clean_sel=clean_sel
    )

    # now format the per-ccd offset into a Table()
    d_band_offsets = Table()
    d_band_offsets["CCDNAME"] = ccdnames
    d_band_offsets["FILTER"] = band
    d_band_offsets["OFFSET"] = ccd_offsets
    d_band_offsets = d_band_offsets["FILTER", "CCDNAME", "OFFSET"]

    return d, d_band_offsets


def get_offsets(d, fkey, fmin=1, fmax=1000):

    bands = get_img_bands("suprime")
    all_d_band_offsets = []

    for band in bands:

        d, d_band_offsets = get_band_offsets(d, band, fkey, fmin=fmin, fmax=fmax)
        all_d_band_offsets.append(d_band_offsets)

    d_offsets = vstack(all_d_band_offsets)

    return d, d_offsets


def get_and_add_offsets_quants(
    offsetsfn,
    d,
    fmin=1,
    fmax=1000,
    brnkey="BRICKNAME",
    rakey="TARGET_RA",
    deckey="TARGET_DEC",
):

    bands = get_img_bands("suprime")
    ccdnames = get_ccdnames()

    all_ccd_offsets = Table.read(offsetsfn)

    for band in bands:

        # estimated per-ccd offsets
        ccd_offsets = []
        for ccdname in ccdnames:
            sel = (all_ccd_offsets["FILTER"] == band) & (
                all_ccd_offsets["CCDNAME"] == ccdname
            )
            ccd_offset = all_ccd_offsets["OFFSET"][sel][0]
            ccd_offsets.append(ccd_offset)
        ccd_offsets = np.array(ccd_offsets)

        # per-ccd quantities
        perccd_gausspsfdepths = get_perccd_gausspsfdepths(
            d, band, brnkey=brnkey, rakey=rakey, deckey=deckey
        )
        iv_ccds = mag2ivar(perccd_gausspsfdepths)

        # add offsets columns to d
        perobj_offsets = (ccd_offsets * iv_ccds).sum(axis=1) / iv_ccds.sum(axis=1)
        d = add_offsets_keys(d, band, perccd_gausspsfdepths, perobj_offsets)

    return d


def apply_offsets(
    tractorfn,
    offsetsfn,
    fkeys=[
        "flux",
        "fiberflux",
        "fibertotflux",
        "apflux",
        "apflux_resid",
        "apflux_blobresid",
    ],
):

    bands = get_img_bands("suprime")

    d = Table(fitsio.read(tractorfn))

    # get offsets quantities
    d = get_and_add_offsets_quants(
        offsetsfn, d, brnkey="brickname", rakey="ra", deckey="dec"
    )

    # correct for the offsets
    for band in bands:

        for fkey in fkeys:

            tband = band.replace("I", "i_a_l")
            key = "{}_{}".format(fkey, tband)
            if len(d[key].shape) == 2:
                for i in range(d[key].shape[1]):
                    d[key][:, i] /= d["OFFSET_{}".format(band)]
            else:
                d[key] /= d["OFFSET_{}".format(band)]

    d.meta["MODIFKEY"] = ",".join(fkeys)

    return d


def plot_suprime_ccd(ax, band, ccdname, **kwargs):

    annotated_ccdfn = get_get_suprime_annotated_ccdfn()
    ccd_d = Table.read(annotated_ccdfn)

    sel = ccd_d["filter"] == band.replace("I", "I-A-L")
    sel &= ccd_d["ccdname"] == ccdname
    ii = np.where(sel)[0]

    for i in ii:

        ras = [
            ccd_d["ra0"][i],
            ccd_d["ra1"][i],
            ccd_d["ra2"][i],
            ccd_d["ra3"][i],
            ccd_d["ra0"][i],
        ]
        decs = [
            ccd_d["dec0"][i],
            ccd_d["dec1"][i],
            ccd_d["dec2"][i],
            ccd_d["dec3"][i],
            ccd_d["dec0"][i],
        ]
        ax.plot(ras, decs, **kwargs)


# name : orig, offsets, corrected
def plot_indiv_band(
    ax,
    axh,
    d,
    band,
    name,
    ccdname=None,
    sample_label=None,
    vmin=0.8,
    vmax=1.2,
    alpha=0.75,
):

    names = ["ORIG", "OFFSETS", "CORR"]
    assert name in names

    clean_sel = d["CLEAN_{}".format(band)]

    true_fs = d["TRUE_FLUX_{}".format(band)].copy()
    orig_fs = d["ORIG_FLUX_{}".format(band)].copy()
    corr_fs = d["CORR_FLUX_{}".format(band)].copy()
    offsets = orig_fs / corr_fs

    ratios = {
        "ORIG": orig_fs / true_fs,
        "OFFSETS": offsets,
        "CORR": corr_fs / true_fs,
    }

    # median values
    medians = {_: np.nanmedian(ratios[_][clean_sel]).round(2) for _ in names}

    # divide by median values
    for _ in names:
        ratios[_] /= medians[_]

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # cmap = matplotlib.cm.viridis
    cmap = matplotlib.cm.coolwarm

    # offsets: plot all objects
    if name == "OFFSETS":

        title = "{}\nAll objects ({})\ncase = {} (median={})".format(
            band, len(d), name, medians[name]
        )
        sc = ax.scatter(
            d["RA"],
            d["DEC"],
            c=ratios["OFFSETS"],
            s=1,
            alpha=alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            zorder=0,
            rasterized=True,
        )
        clabel = "Estimated OFFSETS / median"

    # orig / corrected: plot clean selection
    else:

        if clean_sel.sum() < 5000:
            s = 10
        elif clean_sel.sum() < 10000:
            s = 5
        else:
            s = 1

        if sample_label is None:
            sample_label2 = "Some sample"
        else:
            sample_label2 = sample_label
        title = "{}\n{} ({})\ncase = {} (median = {})".format(
            band,
            sample_label2,
            clean_sel.sum(),
            name,
            medians[name],
        )
        sc = ax.scatter(
            d["RA"][clean_sel],
            d["DEC"][clean_sel],
            c=ratios[name][clean_sel],
            s=s,
            alpha=alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            zorder=0,
            rasterized=True,
        )
        clabel = "{}/TRUE flux ratio / median".format(name)

    cbar = plt.colorbar(mappable=sc, ax=ax, extend="both")
    cbar.mappable.set_clim([vmin, vmax])
    cbar.set_label(clabel)

    # suprime ccds
    if ccdname is not None:
        color, lw, ls, alpha, zorder = "k", 1, "-", 1, 1
        if ccdname == "all":
            ccdnames = get_ccdnames()
            lw = 0.2
            ax.plot(
                np.nan,
                np.nan,
                color=color,
                lw=lw,
                ls=ls,
                alpha=alpha,
                zorder=zorder,
                label="All CCDs",
            )
        else:
            ccdnames = [ccdname]
            ax.plot(
                np.nan,
                np.nan,
                color=color,
                lw=lw,
                ls=ls,
                alpha=alpha,
                zorder=zorder,
                label="{} CCD".format(ccdname),
            )
        for ccdname in ccdnames:
            plot_suprime_ccd(
                ax,
                band,
                ccdname,
                color=color,
                lw=lw,
                ls=ls,
                alpha=alpha,
                zorder=zorder,
            )

    ax.set_title(title)
    ax.set_xlabel("R.A [deg]")
    ax.set_ylabel("Dec. [deg]")
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlim(150.95, 149.3)
    ax.set_ylim(1.35, 3.05)
    if ccdname is not None:
        ax.legend(loc=2)

    # hist
    if axh is not None:
        histtypes = {"ORIG": "step", "CORR": "stepfilled"}
        colors = {"ORIG": "k", "CORR": "g"}
        alphas = {"ORIG": 1.0, "CORR": 0.5}
        zorders = {"ORIG": 1, "CORR": 0}
        bins = np.linspace(0.0, 2.0, 101)
        _ = axh.hist(
            ratios[name][clean_sel],
            bins=bins,
            density=True,
            histtype=histtypes[name],
            color=colors[name],
            alpha=alphas[name],
            zorder=zorders[name],
            label="Flux ratio = {}/TRUE (median={})".format(name, medians[name]),
        )
        # if corrected: also plot orig
        if name == "CORR":
            _ = axh.hist(
                ratios["ORIG"][clean_sel],
                bins=bins,
                density=True,
                histtype=histtypes["ORIG"],
                color=colors["ORIG"],
                alpha=alphas["ORIG"],
                zorder=zorders["ORIG"],
                label="Flux ratio = {}/TRUE (median={})".format(
                    "ORIG", medians["ORIG"]
                ),
            )

        axh.set_xlabel("Flux ratio / median")
        axh.set_ylabel("Norm. counts")
        axh.grid()
        axh.set_axisbelow(True)
        axh.set_xlim(bins[0], bins[-1])
        axh.set_ylim(0, 7)
        axh.legend(loc=2)


def plot_offsets_band(outpng, d, band, ccdname=None, sample_label=None):

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, wspace=0.25, hspace=0.2)

    for iy, name in enumerate(["ORIG", "OFFSETS", "CORR"]):

        ax = fig.add_subplot(gs[0, iy])

        if name == "OFFSETS":

            axh = None
            axlab = fig.add_subplot(gs[1, iy])
            axlab.axis("off")
            for ytxt, txt in zip(
                [0.8, 0.7, 0.6],
                [
                    "TRUE = {}".format(d.meta["TRUE"]),
                    "ORIG = {}".format(d.meta["ORIG"]),
                    "CORR = {} / OFFSETS".format(d.meta["ORIG"]),
                ],
            ):
                axlab.text(0.05, ytxt, txt, fontsize=15, transform=axlab.transAxes)

        else:

            axh = fig.add_subplot(gs[1, iy])
        plot_indiv_band(
            ax, axh, d, band, name, ccdname=ccdname, sample_label=sample_label
        )

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def plot_offsets(d, outpngroot, ccdnames=[None], sample_label=None):

    bands = get_img_bands("suprime")

    for band in bands:

        for ccdname in ccdnames:
            outpng = "{}-{}.png".format(outpngroot, band)
            if ccdname is not None:
                outpng = outpng.replace(".png", "-{}.png".format(ccdname))
            log.info("plot {}".format(outpng))
            plot_offsets_band(
                outpng, d, band, ccdname=ccdname, sample_label=sample_label
            )


# sample = star or laelbg
def get_d_for_plot_offsets(outphotspecfn, fkey, sample):

    assert sample in ["star", "laelbg"]

    if sample == "laelbg":

        fn = os.path.join(
            os.getenv("DESI_ROOT"),
            "survey",
            "fiberassign",
            "special",
            "tertiary",
            "0026",
            "tertiary-targets-0026.fits",
        )
        t = Table.read(fn)
        sel = (t["SUPRIME"]) | (t["LAE_SUBARU"])
        tids = t["TARGETID"][sel]

    bands = get_img_bands("suprime")
    d = Table(fitsio.read(outphotspecfn))

    myd = Table()
    myd["RA"], myd["DEC"] = d["RA"].copy(), d["DEC"].copy()

    for band in bands:

        if sample == "star":

            myd["CLEAN_{}".format(band)] = d[
                "ISCLEAN_FOR_OFFSET_{}".format(band)
            ].copy()

        if sample == "laelbg":

            myd["CLEAN_{}".format(band)] = np.in1d(d["TARGETID"], tids)

        myd["TRUE_FLUX_{}".format(band)] = d[
            "SPECTRO_FIBERTOTFLUX_{}".format(band)
        ].copy()
        myd["ORIG_FLUX_{}".format(band)] = d["{}_{}".format(fkey, band)].copy()
        myd["CORR_FLUX_{}".format(band)] = (
            d["{}_{}".format(fkey, band)] / d["OFFSET_{}".format(band)]
        )

    myd.meta["TRUE"] = "SPECTRO_FIBERTOTFLUX"
    myd.meta["ORIG"] = "TRACTOR {}".format(fkey)

    return myd


# m = 22.5 - 2.5 * log10(f_nmg)
#   = 23.9 - 2.5 * log10(f_uJy)
# f_nmg = 10 ** (-0.4 * (23.9 - 22.5)) * f_uJy
def flux_uJy2nmg(fs):
    return 10 ** (-0.4 * (23.9 - 22.5)) * fs


def get_match_cosmos2020(tractorfn, offsetsfn, fmin=1, fmax=1000):

    bands = get_img_bands("suprime")

    # tractor
    t = Table(fitsio.read(tractorfn))
    sel = t["brick_primary"]
    t = t[sel]

    # compute offsets
    t = apply_offsets(tractorfn, offsetsfn)

    # cosmos2020 ACS stars
    c20fn = get_cosmos2020_fn("cosmos_yr2")
    c20 = Table(fitsio.read(c20fn))

    # match
    iit, iic20, _, _, _ = match_coord(
        t["ra"], t["dec"], c20["ALPHA_J2000"], c20["DELTA_J2000"], search_radius=1.0
    )
    t, c20 = t[iit], c20[iic20]

    # store relevant columns in a Table
    d = Table()
    d["RA"], d["DEC"] = t["ra"], t["dec"]
    d["ACS_MU_CLASS"] = c20["ACS_MU_CLASS"]

    for band in bands:

        tband = get_tractor_band(band, uppercase=False)
        d["ORIG_FLUX_{}".format(band)] = (
            t["flux_{}".format(tband)] * t["OFFSET_{}".format(band)]
        )
        d["CORR_FLUX_{}".format(band)] = t["flux_{}".format(tband)]

        c20band = get_c20_band(band)
        d["TRUE_FLUX_{}".format(band)] = flux_uJy2nmg(c20["{}_FLUX".format(c20band)])

        clean_sel = d["ACS_MU_CLASS"] == 2
        clean_sel &= d["TRUE_FLUX_{}".format(band)] > fmin
        clean_sel &= d["TRUE_FLUX_{}".format(band)] < fmax
        d["CLEAN_{}".format(band)] = clean_sel

    d.meta["TRUE"] = "COSMOS2020 FLUX"
    d.meta["ORIG"] = "TRACTOR FLUX"

    return d
