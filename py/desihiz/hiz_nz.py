#!/usr/bin/env python

"""
    functions to generate an hopefully realistic n(z) from observations.
"""

import os
import numpy as np
from astropy.table import Table
from scipy.interpolate import splrep, BSpline
from sklearn.mixture import GaussianMixture
from desihiz.hizmerge_io import get_img_dir
from desihiz.suprime_analysis import get_selection_selbands
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from desiutil.log import get_logger


log = get_logger()


def get_zbins(zmin, zmax, dz, verbose=False):
    zbins = [zmin]

    while zbins[-1] < zmax:

        zbins.append(np.round(zbins[-1] + dz, 2))

    zbins = np.array(zbins)

    if verbose:

        log.info("zbins = {}".format(zbins))

    return zbins


def get_zpeceff_rawfns(img, field):

    # curves from LBGs in xmm
    # https://desisurvey.slack.com/archives/C0351RV8CBE/p1701373899584069
    if (img == "clauds") & (field == "xmmlss"):

        mydir = os.path.join(
            os.getenv("DESI_ROOT"), "users", "raichoor", "laelbg", "clauds", "analysis"
        )
        fns = {
            "120min": {
                "ZKEY": "photo_z",
                "EFFKEY": "eff",
                "FN": os.path.join(mydir, "efficiency_xmm_120min_zphot.ecsv"),
            },
            "240min": {
                "ZKEY": "photo_z",
                "EFFKEY": "eff",
                "FN": os.path.join(mydir, "efficiency_xmm_240min_zphot.ecsv"),
            },
        }

    return fns


def get_zspeceff_fn(img, field):

    imgdir = get_img_dir(img)
    fn = os.path.join(imgdir, "analysis", "{}-{}-zspec-eff.ecsv".format(img, field))
    return fn


# (clauds, xmm):
# - start from data file
# - 0.0 < z < 1.6 : set eff = 1 # not accurate, but we re not interested in that range
# - 1.6 < z < 2.1 : set eff = 0
# - 3.0 < z < 3.6 : set eff = value of highest z valid raw_eff
# - 3.6 < z < 4.0 : set eff = 0 # maybe not accurate, but no meaningful stat there..
def create_zspeceff(img, field, zbins):

    outfn = get_zspeceff_fn(img, field)

    fns = get_zpeceff_rawfns(img, field)

    d = Table()
    d["ZMIN"], d["ZMAX"] = zbins[:-1], zbins[1:]
    zcens = 0.5 * (d["ZMIN"] + d["ZMAX"])

    fig, ax = plt.subplots()

    for efftime in fns:

        zkey = fns[efftime]["ZKEY"]
        effkey = fns[efftime]["EFFKEY"]
        fn = fns[efftime]["FN"]
        d.meta[efftime] = fn

        outeffkey = "EFF_{}".format(efftime.upper())

        effd = Table.read(fn)
        sel = np.isfinite(effd[effkey])
        effd = effd[sel]
        zs, raw_effs = effd[zkey], effd[effkey]
        zs = zs.round(2)
        ax.plot(zs, raw_effs, "-o", label="{} raw".format(efftime))

        # smooth spline interpolation
        # https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html#spline-smoothing-in-1-d
        tck = splrep(zs, raw_effs, s=1)
        spl_effs = BSpline(*tck)(zs)
        ax.plot(zs, spl_effs, "-x", label="{} smooth-spline".format(efftime))

        # and now interpolate on zcens
        d[outeffkey] = np.interp(zcens, zs, spl_effs, left=np.nan, right=np.nan)

        if (img == "clauds") & (field == "xmmlss"):

            for zmin, zmax, val in zip(
                [0.0, 1.6, 3.0, 3.6],
                [1.6, 2.1, 3.6, 4.0],
                [1.0, 0.0, None, 0.0],
            ):
                sel = (
                    (d["ZMIN"] >= zmin)
                    & (d["ZMAX"] <= zmax)
                    & (~np.isfinite(d[outeffkey]))
                )
                if val is None:
                    sel2 = (
                        (d["ZMIN"] >= zmin)
                        & (d["ZMAX"] <= zmax)
                        & (np.isfinite(d[outeffkey]))
                    )
                    val = d[outeffkey][sel2][-1]
                d[outeffkey][sel] = val

        ax.plot(zcens, d[outeffkey], lw=4, alpha=0.5, label="{} final".format(efftime))

    ax.set_title("{}-{}".format(img, field))
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Spectroscopic efficiency")
    ax.set_xlim(zbins[0], zbins[-1])
    ax.set_ylim(0, 1)
    ax.grid()
    ax.legend(loc=2)

    plt.savefig(outfn.replace(".ecsv", ".png"), bbox_inches="tight")
    plt.close()

    d.write(outfn)


def get_zspeceff(img, field, efftime, zbins):

    zcens = 0.5 * (zbins[:-1] + zbins[1:])

    fn = get_zspeceff_fn(img, field)
    in_d = Table.read(fn)
    in_zcens = 0.5 * (in_d["ZMIN"] + in_d["ZMAX"])
    in_effs = in_d["EFF_{}".format(efftime.upper())]

    # handle boundaries
    # TODO: code in a more robust way...
    if in_zcens[0] > zcens[0]:
        assert in_zcens[0] - zcens[0] < 0.1
        in_zcens = np.append(zcens[0], in_zcens)
        in_effs = np.append(in_effs[0], in_effs)
    if in_zcens[-1] < zcens[-1]:
        assert zcens[-1] - in_zcens[-1] < 0.1
        in_zcens = np.append(in_zcens, zcens[-1])
        in_effs = np.append(in_effs, in_effs[-1])

    sel = np.isfinite(in_effs)
    in_zcens, in_effs = in_zcens[sel], in_effs[sel]

    zcens = 0.5 * (zbins[:-1] + zbins[1:])
    effs = np.interp(
        zcens,
        in_zcens,
        in_effs,
        left=np.nan,
        right=np.nan,
    )
    return effs


def get_1dgmm_mbest(zs, ngmm):

    zs_copy = None
    if len(zs.shape) == 1:
        zs_copy = zs.copy()
        zs = zs.reshape(-1, 1)
    assert len(zs.shape) == 2

    N = np.arange(1, ngmm)
    models = [None for i in range(len(N))]
    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(zs)

    # compute the AIC and the BIC
    AIC = [m.aic(zs) for m in models]
    BIC = [m.bic(zs) for m in models]

    M_best = models[np.argmin(AIC)]

    if zs_copy is not None:
        zs = zs_copy

    return M_best


def get_1dgmm_pdf(M_best, cens):

    assert len(cens.shape) == 1

    logprob = M_best.score_samples(cens.reshape(-1, 1))
    responsibilities = M_best.predict_proba(cens.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    return pdf


def get_nzs(outfn, img_selection, zbins, overwrite=False):

    nbin = len(zbins) - 1
    zcens = 0.5 * (zbins[:-1] + zbins[1:])

    mydir = os.path.join(os.getenv("DESI_ROOT"), "users", "raichoor", "laelbg")
    sel_fn, eff_fn, efftime, targdens, farate = None, None, None, None, None

    outd = Table()
    outd["ZMIN"], outd["ZMAX"] = zbins[:-1], zbins[1:]
    for key in [
        "NRAW_TARG_PER_DEG2",
        "N_TARG_PER_DEG2",
        "SPECTRO_EFF",
        "N_ZOK_PER_DEG2",
        "N_ZOKOBS_PER_DEG2",
    ]:
        outd[key] = 0.0

    # hsc_wide-v20231206
    # - targ_density and fa rate:
    #   - see Kyle s email from 12/5/23, 5:54 PM pacific
    #   - targdens = 1150/deg2
    #   - 3 passes, 4200 working fibers, 5000 deg2 in 5 yrs, 2h/tile
    #   - farate = 0.83
    # - use n(z) from cosmos2020_zphot
    # - apply per-z 2h-spectro. efficiency from Christophe file
    #   (computed from other data, but should get to the right direction)
    #
    # ngmm = 6: nb of Gaussians in the GaussianMixture
    if img_selection == "hsc-wide_v20231206":

        targdens, farate = 1150.0, 0.83
        efftime = "120min"
        eff_img, eff_field = "clauds", "xmmlss"
        eff_fn = get_zspeceff_fn(eff_img, eff_field)
        sel_fn = os.path.join(
            mydir, "hsc-wide", "analysis", "v20231206", "hsc-wide-v20231206-cosmos.fits"
        )
        ngmm = 6

        d = Table.read(sel_fn)
        sel = (d["ISZPHOT"]) & (d["SELECTION"])
        d = d[sel]
        zs = d["ZPHOT"]

        # "raw" nz, i.e. from data, no smoothing
        raw_nzs = np.zeros(nbin)
        for i in range(nbin):

            sel = (zs >= zbins[i]) & (zs < zbins[i + 1])
            raw_nzs[i] = sel.sum()

        # normalize to targdens
        raw_nzs *= targdens / raw_nzs.sum()
        outd["NRAW_TARG_PER_DEG2"] = raw_nzs.copy()

        # now working from the GMM
        M_best = get_1dgmm_mbest(zs, ngmm)
        nzs = get_1dgmm_pdf(M_best, zcens)

        # normalize to targdens
        nzs *= targdens / nzs.sum()
        outd["N_TARG_PER_DEG2"] = nzs.copy()

        # apply spectro. efficiency
        effs = get_zspeceff(eff_img, eff_field, efftime, zbins)
        assert np.all(np.isfinite(effs))
        outd["SPECTRO_EFF"] = effs
        nzs *= effs
        outd["N_ZOK_PER_DEG2"] = nzs.copy()

        # apply fiberassign rate
        nzs *= farate
        outd["N_ZOKOBS_PER_DEG2"] = nzs.copy()

    # suprime_v20231208
    # - targ_density and fa rate:
    #   - see Kyle s email from 12/5/23, 5:54 PM pacific
    #   - targdens = 800/deg2 (I464=300, I484=200, I505=170, I527=130)
    #   - 2 passes, 4200 working fibers, 5000 deg2 in 3.3 yrs, 2h/tile
    #   - farate = 0.74
    # - per-selband:
    #   - use n(z) from VI_Z with VI_QUALITY>=2.0 and 3.5h < EFFTIME_SPEC < 5.0h
    #   - spec. eff.: use the fraction of VI_QUALITY>=2.0 (flat with redshift)
    #       (not super rigourous as we re using here an optimized selection
    #       not the "djs" which has been observed)
    #
    # ngmm: nb of Gaussians in the GaussianMixture
    if img_selection == "suprime_v20231208":

        targdens, farate = 800.0, 0.74
        efftime = "240min"
        efftimemin_hrs, efftimemax_hrs = 3.5, 5.0
        viqualcut = 2.0
        sel_fn = os.path.join(
            mydir, "suprime", "analysis", "v20231208", "suprime-v20231208-photok.fits"
        )

        selbands = get_selection_selbands("v20231208")
        perband_ngmms = {
            "I464": 9,
            "I484": 9,
            "I505": 9,
            "I527": 6,
        }
        perband_targdenss = {
            "I464": 300.0,
            "I484": 200.0,
            "I505": 170.0,
            "I527": 130.0,
        }
        perband_speceff = {}

        d = Table.read(sel_fn)

        # cut on vi (for all selbands)
        sel = d["VI"].copy()
        d = d[sel]

        total_rawtarg_nzs = np.zeros(nbin) # "raw" targets, ie data, no smoothing
        total_targ_nzs = np.zeros(nbin)  # targets
        total_spec_nzs = np.zeros(nbin)  # secure zspec
        total_specfa_nzs = np.zeros(nbin)  # secure zspec + fa

        for band in selbands:

            sel = d["SEL_{}".format(band)].copy()
            sd = d[sel]

            zs = sd["VI_Z"]
            selok = sd["VI_QUALITY"] >= viqualcut

            # "raw" nz, i.e. from data, no smoothing 
            raw_nzs = np.zeros(nbin)
            for i in range(nbin):
                sel = (zs >= zbins[i]) & (zs < zbins[i + 1]) & (selok)
                raw_nzs[i] = sel.sum()

            # normalize to targdens
            raw_nzs *= perband_targdenss[band] / raw_nzs.sum()
            outd["{}_NRAW_TARG_PER_DEG2".format(band)] = raw_nzs.copy()
            total_rawtarg_nzs += raw_nzs

            # now working from the GMM
            M_best = get_1dgmm_mbest(zs[selok], perband_ngmms[band])
            nzs = get_1dgmm_pdf(M_best, zcens)

            # normalize to targdens
            nzs *= perband_targdenss[band] / nzs.sum()
            outd["{}_N_TARG_PER_DEG2".format(band)] = nzs.copy()
            total_targ_nzs += nzs.copy()

            # apply spectro. efficiency
            perband_speceff[band] = selok.mean()

            outd["{}_SPECTRO_EFF".format(band)] = perband_speceff[band]
            nzs *= perband_speceff[band]
            outd["{}_N_OK_PER_DEG2".format(band)] = nzs.copy()
            total_spec_nzs += nzs

            # apply fiberassign rate
            nzs *= farate
            outd["{}_N_ZOKOBS_PER_DEG2".format(band)] = nzs.copy()
            total_specfa_nzs += nzs

        # record total nzs
        outd["NRAW_TARG_PER_DEG2"] = total_rawtarg_nzs
        outd["N_TARG_PER_DEG2"] = total_targ_nzs
        avg_eff = np.zeros(nbin)
        for band in selbands:
            avg_eff += outd["{}_SPECTRO_EFF".format(band)]
        avg_eff /= len(selbands)
        outd["SPECTRO_EFF"] = avg_eff
        outd["N_ZOK_PER_DEG2"] = total_spec_nzs
        outd["N_ZOKOBS_PER_DEG2"] = total_specfa_nzs

        # per-band infos
        outd.meta["BAND_SEL"] = ",".join(selbands)
        outd.meta["BAND_DNS"] = ",".join(
            [str(perband_targdenss[band]) for band in selbands]
        )
        outd.meta["BAND_EFF"] = ",".join(
            ["{:.2f}".format(perband_speceff[band]) for band in selbands]
        )

    # various infos
    outd.meta["IMG_SEL"] = img_selection
    outd.meta["SELFN"] = sel_fn
    outd.meta["EFFFN"] = eff_fn
    outd.meta["EFFTIME"] = efftime
    outd.meta["TARGDENS"] = targdens
    outd.meta["FARATE"] = farate

    return outd


def plot_nzs(outpng, d, zgoalmin=2.2, zgoalmax=3.6, ylim=None):

    zcens = 0.5 * (d["ZMIN"] + d["ZMAX"])

    fig, ax = plt.subplots()

    # all "raw" targets (ie data)
    ax.plot(
        zcens,                                                                                                                                              
        d["NRAW_TARG_PER_DEG2"],
        color="y",
        alpha=1.0,
        zorder=0,
        label="All targets: data ({:.0f}/deg2)".format(d["NRAW_TARG_PER_DEG2"].sum()),
    )

    # all targets, gmm
    ax.fill_between(
        zcens,
        0.0 * zcens,
        d["N_TARG_PER_DEG2"],
        color="orange",
        alpha=0.5,
        zorder=0,
        label="All targets, GMM ({:.0f}/deg2)".format(d["N_TARG_PER_DEG2"].sum()),
    )

    # spectro. efficiency
    ax.plot(zcens, 10 * d["SPECTRO_EFF"], color="c", lw=2, label="10x spectro. eff.")

    # targets with valid zspec
    ax.plot(
        zcens,
        d["N_ZOK_PER_DEG2"],
        color="r",
        label="After spectro. eff. ({:.0f}/deg2)".format(d["N_ZOK_PER_DEG2"].sum()),
    )

    # observed targets with valid zspec
    ax.plot(
        zcens,
        d["N_ZOKOBS_PER_DEG2"],
        color="k",
        lw=2,
        label="After spec. eff. and FA(={}) ({:.0f}/deg2)".format(
            d.meta["FARATE"], d["N_ZOKOBS_PER_DEG2"].sum()
        ),
    )

    # zgoal
    for z in [zgoalmin, zgoalmax]:
        ax.axvline(z, color="k", ls="--")
    sel = (d["ZMIN"] >= zgoalmin) & (d["ZMAX"] <= zgoalmax)
    goaldens = d["N_ZOKOBS_PER_DEG2"][sel].sum()
    txt = "After spec. eff. and FA:\n {:.0f}/deg2 in {} < z < {}".format(
        goaldens,
        zgoalmin,
        zgoalmax,
    )
    ax.text(0.05, 0.6, txt, color="k", transform=ax.transAxes)

    ax.set_title("{} (efftime = {})".format(d.meta["IMG_SEL"], d.meta["EFFTIME"]))
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("N [/deg2]")
    ax.set_xlim(0, 4)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid()
    ax.legend(loc=2, fontsize=10)

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()
