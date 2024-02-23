#!/usr/bin/env python

"""
Utility functions called by desi_laelf_data
"""


import os
import warnings
import numpy as np
import fitsio
from astropy.table import Table
from astropy.io import fits
from desihiz.hizmerge_io import match_coord
from desihiz.suprime_analysis import get_bug2ok_mags, get_selection_maglims
from desihiz.laelf_utils import allowed_lfsrcs, get_filt_lminmax, get_filtmag, get_nlaes
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from desiutil.log import get_logger

log = get_logger()


# file with the photometry
def get_phot_fn(survey, field):

    if survey == "odin":

        # file with N419, N501, N673 photometry over COSMOS
        if field == "cosmos":

            fn = "/pscratch/sd/d/dstn/odin-cosmos-3/odin-cosmos+hsc-forced-final2.fits"

        # file used for the N419 XMM-LSS target selection
        if field == "xmmlss":

            fn = "/global/cfs/cdirs/cosmo/work/users/dstn/ODIN/xmm-N419/tractor-xmm-N419-hsc-forced.fits"

    # non-bugged suprime photometry (as we want the mag cut to be what they actually are)
    if survey in ["suprime_djs", "suprime_ad"]:

        fn = "/global/cfs/cdirs/desi/users/raichoor/laelbg/suprime/phot/Subaru_tractor_forced_all-redux-20231025.fits"

    log.info("fn = {}".format(fn))

    return fn


# file with the spectroscopy
def get_spec_fn(survey):

    if survey == "odin":

        fn = (
            "/global/cfs/cdirs/desi/users/raichoor/laelbg/odin/v20240129/desi-odin.fits"
        )

    if survey in ["suprime_djs", "suprime_ad"]:

        fn = "/global/cfs/cdirs/desi/users/raichoor/laelbg/suprime/v20231120/desi-suprime.fits"

    log.info("fn = {}".format(fn))

    return fn


# (ra, dec) box to simplify things.. and compute the area
def get_radecbox(survey, field):

    if survey == "odin":

        if field == "cosmos":

            ramin, ramax, decmin, decmax = 148.8, 151.3, 1.0, 3.2

        if field == "xmmlss":

            ramin, ramax, decmin, decmax = 34.5, 37.5, -6.0, -3.5

    if survey in ["suprime_djs", "suprime_ad"]:

        if field == "cosmos":

            ramin, ramax, decmin, decmax = 149.4, 150.8, 1.48, 2.90

    return ramin, ramax, decmin, decmax


# overall filter infos used in the script
def get_filt_infos(survey, field):

    # https://noirlab.edu/science/programs/ctio/filters/Dark-Energy-Camera
    if survey == "odin":

        filts = get_filt_lminmax("odin")

    if survey in ["suprime_djs", "suprime_ad"]:

        filts = get_filt_lminmax("suprime")

    # we remove some filters if need be.. ugly coding!
    filts2rmv = []

    if survey == "suprime_djs":

        if field == "cosmos":

            filts2rmv = ["I427"]

    if survey == "suprime_ad":

        if field == "cosmos":

            filts2rmv = ["I527"]

    all_filtnames = list(filts.keys())

    if survey == "odin":

        if field == "xmmlss":

            filts2rmv = ["N501", "N673"]

    all_filtnames = np.array(list(filts.keys()))
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(all_filtnames)]
    ii = []

    for i, filt in enumerate(all_filtnames):

        if filt in filts2rmv:

            _ = filts.pop(filt)
            ii.append(i)

    filtnames = np.delete(all_filtnames, ii)
    cols = np.delete(cols, ii)

    # print(filts, cols, filtnames, all_filtnames)

    return filts, cols, filtnames, all_filtnames


# minimal efftime_spec to consider a spectro. observation
# odin: 2h
# suprime_djs: 3h (4h were requested)
# suprime_ad: 1.5h (2h were requested)
def get_minefftime(survey):

    if survey == "odin":

        return 2.0 * 3600.0

    if survey == "suprime_djs":

        return 3.0 * 3600.0

    if survey == "suprime_ad":

        return 1.5 * 3600.0


# retrieve the (ra, decs) of the targets
def get_targ_radecs(survey, field, filt, ramin, ramax, decmin, decmax):

    if survey == "odin":

        if filt == "N419":

            if field == "cosmos":

                fn = "/global/cfs/cdirs/desi/survey/fiberassign/special/tertiary/0026/inputcats/COSMOS_LAE_Candidates_2023apr04v2.fits.gz"
                d = Table.read(fn)
                sel = np.array(["419" in _ for _ in d["CANDTYPE"]])

            if field == "xmmlss":

                # remove WIRO targets
                # note there were two classes of ODIN targets (bright, faint)
                fn = "/global/cfs/cdirs/desi/survey/fiberassign/special/tertiary/0018/inputcats/xmm_odin_wiro_merged_targets.fits"
                d = Table.read(fn)

                for key in d.colnames:

                    d[key].name = key.upper()

                sel = d["ODIN"]

            d = d[sel]

        if filt == "N501":

            fn = "/global/cfs/cdirs/desi/survey/fiberassign/special/20220324/LAE_Candidates_NB501_v1_targeting.fits.gz"
            d = Table.read(fn)

        if filt == "N673":

            fn = "/global/cfs/cdirs/desi/survey/fiberassign/special/20220324/LAE_Candidates_NB673_v0_targeting.fits.gz"
            d = Table.read(fn)

    if survey == "suprime_djs":

        assert filt in ["I464", "I484", "I505", "I527"]
        fn = "/global/cfs/cdirs/desi/survey/fiberassign/special/tertiary/0026/inputcats/Suprime-LBG-Schlegel.fits"
        d = Table.read(fn)

        if filt == "I464":

            sel = (d["LBG_SELECTION"] & 1) > 0

        if filt == "I484":

            sel = (d["LBG_SELECTION"] & 2) > 0

        if filt == "I505":

            sel = (d["LBG_SELECTION"] & 4) > 0

        if filt == "I527":

            sel = (d["LBG_SELECTION"] & 8) > 0

        d = d[sel]

    if survey == "suprime_ad":

        assert filt in ["I427", "I464", "I484", "I505"]
        fn = "/global/cfs/cdirs/desi/survey/fiberassign/special/tertiary/0026/inputcats/COSMOS_LAE_Candidates_2023apr04v2.fits.gz"
        d = Table.read(fn)

        if filt == "I427":

            sel = np.array(["427" in _ for _ in d["CANDTYPE"]])

        if filt == "I464":

            sel = np.array(["464" in _ for _ in d["CANDTYPE"]])

        if filt == "I484":

            sel = np.array(["484" in _ for _ in d["CANDTYPE"]])

        if filt == "I505":

            sel = np.array(["505" in _ for _ in d["CANDTYPE"]])

        d = d[sel]

    sel = (
        (d["RA"] > ramin)
        & (d["RA"] < ramax)
        & (d["DEC"] > decmin)
        & (d["DEC"] < decmax)
    )
    d = d[sel]

    return d["RA"], d["DEC"]


def get_sel_infos(survey, filt, verbose=True):

    if survey == "odin":

        if filt == "N419":

            mag_key = "FLUX_N419"
            magmin, magmax = 22.0, 25.0
            col_bands = ["N419", "G"]
            col_keys = ["FLUX_N419", "FLUX_G"]

        if filt == "N501":

            magmin, magmax = 22.5, 25.0
            mag_key = "FLUX_N501"
            col_bands = ["N501", "G"]
            col_keys = ["FLUX_N501", "FLUX_G"]

        if filt == "N673":

            magmin, magmax = 23.0, 26.0
            mag_key = "FLUX_N673"
            col_bands = ["N673", "R"]
            col_keys = ["FLUX_N673", "FLUX_R"]

    # magmin: set to 23.0 for all selections; does not really matters
    # magmax: bug-corrected values
    if survey == "suprime_djs":

        isbug = False
        magmaxs = get_selection_maglims("djs", isbug)
        magmin, magmax = 23.0, magmaxs[filt]

        if filt == "I464":

            mag_key = "FLUX_I_A_L464"
            col_bands = ["I427", "I464"]
            col_keys = ["FLUX_I_A_L427", "FLUX_I_A_L464"]

        if filt == "I484":

            magmin, magmax = 23.0, 24.70
            mag_key = "FLUX_I_A_L484"
            col_bands = ["I464", "I484"]
            col_keys = ["FLUX_I_A_L464", "FLUX_I_A_L484"]

        if filt == "I505":

            magmin, magmax = 23.0, 24.69
            mag_key = "FLUX_I_A_L505"
            col_bands = ["I484", "I505"]
            col_keys = ["FLUX_I_A_L484", "FLUX_I_A_L505"]

        if filt == "I527":

            magmin, magmax = 23.0, 24.99
            mag_key = "FLUX_I_A_L527"
            col_bands = ["I505", "I527"]
            col_keys = ["FLUX_I_A_L505", "FLUX_I_A_L527"]

    # magmin: set to 23.0 for all selections; does not really matters
    # magmax: bug-corrected values (bugged values taken from Subaru_COSMOS_all.ipynb)
    if survey == "suprime_ad":

        magmin = 23.0
        bug2ok_mags = get_bug2ok_mags()

        if filt == "I427":

            magmax = 25.93
            mag_key = "FLUX_I_A_L427"
            col_bands = ["I427", "G"]
            col_keys = ["FLUX_I_A_L427", "FLUX_G"]

        if filt == "I464":

            magmax = 25.84
            mag_key = "FLUX_I_A_L464"
            col_bands = ["I464", "G"]
            col_keys = ["FLUX_I_A_L464", "FLUX_G"]

        if filt == "I484":

            magmax = 25.78
            mag_key = "FLUX_I_A_L484"
            col_bands = ["I484", "G"]
            col_keys = ["FLUX_I_A_L484", "FLUX_G"]

        if filt == "I505":

            magmax = 25.71
            mag_key = "FLUX_I_A_L505"
            col_bands = ["I505", "G"]
            col_keys = ["FLUX_I_A_L505", "FLUX_G"]

        magmax += bug2ok_mags[filt]  # correct for bug
        magmax -= 0.25  # faint cut for 2h, instead of 4h

    if verbose:

        log.info(
            "{}\t{}\t: (magkey, magmin, magmax) = ({}, {}, {}) ; (col_bands, col_keys) = ({}, {})".format(
                survey, filt, mag_key, magmin, magmax, col_bands, col_keys
            )
        )

    return mag_key, magmin, magmax, col_bands, col_keys


def make_plot(d, outpng, viqualcut, lfsrc=None):

    if lfsrc is not None:

        assert lfsrc in allowed_lfsrcs

    survey = d.meta["SURVEY"]
    field = d.meta["FIELD"]
    pfn, sfn = d.meta["PHOTFN"], d.meta["SPECFN"]
    ramin, ramax, decmin, decmax = (
        d.meta["RAMIN"],
        d.meta["RAMAX"],
        d.meta["DECMIN"],
        d.meta["DECMAX"],
    )
    area = d.meta["AREA"]
    minefftime = d.meta["MINEFF"]

    # filter infos
    filts, cols, filtnames, all_filtnames = get_filt_infos(survey, field)

    # flims
    flims = np.array([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]) * 1e-17
    flims = np.append(flims, np.array([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]) * 1e-16)
    flims = np.append(flims, np.array([1e-15]))
    log.info("flims = {}".format(flims))

    # mag, col plotting range
    if survey == "odin":

        xlim, ylim = (21, 26.5), (-3, 1)

    if survey in ["suprime_djs", "suprime_ad"]:

        xlim, ylim = (21, 26.5), (-3, 4)

    extent = (xlim[0], xlim[1], ylim[0], ylim[1])

    # title
    title = "{} ({} < ra < {} , {} < dec < {}) ; phot : {} ; spec : {} ; efftime_spec > {:.1f} hrs".format(
        field,
        ramin,
        ramax,
        decmin,
        decmax,
        os.path.basename(pfn),
        os.path.basename(sfn),
        minefftime / 3600.0,
    )

    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(7, len(all_filtnames), hspace=0.5, wspace=0.2)

    axs = {}
    axs["dens"] = fig.add_subplot(gs[0:2, :])
    axs["maglim"] = fig.add_subplot(gs[2, :])
    axs["fraclae"] = fig.add_subplot(gs[3, :])
    axs["fracobs"] = fig.add_subplot(gs[4, :])
    for i, filt in enumerate(all_filtnames):
        axs[filt] = fig.add_subplot(gs[5:7, i])

    xticks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) * 1e-17
    xticks = np.append(xticks, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) * 1e-16)
    xticks = np.append(xticks, np.array([1e-15]))

    log.info("")
    log.info("# FILT FLIM MAGLIM NTARG NOBS NLAE LAEDENS LAEDENS_ERR PREDICT_LAEDENS")

    for filt, col in zip(filtnames, cols):

        #
        lmin, lmax = filts[filt]["lmin"], filts[filt]["lmax"]
        zmin, zmax = lmin / 1215.67 - 1.0, lmax / 1215.67 - 1.0
        # print("{}\t{:.2f}\t{:.2f}".format(filt, zmin, zmax))

        # LF prediction
        if lfsrc is not None:

            _, lf_denss = get_nlaes(flims, lfsrc, zmin, zmax)

        else:

            lf_denss = np.nan + flims

        #
        maglims = get_filtmag(flims, 0.5 * (lmin + lmax), lmax - lmin)
        axs["maglim"].plot(
            flims,
            maglims,
            color=col,
            label="{} : {:.0f}A - {:.0f}A".format(filt, lmin, lmax),
        )

        mag_key, magmin, magmax, col_bands, col_keys = get_sel_infos(
            survey, filt, verbose=False
        )
        data_mags = np.nan + np.zeros(len(d))
        sel = d[mag_key] > 0
        data_mags[sel] = 22.5 - 2.5 * np.log10(d[mag_key][sel])
        data_cols = np.nan + np.zeros(len(d))
        sel = (d[col_keys[0]] > 0) & (d[col_keys[1]] > 0)
        data_cols[sel] = -2.5 * np.log10(d[col_keys[0]][sel] / d[col_keys[1]][sel])

        sel = (data_mags > magmin) & (data_mags < magmax)

        if survey == "odin":

            sel &= data_cols > -4.0

            if filt == "N419":

                sel &= data_cols < -0.1 - 0.1 * (data_mags - 22.0) ** 2

            if filt == "N501":

                sel &= data_cols < -0.5 - 0.1 * (data_mags - 22.5) ** 2

            if filt == "N673":

                sel &= data_cols < -0.1 - 0.15 * (data_mags - 23.0) ** 2

        # spectro
        tsel = d["TARG_{}".format(filt)].copy()
        axs[filt].scatter(
            data_mags[tsel], data_cols[tsel], c="0.5", s=1, label="spec. targ."
        )
        ssel = d["SPEC_{}".format(filt)].copy()
        axs[filt].scatter(
            data_mags[ssel], data_cols[ssel], c="b", s=1, label="spec. obs."
        )
        ssel &= d["VI_QUALITY"] >= viqualcut
        ssel &= np.array(["LAE" in _ for _ in d["VI_COMMENTS"]])
        axs[filt].scatter(
            data_mags[ssel],
            data_cols[ssel],
            c="orange",
            s=1,
            label="spec. obs. confirmed LAE",
        )

        # cuts
        xs = np.linspace(magmin, magmax, 1000)

        if survey == "odin":

            if filt == "N419":

                axs[filt].plot(
                    xs,
                    -0.1 - 0.1 * (xs - 22) ** 2,
                    color="k",
                    label="y = -0.1 - 0.1 * (x - 22) ** 2",
                )

            if filt == "N501":

                axs[filt].plot(
                    xs,
                    -0.5 - 0.1 * (xs - 22.5) ** 2,
                    color="k",
                    label="y = -0.5 - 0.1 * (x - 22.5) ** 2",
                )

            if filt == "N673":

                axs[filt].plot(
                    xs,
                    -1.0 - 0.15 * (xs - 23) ** 2,
                    color="k",
                    label="y = -1.0 - 0.15 * (x - 23) ** 2",
                )

        axs[filt].axvline(magmin, color="k", ls="--", label="{:.1f}".format(magmin))
        axs[filt].axvline(magmax, color="k", ls="--", label="{:.1f}".format(magmax))

        axs[filt].set_xlabel(filt)
        axs[filt].set_ylabel("{} - {}".format(col_bands[0], col_bands[1]))
        axs[filt].set_xlim(xlim)
        axs[filt].set_ylim(ylim)
        axs[filt].grid()

        if survey == "suprime_ad":

            axs[filt].legend(loc=2, markerscale=5)

        else:

            axs[filt].legend(loc=3, markerscale=5)

        # spec sels
        tsel = (sel) & (d["TARG_{}".format(filt)])
        ssel = (sel) & (d["SPEC_{}".format(filt)])
        sselok = (ssel) & (d["VI_QUALITY"] >= viqualcut)
        sselok &= (d["VI_Z"] >= zmin) & (d["VI_Z"] <= zmax)
        sselok &= np.array(["LAE" in _ for _ in d["VI_COMMENTS"]])

        # phot dens
        data_ntargs = np.nan + np.zeros(len(maglims))
        data_nspecs = np.nan + np.zeros(len(maglims))
        data_nlaes = np.nan + np.zeros(len(maglims))

        for i in range(len(maglims)):

            if maglims[i] < magmax:
                tseli = (tsel) & (data_mags < maglims[i])
                sseli = (ssel) & (data_mags < maglims[i])
                sseloki = (sselok) & (sseli)
                data_ntargs[i] = tseli.sum()
                data_nspecs[i] = sseli.sum()
                data_nlaes[i] = sseloki.sum()
                # print("{}\t{:.1e}\t{:.2f}\t{}\t{}\t{}".format(filt, flims[i], maglims[i], data_ntargs[i], data_nspecs[i], data_nlaes[i]))

        with warnings.catch_warnings():

            warnings.filterwarnings("ignore", category=RuntimeWarning)

            fracobss = data_nspecs / data_ntargs
            fracobs_errs = fracobss * np.sqrt(1 / data_nspecs + 1 / data_ntargs)

            fraclaes = data_nlaes / data_nspecs
            fraclae_errs = fracobss * np.sqrt(1 / data_nlaes + 1 / data_nspecs)

            denss = data_nlaes / area / fracobss
            dens_errs = denss * np.sqrt(
                1.0 / data_nlaes + (fracobs_errs / fracobss) ** 2
            )

        for i in range(len(maglims)):

            log.info(
                "{}\t{:.1e}\t{:.1f}\t{}\t{:.2f}\t{:.0f}\t{:.0f}\t{:.0f}".format(
                    filt,
                    flims[i],
                    maglims[i],
                    data_nlaes[i],
                    fracobss[i],
                    denss[i],
                    dens_errs[i],
                    lf_denss[i],
                )
            )

        for axname, ys, yes, lab in zip(
            ["dens", "fraclae", "fracobs"],
            [denss, fraclaes, fracobss],
            [dens_errs, fraclae_errs, fracobs_errs],
            [
                "{} : data ({:.2f} < z < {:.2f})".format(filt, zmin, zmax),
                "{} ({:.2f})".format(filt, sselok.sum() / ssel.sum()),
                "{} ({:.2f})".format(filt, ssel.sum() / tsel.sum()),
            ],
        ):

            axs[axname].errorbar(flims, ys, yes, color=col, ecolor=col, label=lab)
            axs[axname].fill_between(
                flims, np.clip(ys - yes, 0.01, None), ys + yes, color=col, alpha=0.25
            )

        # LF prediction
        if lfsrc is not None:
            zlab = "z={:.2f}".format(0.5 * (zmin + zmax))
            _, lf_denss = get_nlaes(flims, lfsrc, zmin, zmax)
            axs["dens"].plot(
                flims,
                lf_denss,
                color=col,
                ls="--",
                label="{} : {} {} LF".format(filt, lfsrc, zlab),
            )

    axs["dens"].set_title(title)
    axs["dens"].set_xlabel("Flim [erg/s/cm2/A]")
    axs["dens"].set_ylabel("{} confirmed LAEs [/deg2]".format(survey))
    axs["dens"].set_xlim(1e-17, 1e-15)
    if survey == "odin":
        axs["dens"].set_ylim(1, 500)
    if survey in ["suprime_djs", "suprime_ad"]:
        axs["dens"].set_ylim(1, 500)
    axs["dens"].set_xscale("log")
    axs["dens"].set_yscale("log")
    axs["dens"].set_xticks(xticks)
    axs["dens"].grid()
    axs["dens"].legend(loc=3, ncol=2)

    axs["fraclae"].set_xlabel("Flim [erg/s/cm2/A]")
    axs["fraclae"].set_ylabel("Fraction of conf. LAEs")
    axs["fraclae"].set_xlim(1e-17, 1e-15)
    axs["fraclae"].set_ylim(0, 1)
    axs["fraclae"].set_xscale("log")
    axs["fraclae"].set_xticks(xticks)
    axs["fraclae"].grid()
    axs["fraclae"].legend(loc=3, ncol=1)

    axs["fracobs"].set_xlabel("Flim [erg/s/cm2/A]")
    axs["fracobs"].set_ylabel("Fraction of obs. targs.")
    axs["fracobs"].set_xlim(1e-17, 1e-15)
    axs["fracobs"].set_ylim(0, 1)
    axs["fracobs"].set_xscale("log")
    axs["fracobs"].set_xticks(xticks)
    axs["fracobs"].grid()
    axs["fracobs"].legend(loc=3, ncol=1)

    axs["maglim"].set_xlabel("Flim [erg/s/cm2/A]")
    axs["maglim"].set_ylabel("Limiting magnitude [AB]")
    axs["maglim"].set_xlim(1e-17, 1e-15)
    axs["maglim"].set_ylim(22, 28)
    axs["maglim"].set_xscale("log")
    axs["maglim"].set_xticks(xticks)
    axs["maglim"].grid()
    axs["maglim"].legend(loc=3, ncol=1)

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()
