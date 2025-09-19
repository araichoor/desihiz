#!/usr/bin/env python

import os
import tempfile
import pickle
from time import time
from glob import glob
import multiprocessing
import fitsio
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy import units as u
from astropy.cosmology import Planck18
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

from desitarget.geomask import match_to, match
from redrock.results import read_zscan
from desiutil.log import get_logger
from speclite import filters as speclite_filters

from desihiz.specphot_utils import get_smooth
from desihiz.extras_rr_cnn import wave_lya
from fastspecfit.igm import Inoue14
import multiprocessing

import Lya_zelda as zelda

log = get_logger()

allowed_zelda_models = ["outflow"]
allowed_zelda_geometries = ["tsc"]


def get_zelda_geometry_dict():

    return {
        "ts": "Thin_Shell",
        "gw": "Galactic_Wind",
        "bxsi": "Bicone_X_Slab_In",
        "bxso": "Bicone_X_Slab_Out",
        "tsc": "Thin_Shell_Cont",
    }


def get_default_zelda_dirs():

    zelda_dir = os.path.join(
        os.getenv("DESI_ROOT"), "users", "raichoor", "laelbg", "zelda"
    )
    zelda_grids_dir = os.path.join(zelda_dir, "grids/")  # need the trailing "/" !
    zelda_dnn_dir = os.path.join(zelda_dir, "sklearn-1.4.2")

    return zelda_dir, zelda_grids_dir, zelda_dnn_dir


def get_zelda_init_table(n):

    d = Table()
    d["FIT"] = np.zeros(n, dtype=bool)
    for key in [
        "Z",
        "ZLO",
        "ZHI",
        "LYAPEAK_Z",
        "RCHI2",
        "LYAPEAK_PNR",
        "LYAFLUX",
        "LYACONT",
        "LYAEW",
        "LUMDIST_CM",
    ]:
        if key == "ZHI":
            d[key] = +99.0
        else:
            d[key] = -99.0
    for key in [
        "LYABLU_WLO", "LYABLU_WCEN", "LYABLU_WHI",
        "LYARED_WLO", "LYARED_WCEN", "LYARED_WHI",
        "LYABLU_FLUX", "LYARED_FLUX",
    ]:
        d[key] = np.nan

    return d


# zelda: get desi fwhm value
# R = lambda / (1+z) / fwhm
# eyeballing fig. 33 of desi+22...
def get_zelda_desi_spectro_fwhm(wave, z):
    if (wave < 3600) | (wave > 9800):
        return None
    if (wave >= 3600) & (wave < 5790):
        x0, y0, x1, y1 = 3560, 2023, 5973, 3465
    elif (wave >= 5790) & (wave < 7570):
        x0, y0, x1, y1 = 5560, 3372, 7802, 4877
    else:
        x0, y0, x1, y1 = 7367, 3829, 9893, 5239
    sl = (y1 - y0) / (x1 - x0)
    zp = y0 - sl * x0
    res = sl * wave + zp
    return wave / res / (1 + z)


# zelda: to rescale output model
def get_zelda_rescale(fs, ivs, mod_fs):
    sls = np.logspace(0, 10, 1000)
    n = sls.size
    chi2s = np.array([((fs - mod_fs * sl) ** 2 * ivs).sum() for sl in sls])
    i = chi2s.argmin()
    return sls[i]


def initialize_zelda_peaks():

    return {
        key : np.nan for key in [
            "LYABLU_WLO", "LYABLU_WCEN", "LYABLU_WHI", "LYABLU_FLUX",
            "LYARED_WLO", "LYARED_WCEN", "LYARED_WHI", "LYARED_FLUX",
        ]
    }


def get_zelda_peaks(ws, zelda_z50, zelda_fs50, peak_min_height=0.1):

    # AR initialize
    mydict = initialize_zelda_peaks()

    # AR subtract continuum, normalize peak to 1
    tmpfs = zelda_fs50.copy()
    tmpfs -= np.median(tmpfs)
    tmpfs /= tmpfs.max()

    # AR search peaks
    ii, _ = find_peaks(tmpfs, height=peak_min_height, width=1)

    bound_ii = []
    if ii.size > 0:
        # AR case where there s a single peak, with a bump on the red side
        # AR the code will identify two peaks
        # AR we thus add the constraint that one of the two peaks should be
        # AR bluer than wave_lya
        # AR if not, we just pick the highest flux value peak
        if len(ii) == 2:
            if ws[ii[0]] / (1 + zelda_z50) > wave_lya:
                ii = np.array([ii[tmpfs[ii].argmax()]])

        if len(ii) == 1:
            i = ii[0]
            jj = np.where((ws < ws[i]) & (tmpfs > 0.01))[0]
            if jj.size == 0:
                return mydict
            bound_ii.append(jj[0])
            jj = np.where((ws > ws[i]) & (tmpfs > 0.01))[0]
            if jj.size == 0:
                return mydict
            bound_ii.append(jj[-1])

        if ii.size == 2:
            # AR blue boundary of blue peak
            i = ii[0]
            bound_ii.append(np.where((ws < ws[i]) & (tmpfs > 0.01))[0][0])
            # AR peaks separation wavelength
            # AR if no separation found, we consider it as a single peak
            jj = np.where((np.diff(tmpfs) > 0) & (ws[1:] > ws[ii[0]]) & (ws[1:] < ws[ii[1]]))[0]
            kk = 1 + np.where((np.diff(tmpfs) < 0) & (ws[1:] > ws[ii[0]]) & (ws[1:] < ws[ii[1]]))[0]
            jj = jj[np.in1d(jj, kk)]
            if jj.size == 1:
                bound_ii.append(jj[0])
            else:
                ii = ii[[tmpfs[ii].argmax()]]
            # AR red boundary of red peak
            i = ii[-1]
            bound_ii.append(np.where((ws > ws[i]) & (tmpfs > 0.01))[0][-1])

        if len(ii) == 1:
            assert len(bound_ii) == 2
            mydict["LYARED_WLO"] = ws[bound_ii[0]]
            mydict["LYARED_WHI"] = ws[bound_ii[1]]
            mydict["LYARED_WCEN"] = ws[ii][0]
        if len(ii) == 2:
            assert len(bound_ii) == 3
            mydict["LYABLU_WLO"] = ws[bound_ii[0]]
            mydict["LYABLU_WHI"] = ws[bound_ii[1]]
            mydict["LYABLU_WCEN"] = ws[ii][0]
            mydict["LYARED_WLO"] = ws[bound_ii[1]]
            mydict["LYARED_WHI"] = ws[bound_ii[2]]
            mydict["LYARED_WCEN"] = ws[ii][1]

    return mydict



def zelda_make_plot(
    outpng,
    zelda_model,
    zelda_geometry,
    ws,
    fs,
    ivs,
    z,
    z_peak,
    cont,
    zelda_z50,
    zelda_z16,
    zelda_z84,
    zelda_fs50,
    zelda_fs16,
    zelda_fs84,
    rchi2,
    PNR_t,
    zelda_ws_dict,
    title,
):

    geometry_dict = get_zelda_geometry_dict()

    # flux error
    efs = 1e99 + np.zeros(len(fs))
    sel = ivs > 0
    efs[sel] = 1.0 / np.sqrt(ivs[sel])

    fig, ax = plt.subplots()

    ax.plot(ws, fs, color="b", label="data")
    ax.fill_between(ws, fs - efs, fs + efs, color="b", alpha=0.25)

    ax.axhline(cont, ls="--", color="b", label="data continuum")

    ax.plot(ws, zelda_fs50, color="r", label="zelda fit")
    ax.fill_between(ws, zelda_fs16, zelda_fs84, color="r", alpha=0.25)

    ax.axvline(
        wave_lya * (1 + z_peak),
        ls="--",
        color="g",
        label="lyapeak_z={:.4f}".format(z_peak),
    )
    ax.axvline(wave_lya * (1 + z), ls="--", color="c", label="input_z={:.4f}".format(z))
    ax.axvline(
        wave_lya * (1 + zelda_z50),
        ls="--",
        color="r",
        label="zelda_z={:.4f}".format(zelda_z50),
    )
    ax.axvspan(
        wave_lya * (1 + zelda_z16), wave_lya * (1 + zelda_z84), color="r", alpha=0.25
    )
    ax.legend(loc=2)

    ax.set_title(title)
    ax.set_xlabel("Observed wavelength [A]")
    ax.set_ylabel("Flux density")
    ax.set_xlim((wave_lya - 18.5) * (1 + z), (wave_lya + 18.5) * (1 + z))
    ax.grid()
    ax.set_ylim(-0.5, 6)
    x0, x1, y, dy = 0.55, 0.85, 0.95, -0.05
    for txt0, txt1 in [
        ["model(zelda)", zelda_model.capitalize()],
        ["geometry(zelda)", geometry_dict[zelda_geometry]],
        ["obs_offset(zelda - input)", "{:.1f} A".format(wave_lya * (zelda_z50 - z))],
        [
            "rf_offset(zelda - input)",
            "{:.1f} A".format(wave_lya * (zelda_z50 - z) / (1 + z)),
        ],
        ["rchi2", "{:.1f}".format(rchi2)],
        ["SNR(peak)", "{:.1f}".format(PNR_t)],
    ]:
        ax.text(x0, y, txt0, fontsize=7, fontweight="bold", transform=ax.transAxes)
        ax.text(x1, y, txt1, fontsize=7, fontweight="bold", transform=ax.transAxes)
        y += dy

    # AR plot identified peaks
    npeak = np.isfinite([zelda_ws_dict["LYABLU_WCEN"], zelda_ws_dict["LYARED_WCEN"]]).sum()
    txt = "found {} peak(s)".format(npeak)
    if npeak == 1:
        wlo, whi = zelda_ws_dict["LYARED_WLO"], zelda_ws_dict["LYARED_WHI"]
        sel = (ws >= wlo) & (ws <= whi)
        tmpmax = fs[sel].max()
        ax.fill_between([wlo, whi], [-1, -1], [tmpmax, tmpmax], color="g", alpha=0.5, zorder=0)
    if npeak == 2:
        for wlo, whi in zip(
            [zelda_ws_dict["LYABLU_WLO"], zelda_ws_dict["LYARED_WLO"]],
            [zelda_ws_dict["LYABLU_WHI"], zelda_ws_dict["LYARED_WHI"]],
        ):
            sel = (ws >= wlo) & (ws <= whi)
            tmpmax = fs[sel].max()
            ax.fill_between([wlo, whi], [-1, -1], [tmpmax, tmpmax], alpha=0.5, zorder=0)
        txt += " (b/r ratio={:.1f})".format(zelda_ws_dict["LYABLU_FLUX"] / zelda_ws_dict["LYARED_FLUX"])
    ax.text(0.05, 0.25, txt, transform=ax.transAxes)

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


# zelda: fit the lya line in a spectrum
# - LyaRT_Grid, machine_data: global variables defined in get_zelda_fit()
# - zelda_geometry: tsc only for now
# - z: redshift
# - ws, fs, ivs: spectrum (i.e. 1d arrays)
# - gauss_smooth: smoothing? (None -> no smoothing)
# - outpng: if not None, do a plot
# fitting calling sequence based on https://zelda.readthedocs.io/en/latest/Tutorial_DNN.html
# def get_zelda_fit_one_spectrum(LyaRT_Grid, machine_data, zelda_model, zelda_geometry, z, ws, fs, ivs, gauss_smooth, outpng, title):
def get_zelda_fit_one_spectrum(
    zelda_model, zelda_geometry, tid, z, ws, fs, ivs, phot_conts, gauss_smooth, outpng, title
):

    # log.info(outpng)

    assert len(fs.shape) == 1
    assert len(ivs.shape) == 1

    # make copies
    ws_copy, fs_copy, ivs_copy = ws.copy(), fs.copy(), ivs.copy()

    # initialize table with non-valid values
    d = get_zelda_init_table(1)

    # zelda stuff
    geometry_dict = get_zelda_geometry_dict()
    machine = machine_data["Machine"]
    w_rest_Arr = machine_data["w_rest"]

    # take 18.5A on each side of the lya line; cf. zelda, sect. 3.1.1
    # only fit if that range is fully included in ws
    dofit = True
    if not (z > 0):
        dofit = False

    if dofit:
        wmin, wmax = (wave_lya - 18.5) * (1 + z), (wave_lya + 18.5) * (1 + z)
        if (ws.min() > wmin) | (ws.max() < wmax):
            dofit = False

    # no fit? (i.e. z < 2, approximately)
    if not dofit:
        if outpng is not None:
            fig, ax = plt.subplots()
            ax.set_title(title)
            plt.savefig(outpng, bbox_inches="tight")
            plt.close()
        return d

    sel = (ws > wmin) & (ws < wmax)
    ws, fs, ivs = ws[sel], fs[sel], ivs[sel]
    if phot_conts is not None:
        phot_conts = phot_conts[sel]

    # smoothing?
    if gauss_smooth is not None:
        fs, ivs = get_smooth(fs, ivs, gauss_smooth)

    # we will remove the continuum for the fitting (even we use the tsc model)
    # if phot_conts is provided we use that (median over +/- 2 rest-frame A)
    # else we use spec_cont
    spec_cont = np.median(fs[ivs > 0])
    if phot_conts is not None:
        phot_sel = (ws > (wave_lya - 2) * (1 + z)) & (ws < (wave_lya + 2) * (1 + z))
        phot_cont = np.median(phot_conts[phot_sel])
        log.info(
            "TARGETID={}:\tfor EW, we use phot_cont={:.2f} (spec_cont={:.2f}; phot_cont - spec_cont = {:.2f}; differs by {:.1f}%)".format(
                tid, phot_cont, spec_cont, phot_cont - spec_cont, 100 * (phot_cont - spec_cont) / spec_cont,
            )
        )
        cont = phot_cont
    else:
        log.info("TARGETID={}:\tfor EW, we use spec_cont={:.2f}".format(tid, spec_cont))
        cont = spec_cont

    # flux error
    efs = 1e99 + np.zeros(len(fs))
    sel = ivs > 0
    efs[sel] = 1.0 / np.sqrt(ivs[sel])

    # PNR: peak-to-noise ratio
    # assumes the peak is within 3A (rf) from 1215.67
    i = fs.argmax()
    z_peak = ws[i] / wave_lya - 1.0
    PIX_t = (ws[1] - ws[0]) / (1 + z)
    FWHM_t = get_zelda_desi_spectro_fwhm(ws[i], z)
    PNR_t = fs[i] / efs[i]
    wc = ws[i] / (1 + z)
    wlo, whi = (wc - 3) * (1 + z), (wc + 3) * (1 + z)
    sel = (ws > wlo) & (ws < whi)
    F_t = np.trapz(fs[sel] - cont, x=ws[sel])

    # fit with zelda
    # use spectrum with continuum removed
    # use 100 iterations
    try:
        (
            Sol,
            z_sol,
            log_V_Arr,
            log_N_Arr,
            log_t_Arr,
            z_Arr,
            log_E_Arr,
            log_W_Arr,
        ) = zelda.NN_measure(
            ws, fs - cont, efs, FWHM_t, PIX_t, machine, w_rest_Arr, N_iter=100
        )
    except IndexError:
        log.warning("zelda.NN_measure() failed for TARGETID={}, Z={}".format(tid, z))
        if outpng is not None:
            fig, ax = plt.subplots()
            ax.set_title(title)
            plt.savefig(outpng, bbox_inches="tight")
            plt.close()
        return d

    mod = {}
    PNR = 100000.0  # let's put infinite signal to noise in the model line

    # record the median and (16, 84) interval
    for perc in [16, 50, 84]:

        mod["z_{}".format(perc)] = np.percentile(z_Arr, perc)
        #
        V = 10 ** np.percentile(log_V_Arr, perc)
        log_N = np.percentile(log_N_Arr, perc)
        t = 10 ** np.percentile(log_t_Arr, perc)
        log_E = np.percentile(log_E_Arr, perc)
        W = 10 ** np.percentile(log_W_Arr, perc)

        # creates the line
        if (mod["z_{}".format(perc)] < 0) | (mod["z_{}".format(perc)] > 10):
            mod["flux_{}".format(perc)] = np.nan + fs
            continue

        ws_perc, fs_perc, _ = zelda.Generate_a_real_line(
            mod["z_{}".format(perc)],
            V,
            log_N,
            t,
            F_t,
            log_E,
            W,
            PNR,
            FWHM_t,
            PIX_t,
            LyaRT_Grid,
            geometry_dict[zelda_geometry],
        )

        # Get cooler profiles
        (
            mod["wave_{}".format(perc)],
            mod["flux_{}".format(perc)],
        ) = zelda.plot_a_rebinned_line(ws_perc, fs_perc, PIX_t)
        sel = (mod["wave_{}".format(perc)] >= ws.min()) & (
            mod["wave_{}".format(perc)] <= ws.max()
        )
        if sel.sum() == 0:
            mod["flux_{}".format(perc)] = np.nan + fs
        else:
            mod["wave_{}".format(perc)] = mod["wave_{}".format(perc)][sel]
            mod["flux_{}".format(perc)] = mod["flux_{}".format(perc)][sel]
            mod["flux_{}".format(perc)] = np.interp(
                ws,
                mod["wave_{}".format(perc)],
                mod["flux_{}".format(perc)],
            )
            # k = f_pix_One_Arr.argmax()
            # f_pix_One_Arr /= f_pix_One_Arr[k] / fs[j]

    # rescale the output model
    sl = get_zelda_rescale(fs - cont, efs, mod["flux_50"])
    for perc in [16, 50, 84]:
        mod["flux_{}".format(perc)] = cont + sl * mod["flux_{}".format(perc)]

    # rchi2
    rchi2 = np.dot((fs - mod["flux_50"]) ** 2, ivs) / fs.size

    # lya flux
    lya_flux = np.trapz(mod["flux_50"] - cont, x=ws)

    # lya ew
    # https://github.com/desihub/fastspecfit/blob/2827be47fb46846de43dd167ceb9426387e82631/py/fastspecfit/emlines.py#L944
    lya_ew = lya_flux / cont / (1.0 + mod["z_50"])  # rest frame [A]

    # blue/red peaks
    zelda_ws_dict = initialize_zelda_peaks()
    if mod["z_84"] - mod["z_16"] < 0.01:
        zelda_ws_dict = get_zelda_peaks(ws, mod["z_50"], mod["flux_50"])
        if np.isfinite(zelda_ws_dict["LYABLU_WCEN"]):
            sel = (ws >= zelda_ws_dict["LYABLU_WLO"]) & (ws <= zelda_ws_dict["LYABLU_WHI"])
            zelda_ws_dict["LYABLU_FLUX"] = np.diff(ws)[0] * sel.sum() * ((fs[sel] * ivs[sel]).sum() / ivs[sel].sum() - cont)
        if np.isfinite(zelda_ws_dict["LYARED_WCEN"]):
            sel = (ws >= zelda_ws_dict["LYARED_WLO"]) & (ws <= zelda_ws_dict["LYARED_WHI"])
            zelda_ws_dict["LYARED_FLUX"] = np.diff(ws)[0] * sel.sum() * ((fs[sel] * ivs[sel]).sum() / ivs[sel].sum() - cont)

    # plot?
    if outpng is not None:
        zelda_make_plot(
            outpng,
            zelda_model,
            zelda_geometry,
            ws,
            fs,
            ivs,
            z,
            z_peak,
            cont,
            mod["z_50"],
            mod["z_16"],
            mod["z_84"],
            mod["flux_50"],
            mod["flux_16"],
            mod["flux_84"],
            rchi2,
            PNR_t,
            zelda_ws_dict,
            title,
        )

    # fill table with outputs
    d["FIT"] = True
    d["Z"], d["ZLO"], d["ZHI"] = mod["z_50"], mod["z_16"], mod["z_84"]
    d["LYAPEAK_Z"] = z_peak
    d["RCHI2"], d["LYAPEAK_PNR"] = rchi2, PNR_t
    d["LYACONT"], d["LYAFLUX"], d["LYAEW"] = cont, lya_flux, lya_ew
    for key in [
        "LYABLU_WLO", "LYABLU_WCEN", "LYABLU_WHI", "LYABLU_FLUX",
        "LYARED_WLO", "LYARED_WCEN", "LYARED_WHI", "LYARED_FLUX",
    ]:
        d[key] = zelda_ws_dict[key]

    # luminosity distance (in cm)
    d["LUMDIST_CM"] = Planck18.luminosity_distance(mod["z_50"]).to(u.cm).value

    # restore copies
    ws, fs, ivs = ws_copy, fs_copy, ivs_copy

    return d


# zs : redshifts (1d array)
# fss, ivss : fluxes and ivars (2d arrays)
# gauss_smooth : smooth? (None -> no smoothing)
# outpdf, nplot : None -> no plotting; pick nplot random spectra
def get_zelda_fit(
    zelda_grids_dir,
    zelda_dnn_dir,
    zelda_model,
    zelda_geometry,
    tids,
    zs,
    ws,
    fss,
    ivss,
    phot_cont_coeffs,
    phot_cont_betas,
    phot_cont_ress,
    gauss_smooth,
    outpdf,
    nplot,
    numproc,
):

    assert len(zs.shape) == 1
    nrow = len(zs)
    assert len(ws.shape) == 1
    nwave = len(ws)
    assert fss.shape == (nrow, nwave)
    assert ivss.shape == (nrow, nwave)
    assert tids.shape == (nrow,)

    # zelda stuff
    global LyaRT_Grid
    global machine_data
    geometry_dict = get_zelda_geometry_dict()
    zelda.funcs.Data_location = zelda_grids_dir
    LyaRT_Grid = zelda.load_Grid_Line(geometry_dict[zelda_geometry])
    fn = os.path.join(
        zelda_dnn_dir, "{}_{}_dnn.sav".format(zelda_model, zelda_geometry)
    )
    machine_data = pickle.load(open(fn, "rb"))

    phot_contss = np.array([None for _ in range(nrow)])
    if phot_cont_coeffs is not None:
        from desihiz.extras_phot_continuum import get_cont_powerlaw
        phot_contss = np.nan + np.zeros((nrow, nwave))
        for i in range(nrow):
            # AR power law (with IGM)
            phot_contss[i] = get_cont_powerlaw(
                ws, zs[i], phot_cont_coeffs[i], phot_cont_betas[i]
            )
            # AR add a correction to account for the spectro/phot offsets
            phot_contss[i] += phot_cont_ress[i]

    # AR pdf?
    outpngs = np.array([None for _ in zs])
    titles = np.array([None for _ in zs])
    if outpdf is not None:

        tmpdir = tempfile.mkdtemp()
        np.random.seed(1234)
        if len(tids) < nplot:
            ii = np.arange(len(tids))
        else:
            ii = np.random.choice(len(tids), size=nplot, replace=False)
        outpngs[ii] = [os.path.join(tmpdir, "tmp-{:08d}.png".format(i)) for i in ii]
        titles[ii] = ["TARGETID = {}".format(tid) for tid in tids[ii]]

    # launch multiprocessing
    myargs = []
    for i in range(len(zs)):
        myargs.append(
            (
                zelda_model,
                zelda_geometry,
                tids[i],
                zs[i],
                ws,
                fss[i],
                ivss[i],
                None, # phot_contss[i],
                gauss_smooth,
                outpngs[i],
                titles[i],
            )
        )

    start = time()
    log.info(
        "launch pool for {} calls of get_zelda_fit_one_spectrum() with {} processors".format(
            len(myargs), numproc
        )
    )
    pool = multiprocessing.Pool(numproc)
    with pool:
        ds = pool.starmap(get_zelda_fit_one_spectrum, myargs)
    log.info(
        "get_zelda_fit_one_spectrum() on {} spectra done (took {:.1f}s)".format(
            len(myargs), time() - start
        )
    )

    d = vstack(ds)

    if outpdf is not None:
        os.system("convert {} {}".format(" ".join(outpngs[ii]), outpdf))
        for outpng in outpngs:
            if outpng is not None:
                os.remove(outpng)

    return d
