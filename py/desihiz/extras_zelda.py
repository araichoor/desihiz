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
from scipy.optimize import curve_fit
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
        "LYAEW",
    ]:
        if key == "ZHI":
            d[key] = +99.0
        else:
            d[key] = -99.0

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


def zelda_make_plot(
    outpng,
    zelda_model,
    zelda_geometry,
    ws,
    fs,
    ivs,
    z,
    z_peak,
    zelda_z50,
    zelda_z16,
    zelda_z84,
    zelda_fs50,
    zelda_fs16,
    zelda_fs84,
    rchi2,
    PNR_t,
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

    cont = np.median(fs[ivs > 0])
    ax.axhline(cont, ls="--", color="b", label="data continuum")

    ax.plot(ws, zelda_fs50, color="orange", label="zelda fit")
    ax.fill_between(ws, zelda_fs16, zelda_fs84, color="orange", alpha=0.25)

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
    zelda_model, zelda_geometry, tid, z, ws, fs, ivs, gauss_smooth, outpng, title
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

    # smoothing?
    if gauss_smooth is not None:
        fs, ivs = get_smooth(fs, ivs, gauss_smooth)

    # we will remove the continuum for the fitting (even we use the tsc model)
    cont = np.median(fs[ivs > 0])

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
            mod["z_50"],
            mod["z_16"],
            mod["z_84"],
            mod["flux_50"],
            mod["flux_16"],
            mod["flux_84"],
            rchi2,
            PNR_t,
            title,
        )

    # fill table with outputs
    d["FIT"] = [True]
    d["Z"], d["ZLO"], d["ZHI"] = [mod["z_50"]], [mod["z_16"]], [mod["z_84"]]
    d["LYAPEAK_Z"] = [z_peak]
    d["RCHI2"], d["LYAPEAK_PNR"] = [rchi2], [PNR_t]
    d["LYAFLUX"], d["LYAEW"] = [lya_flux], [lya_ew]

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
    gauss_smooth,
    outpdf,
    nplot,
    numproc,
):

    assert len(zs.shape) == 1
    assert len(ws.shape) == 1
    assert len(fss.shape) == 2
    assert ivss.shape == fss.shape
    assert fss.shape[0] == tids.size
    assert zs.size == tids.size
    assert fss.shape[1] == ws.size

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

    #
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
                # LyaRT_Grid,
                # machine_data,
                zelda_model,
                zelda_geometry,
                tids[i],
                zs[i],
                ws,
                fss[i],
                ivss[i],
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
    log.info("get_zelda_fit_one_spectrum() on {} spectra done (took {:.1f}s)".format(len(myargs), time() - start))

    d = vstack(ds)

    if outpdf is not None:
        os.system("convert {} {}".format(" ".join(outpngs[ii]), outpdf))
        for outpng in outpngs:
            if outpng is not None:
                os.remove(outpng)

    return d
