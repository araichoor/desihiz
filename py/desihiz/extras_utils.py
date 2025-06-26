#!/usr/bin/env python

import os
import tempfile
import pickle
from time import time
from glob import glob
import multiprocessing
import fitsio
import numpy as np
from astropy.table import Table, vstack
from astropy import units as u
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

from desitarget.geomask import match_to, match
from redrock.results import read_zscan
from desiutil.log import get_logger
from speclite import filters as speclite_filters

from desihiz.specphot_utils import get_smooth
from fastspecfit.igm import Inoue14
import multiprocessing

import Lya_zelda as zelda

log = get_logger()

wave_lya = 1215.67
wave_oii = 0.5 * (3726.1 + 3728.8)

allowed_extras = ["redrock", "cont", "zelda", "cnn"]

allowed_zelda_models = ["outflow"]
allowed_zelda_geometries = ["tsc"]

igm = None
igm_at_z = None


def get_rrkeys():

    keys = [
        "Z",
        "ZERR",
        "ZWARN",
        "CHI2",
        "COEFF",
        "FITMETHOD",
        "NPIXELS",
        "SPECTYPE",
        "SUBTYPE",
        "NCOEFF",
        "DELTACHI2",
    ]
    keys += ["ALL_Z", "ALL_SPECTYPE", "ALL_DELTACHI2", "RRFN"]
    keys += ["Z_01_BEST"]

    return keys


def get_rrfn(cofn, rrsubdir):

    return os.path.join(
        os.path.dirname(cofn),
        rrsubdir,
        os.path.basename(cofn).replace("coadd", "redrock"),
    )


def read_rrfn(rrfn):

    start = time()
    nbest = 9
    d = Table.read(rrfn, "REDSHIFTS")
    _, zfit = read_zscan(rrfn.replace("redrock", "rrdetails").replace(".fits", ".h5"))
    assert len(zfit) == nbest * len(d)
    assert np.all(zfit["targetid"][::nbest] == d["TARGETID"])
    for key in ["Z", "SPECTYPE", "DELTACHI2"]:
        d["ALL_{}".format(key)] = zfit[key.lower()].reshape((len(d), nbest))
    d["RRFN"] = rrfn
    d.meta = None
    log.info("{:.1f}s\t{}\t{}".format(time() - start, len(d), rrfn))
    return d


def get_rr(tids, cofns, rrsubdir, dchi2_0_min, numproc):

    # read redrock
    rrfns = [get_rrfn(cofn, rrsubdir) for cofn in np.unique(cofns)]
    pool = multiprocessing.Pool(numproc)
    with pool:
        rrs = pool.map(read_rrfn, rrfns)
    d = vstack(rrs)

    # propagate redrock outputs
    unqids = np.array(
        [
            "{}-{}".format(tid, get_rrfn(cofn, rrsubdir))
            for tid, cofn in zip(tids, cofns)
        ]
    )
    rrunqids = np.array(
        ["{}-{}".format(tid, rrfn) for tid, rrfn in zip(d["TARGETID"], d["RRFN"])]
    )
    ii = match_to(rrunqids, unqids)
    assert np.all(rrunqids[ii] == unqids)
    d = d[ii]

    # add a "best" redshift:
    # - if DELTACHI2 > dchi2_0_min -> keep Z=Z_0
    # - if (DELTACHI2 < dchi2_0_min) & (oii is picked but lya is the 2nd best z) -> take Z_1
    d["Z_01_BEST"] = d["Z"].copy()
    sel = d["DELTACHI2"] < dchi2_0_min
    sel &= (
        np.abs((1 + d["ALL_Z"][:, 0]) * wave_oii / wave_lya - 1 - d["ALL_Z"][:, 1]) < 0.01
    )
    d["Z_01_BEST"][sel] = d["ALL_Z"][:, 1][sel]

    return d


def get_igm_inoue14(z, ws, zrounding=3):
    """
    Compute the Inoue+14 IGM transmission for some wavelengths and a redshift.

    Args:
        z: redshift (float)
        ws: wavelengths (np.array of floats)
        zrounding (optional, defaults to 3): rounding to the redshift (int)

    Returns:
        igm_at_z: the transmission values (np.array of floats)
    """

    # AR Inoue+14
    global igm
    if igm is None:
        igm = Inoue14()

    # AR value at z
    # AR force ws to be float (https://github.com/desihub/fastspecfit/issues/230)
    global igm_at_z
    zround_str = "{0:.{1}f}".format(z, zrounding)
    if zround_str not in igm_at_z:
        log.info("compute igm.full_IGM() for zround_str={}".format(zround_str))
        igm_at_z[zround_str] = igm.full_IGM(float(zround_str), get_igm_ref_ws())
    return np.interp(ws, igm_at_z["wave"], igm_at_z[zround_str])


def get_igm_ref_ws(wmin=4000, wmax=10000, dw=1):
    """
    Wavelengths grid for the global igm_at_z variable.

    Args:
        wmin (optional, defaults to 4000): mininum wavelength in A (int)
        wmax (optional, defaults to 10000): maximum wavelength in A (int)
        dw (optional, defaults to 1): the wavelength bin (float)

    Returns:
        np.arange(wmin, wmax + dw, dw)
    """

    return np.arange(wmin, wmax + dw, dw, dtype=float)


def build_igm_inoue14_zgrid(zs, numproc, zrounding=3):
    """
    Compute the Inoue+14 IGM transmission for different redshifts, recorded in the igm_at_z global variable..

    Args:
        zs: redshifts (np.array of floats)
        numproc: number of parallel process (int)
        zrounding (optional, defaults to 3): rounding to the redshift (int)
        ref_ws (optional, defaults to np.arange(4000, 10001)): wavelengths to use (np.array of floats)

    Notes:
        Results are stored in igm_at_z[zround_str].
    """

    # AR Inoue+14
    global igm
    if igm is None:
        igm = Inoue14()

    global igm_at_z
    if igm_at_z is None:
        igm_at_z = {}
        igm_at_z["wave"] = get_igm_ref_ws()

    start = time()
    zrounds = np.unique(zs.round(zrounding))
    zround_strs = np.unique(["{0:.{1}f}".format(z, zrounding) for z in zs])
    zround_strs = [_ for _ in zround_strs if _ not in igm_at_z]
    myargs = [
        (float(zround_str), igm_at_z["wave"].astype(float))
        for zround_str in zround_strs if zround_str not in igm_at_z
    ]
    log.info("start computing igm.full_IGM() for {} values".format(len(myargs)))
    pool = multiprocessing.Pool(processes=numproc)
    with pool:
        outs = pool.starmap(igm.full_IGM, myargs)
    for zround_str, out in zip(zround_strs, outs):
        igm_at_z[zround_str] = out
    log.info("done computing igm.full_IGM() (took {:.1f}s)".format(time() - start))



def get_cont_powerlaw(ws, z, coeff, beta):
    """
    Returns a power law spectrum, used to model LAE/LBG continuum, optionally including the IGM transmission.

    Args:
        ws: the wavelength array (in A) (np.array of floats)
        z: the redshift (float)
        coeff: the multiplicative coefficient (float)
        beta: the slope (float)

    Returns:
        conts: the power law values (np.array of floats)

    Notes:
        If z is set to None, no IGM is used:
            coeff * (ws / 1000) ** beta
        else:
            IGM_transmission * coeff * (ws / 1000) ** beta
    """

    conts = coeff * (ws / 1000) ** beta
    if z is not None:
        taus = get_igm_inoue14(z, ws)
        conts *= taus

    return conts


def get_powerlaw_desifs(specliteindxs_zs, coeff, beta):
    """
    Returns the flux values of a power law convolved with passbands.

    Args:
        specliteindxs_zs: (specliteindxs, zs) where
            specliteindxs refers to the index in get_speclite_all_filts() output
            and zs is repeats of the same redshift, with the same length as
            specliteindxs
        coeff: the multiplicative coefficient (float)
        beta: the slope (float)

    Returns:
        filt_fs: the flux values.

    Notes:
        This function is used in the curve_fit() call in get_continuum_params().
        curve_fit wants independent variables to have the same shape,
        hence the speclitefilts_zs, with artificial zs, instead
        of a single z value.
    """

    specliteindxs, zs = specliteindxs_zs
    specliteindxs = specliteindxs.astype(int)
    nband = len(specliteindxs)

    #all_filts = get_speclite_all_filts()
    speclitefilts = np.array([all_filts[_] for _ in specliteindxs])
    for specliteband in speclitefilts:
        assert hasattr(specliteband, "name")

    assert np.unique(zs).size == 1
    z = zs[0]

    # AR get the power law (+igm) for all wavelengths
    wmin = int(np.min([_.wavelength.min() for _ in speclitefilts])) - 10
    wmax = int(np.max([_.wavelength.max() for _ in speclitefilts])) + 10
    # ws = np.arange(wmin, wmax + 10, 10, dtype=float)
    ws = np.arange(wmin, wmax + 1, 1, dtype=float)
    assert np.all([_.wavelength.min() >= ws[0] for _ in speclitefilts])
    assert np.all([_.wavelength.max() <= ws[-1] for _ in speclitefilts])
    fs = get_cont_powerlaw(ws, z, coeff, beta)

    # AR get the average over the filters
    filt_fs = np.zeros(nband)
    for i in range(nband):
        filt_ws = speclitefilts[i].wavelength
        filt_rs = speclitefilts[i].response
        interp_fs = np.interp(filt_ws, ws, fs)
        filt_fs[i] = np.trapz(interp_fs * filt_rs, filt_ws) / np.trapz(filt_rs, filt_ws)

    return filt_fs


def get_allowed_phot_bands():
    """
    Allowed bands in get_continuum_params().

    Returns:
        bb_bands: list of broad-bands (list of strs)
        not_bb_bands: list of narrow-/medium-bands (list of strs)
    """

    bb_bands = ["U", "US", "G", "R", "R2", "I", "I2", "Z", "Y"]
    not_bb_bands = [
        "N419", "N501", "N673",
        "I427", "I464", "I484", "I505", "I527",
        "M411", "M438", "M464", "M490", "M517",
        "N540"
    ]
    return bb_bands, not_bb_bands


def get_speclite_all_filts():
    """
    All filters used so far in the DESI LAE/LBG TS photometry.

    Returns:
        speclite_filters.load_filters("odin-*", "suprime-*", "ibis-*", "cfht_megacam-*", "hsc2017-*", "decamDR1-*", "merian-*")

    Notes:
        Will need to be updated.
        The Merian/N540 is from some file I have.
    """

    # AR Merian/N540
    from desihiz.specphot_utils import get_filt_fns
    fn = get_filt_fns()["DECAM_N540"]
    d = Table.read(fn)
    d0 = Table()
    d0["WAVE"] = [d["WAVE"][0] - np.diff(d["WAVE"])[0]]
    d0["TRANS"] = 0.
    d1 = Table()
    d1["WAVE"] = [d["WAVE"][-1] + np.diff(d["WAVE"])[-1]]
    d1["TRANS"] = 0.
    d = vstack([d0, d, d1])
    merian_n540 = speclite_filters.FilterResponse(
        wavelength = d["WAVE"] * u.Angstrom,
        response = d["TRANS"],
        meta=dict(group_name="merian", band_name="N540")
    )

    return speclite_filters.load_filters(
            "odin-*",
            "suprime-*",
            "ibis-*",
            "cfht_megacam-*",
            "hsc2017-*",
            "decamDR1-*",
            "merian-N540",
        )


def get_speclite_filtname(band, bb_img=None):
    """
    Get the speclite name for a band.

    Args:
        band: the filter name (str)
        bb_img (optional, defaults to None): the broad-band imaging ("", "CLAUDS", "HSC", or "LS") (str)
        band: the filter name (str)

    Returns:
        The related speclite "name" in the get_speclite_all_filts() output.

    Notes:
        if bb_img = "", then the function returns None.
    """
    if bb_img is not None:
        assert bb_img in ["", "CLAUDS", "HSC", "LS"]
        if bb_img == "":
            return None
        if bb_img in ["CLAUDS", "HSC"]:
            if band.lower() in ["u", "us"]:
                return "cfht_megacam-{}".format(band.lower().replace("us", "ustar"))
            else:
                return "hsc2017-{}".format(band.lower())
        else:
            return "decamDR1-{}".format(band.lower())
    else:
        if band in ["N419", "N501", "N673"]:
            return "odin-{}".format(band)
        elif band in ["I427", "I464", "I484", "I505", "I527"]:
            return "suprime-{}".format(band.replace("I", "IB"))
        elif band in ["M411", "M438", "M464", "M490", "M517"]:
            return "ibis-{}".format(band)
        elif band in ["N540"]:
            return "merian-{}".format(band)
        else:
            msg = "unexpected band={}".format(band)
            log.error(msg)
            raise ValueError(msg)


def get_continuum_params_indiv(s, p, z, phot_bands):
    """
    Estimate a power-law for the continuum, based on the tractor photometry.

    Args:
        s: the "SPECINFO" table for a single object
        p: the "PHOTINFO" or "PHOTV2INFO" table for a single object
        z: redshift (float)
        phot_bands: list of photometric bands used for the continuum estimation (list of strs)

    Returns:
        coeff: the multiplicative coefficient, in 1e-17 erg/s/cm2/A (float)
        beta: the slopes (float)
        weffs: the effective wavelengths of the photometry (np.array of floats)
        wmins: the minimum wavelengths of the bands (np.array of floats)
        wmaxs: the maximum wavelengths of the bands (np.array of floats)
        islyas: is the lya line falling in the band? (np.array of bools)
        phot_fs: the photometry fluxes (np.array of floats)
        phot_ivs: the photometry inverse-variances (np.array of floats)

    Notes:
        If a band contains the lya line, we exclude it from the fit.
        (coeffs, betas) are to be used with get_cont_powerlaw(), ie:
            flux[i] = IGM_trans * coeffs[i] * (ws / 1000) ** betas[i]
        Output coeffs[i] * ws ** betas[i] should be corresponding to the desi flux,
            i.e. 1e-17 erg/cm2/s/A for total flux for psf object.
        The code relies on the tractor photometry.
        From tractor the total flux is converted to fiberflux, then
            MEAN_PSF_TO_FIBER_SPECFLUX is used to convert to
            desi-like total psf flux.
        Columns used from s:
            TARGETID, MEAN_PSF_TO_FIBER_SPECFLUX
        Columns used from p:
            FLUX_{PHOT_BAND} and FLUX_IVAR_{PHOT_BAND}
            FLUX_* and FIBERFLUX_*.
        Output shapes:
            coeffs, betas: scalars
            weffs, wmins, wmaxs, islyas, phot_fs, phot_ivs: (Nband)
    """

    assert isinstance(s["TARGETID"], np.int64)

    nband = len(phot_bands)

    # AR first grab the photometry totalflux -> fiberflux factor
    # AR fiberflux columns will be there for tractor-based catalogs
    # AR if no fiberflux columns (or valid values), we assume a psf profile
    # AR with a fiberflux/totalflux=0.782
    ffkeys = [_ for _ in p.colnames if _[:9] == "FIBERFLUX" and _ != "FIBERFLUX_SYNTHG"]
    tot2fib = np.nan
    for ffkey in ffkeys:
        #
        if p[ffkey] != 0:
            x = p[ffkey] / p[ffkey.replace("FIBERFLUX", "FLUX")]
            if np.isfinite(tot2fib):
                assert np.abs(tot2fib - x) < 1e-6
            else:
                tot2fib = x
    if ~np.isfinite(tot2fib):
        tot2fib = 0.782

    # AR now grab the photometry
    allowed_bb_bands, allowed_not_bb_bands = get_allowed_phot_bands()
    bb_bands = [_ for _ in phot_bands if _ in allowed_bb_bands]
    not_bb_bands = [_ for _ in phot_bands if _ in allowed_not_bb_bands]
    #log.info("bb_bands = {}".format(", ".join(bb_bands)))
    #log.info("not_bb_bands = {}".format(", ".join(not_bb_bands)))
    assert np.all(
        np.isin(
            np.unique(phot_bands),
            np.unique(bb_bands + not_bb_bands)
        )
    )

    fkeys = ["FLUX_{}".format(_) for _ in phot_bands]

    # AR tractor total fluxes (nband)
    phot_fs = np.array([p[k] for k in fkeys])
    phot_ivs = np.array([p[k.replace("FLUX", "FLUX_IVAR")] for k in fkeys])
    # AR tractor fiber fluxes
    phot_fs *= tot2fib
    phot_ivs /= tot2fib ** 2
    # AR (desi-like) psf fluxes
    phot_fs /= s["MEAN_PSF_TO_FIBER_SPECFLUX"]
    phot_ivs *= s["MEAN_PSF_TO_FIBER_SPECFLUX"] ** 2

    # AR speclite filtname for each row/band
    bb_img = p["BB_IMG"]
    #all_filts = get_speclite_all_filts()
    speclite_filtnames = np.zeros(nband, dtype=object)
    for j in range(nband):
        band = phot_bands[j]
        if band in bb_bands:
            if bb_img == "":
                #log.warning("{} has empty bb_img".format(s["TARGETID"]))
                continue
            if (np.isfinite(p[fkeys[j]])) & (p[fkeys[j]] != 0):
                speclite_filtnames[j] = get_speclite_filtname(band, bb_img=bb_img)
        else:
            assert band in not_bb_bands
            speclite_filtnames[j] = get_speclite_filtname(band)

    # AR effective wavelengths
    weffs = np.nan + np.zeros(nband)
    wmins = np.nan + np.zeros(nband)
    wmaxs = np.nan + np.zeros(nband)
    for j in range(nband):
        speclite_filtname = speclite_filtnames[j]
        if speclite_filtname != 0:
            i_filt = [_ for _ in range(len(all_filts.names)) if all_filts.names[_] == speclite_filtname][0]
            weffs[j] = all_filts.effective_wavelengths[i_filt].value
            tmpws, tmprs = all_filts[i_filt].wavelength, all_filts[i_filt].response
            tmpws = tmpws[tmprs > 0.01 * tmprs]
            wmins[j], wmaxs[j] = tmpws[0], tmpws[-1]

    # AR does the band cover lya? (we take a +/- 5A buffer)
    islyas = (wmins < wave_lya * (1 + z) + 5) & (wmaxs > wave_lya * (1 + z) - 5)

    # AR convert fluxes from nanonmaggies to 1e-17 * erg/cm2/s/A
    # AR https://en.wikipedia.org/wiki/AB_magnitude#Expression_in_terms_of_f%CE%BB
    # AR first to Jy:
    # AR                f_Jy = 3.631 * 1e-6 * f_nmgy
    # AR then to erg/s/cm2/A:
    # AR                f_lam = 1 / (3.34 * 1e4 * w ** 2) * f_Jy
    factors = 3.631 * 1e-6 / (3.34 * 1e4 * weffs ** 2) * 1e17
    phot_fs *= factors
    phot_ivs /= factors ** 2

    # AR set ivar to zero for non-valid values
    sel = (~np.isfinite(weffs)) | (~np.isfinite(phot_fs))
    phot_ivs[sel] = 0.

    # AR set ivar to zero for bands covering lya
    phot_ivs[islyas] = 0.

    coeff, beta = np.nan, np.nan
    p0 = np.array([1., -2.])
    bounds = ((0, -5), (100, 5))
    sel = (phot_ivs != 0) & (np.isfinite(phot_fs))
    if sel.sum() == 0:
        log.warning("no valid tractor flux for TARGETID={}".format(s["TARGETID"]))
    else:
        forfit_phot_fs = phot_fs[sel]
        forfit_phot_ivs = phot_ivs[sel]
        forfit_filt_indxs = []
        for filtname in speclite_filtnames[sel]:
            forfit_filt_indxs.append([_ for _ in range(len(all_filts.names)) if all_filts.names[_] == filtname][0])
        try:
            popt, pcov = curve_fit(
                get_powerlaw_desifs,
                (
                    forfit_filt_indxs,
                    np.array([z for _ in weffs[sel]])
                ),
                forfit_phot_fs,
                maxfev=10000000,
                p0=p0,
                sigma=1. / np.sqrt(forfit_phot_ivs),
                bounds=bounds,
            )
            coeff, beta = popt[0], popt[1]
        except ValueError:
            log.warning("fit failed for TARGETID={}".format(s["TARGETID"]))

    return coeff, beta, weffs, wmins, wmaxs, islyas, phot_fs, phot_ivs


def get_continuum_params(s, p, zs, phot_bands, numproc):
    """
    Estimate a power-law for the continuum, based on the tractor photometry.

    Args:
        s: the "SPECINFO" table
        p: the "PHOTINFO" or "PHOTV2INFO" table
        zs: redshifts (np.array of floats)
        phot_bands: list of photometric bands used for the continuum estimation (list of strs)

    Returns:
        coeffs: the multiplicative coefficients, in 1e-17 erg/s/cm2/A (np.array of floats)
        betas: the slopes (np.array of floats)
        weffs: the effective wavelengths of the photometry (np.array of floats)
        wmins: the minimum wavelengths of the bands (np.array of floats)
        wmaxs: the maximum wavelengths of the bands (np.array of floats)
        islyas: is the lya line falling in the band? (np.array of bools)
        phot_fs: the photometry fluxes (np.array of floats)
        phot_ivs: the photometry inverse-variances (np.array of floats)

    Notes:
        If a band contains the lya line, we exclude it from the fit.
        (coeffs, betas) are to be used with get_cont_powerlaw(), ie:
            flux[i] = IGM_trans * coeffs[i] * (ws / 1000) ** betas[i]
        Output coeffs[i] * ws ** betas[i] should be corresponding to the desi flux,
            i.e. 1e-17 erg/cm2/s/A for total flux for psf object.
        The code relies on the tractor photometry.
        From tractor the total flux is converted to fiberflux, then
            MEAN_PSF_TO_FIBER_SPECFLUX is used to convert to
            desi-like total psf flux.
        Columns used from s:
            TARGETID, MEAN_PSF_TO_FIBER_SPECFLUX
        Columns used from p:
            FLUX_{PHOT_BAND} and FLUX_IVAR_{PHOT_BAND}
            FLUX_* and FIBERFLUX_*.
        Output shapes:
            coeffs, betas: (Nrow)
            weffs, wmins, wmaxs, islyas, phot_fs, phot_ivs: (Nrow, Nband)
    """

    nrow = len(p)

    # AR
    global all_filts
    all_filts = get_speclite_all_filts()

    # AR first build the Inoue14 grid
    build_igm_inoue14_zgrid(zs, numproc)

    # AR now grab the photometry
    allowed_bb_bands, allowed_not_bb_bands = get_allowed_phot_bands()
    bb_bands = [_ for _ in phot_bands if _ in allowed_bb_bands]
    not_bb_bands = [_ for _ in phot_bands if _ in allowed_not_bb_bands]
    log.info("bb_bands = {}".format(", ".join(bb_bands)))
    log.info("not_bb_bands = {}".format(", ".join(not_bb_bands)))
    assert np.all(
        np.isin(
            np.unique(phot_bands),
            np.unique(bb_bands + not_bb_bands)
        )
    )

    ffkeys = [_ for _ in p.colnames if _[:9] == "FIBERFLUX" and _ != "FIBERFLUX_SYNTHG"]
    log.info("found these FIBERFLUX columns: {}".format(", ".join(ffkeys)))

    # AR launch fit on each spectrum
    myargs = [
        (s[i], p[i], zs[i], phot_bands) for i in range(nrow)
    ]
    start = time()
    log.info("launch get_continuum_params_indiv() for {} spectra".format(nrow))
    pool = multiprocessing.Pool(processes=numproc)
    with pool:
        outs = pool.starmap(get_continuum_params_indiv, myargs)

    log.info("done computing get_continuum_params_indiv() for {} spectra (took {:.1f}s)".format(nrow, time() - start))

    coeffs = np.array([out[0] for out in outs])
    betas = np.array([out[1] for out in outs])
    weffs = np.vstack([out[2] for out in outs])
    wmins = np.vstack([out[3] for out in outs])
    wmaxs = np.vstack([out[4] for out in outs])
    islyas = np.vstack([out[5] for out in outs])
    phot_fs = np.vstack([out[6] for out in outs])
    phot_ivs = np.vstack([out[7] for out in outs])

    return coeffs, betas, weffs, wmins, wmaxs, islyas, phot_fs, phot_ivs


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



def get_cnnkeys():

    keys = [
        "Z",
        "CL",
    ]

    return keys

def get_cnn(tids, cnnfn):

    d = Table()
    d["CL"] = -99. + np.zeros(len(tids))
    d["Z"] = -99.

    # read the cnn file
    c = Table(fitsio.read(cnnfn))
    ii, iic = match(tids, c["TARGETID"])
    log.info("{}/{} TARGETIDs are in {}".format(ii.size, len(d), cnnfn))
    d["CL"][ii] = c["CL_cnn"][iic]

    # now read the redrock files
    pattern = os.path.join(os.path.dirname(cnnfn), "redrock-*.fits")
    fns = sorted(glob(pattern))
    log.info("found {} {} files".format(len(fns), pattern))
    rr = vstack([Table(fitsio.read(fn, "REDSHIFTS")) for fn in fns])
    ii, iirr = match(tids, rr["TARGETID"])
    log.info("{}/{} TARGETIDs are in the {} {} files".format(ii.size, len(d), len(fns), pattern))
    d["Z"][ii] = rr["Z"][iirr]

    return d
