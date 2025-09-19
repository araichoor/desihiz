#!/usr/bin/env python

import os
import tempfile
from time import time
import multiprocessing
import fitsio
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy import units as u
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

from desiutil.log import get_logger
from speclite import filters as speclite_filters

from desihiz.specphot_utils import get_smooth
from desihiz.extras_rr_cnn import wave_lya
from fastspecfit.igm import Inoue14


log = get_logger()


igm = None
igm_at_z = None
all_filts = None


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
    if igm_at_z is None:
        igm_at_z = {}
        igm_at_z["wave"] = get_igm_ref_ws()

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

    Notes:
        The used wavelength grid is from get_igm_ref_ws().
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
        for zround_str in zround_strs
        if zround_str not in igm_at_z
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

    # AR all filter speclite curves
    global all_filts
    if all_filts is None:
        all_filts = get_speclite_all_filts()

    specliteindxs, zs = specliteindxs_zs
    specliteindxs = specliteindxs.astype(int)
    nband = len(specliteindxs)

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
        "N419",
        "N501",
        "N673",
        "I427",
        "I464",
        "I484",
        "I505",
        "I527",
        "M411",
        "M438",
        "M464",
        "M490",
        "M517",
        "N540",
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
    d0["TRANS"] = 0.0
    d1 = Table()
    d1["WAVE"] = [d["WAVE"][-1] + np.diff(d["WAVE"])[-1]]
    d1["TRANS"] = 0.0
    d = vstack([d0, d, d1])
    merian_n540 = speclite_filters.FilterResponse(
        wavelength=d["WAVE"] * u.Angstrom,
        response=d["TRANS"],
        meta=dict(group_name="merian", band_name="N540"),
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


def get_tractor2desi_factors(p, mean_psf_to_fiber_specfluxes):
    """
    Compute the conversion factor from Tractor fluxes to DESI fluxes.

    Args:
        p: the table with the tractor photometry (FLUX_* columns) (table for a single or multiple rows)
        mean_psf_to_fiber_specfluxes: the DESI MEAN_PSF_TO_FIBER_SPECFLUX column (float or np.array of floats)

    Returns
        tractor2desifs_factors: conversion factors (float or np.array of floats)

    Notes:
        Tractor input fluxes are total fluxes (FLUX_*) in nanomaggies (so per-frequency flux densities).
        We use the available FIBERFLUX columns to infer the 1.5-arcsec-diameter
            fiber fluxes: tot2fibs = FIBERFLUX/FLUX;
            if no available info, we use the default value for a PSF profile: tot2fibs = 0.782.
        The factors are: tot2fibs / mean_psf_to_fiber_specfluxes.
    """

    # AR scalar?
    is_scalar = np.isscalar(p["TARGETID"])
    assert np.isscalar(mean_psf_to_fiber_specfluxes) == is_scalar
    if is_scalar:
        p = Table(p)
        mean_psf_to_fiber_specfluxes = np.array([mean_psf_to_fiber_specfluxes])


    # AR first grab the photometry totalflux -> fiberflux factor
    # AR fiberflux columns will be there for tractor-based catalogs
    # AR if no fiberflux columns (or valid values), we assume a psf profile
    # AR with a fiberflux/totalflux=0.782
    ffkeys = [_ for _ in p.colnames if _[:9] == "FIBERFLUX" and _ != "FIBERFLUX_SYNTHG"]
    tot2fibs = np.nan + np.zeros(len(p))
    for ffkey in ffkeys:
        sel = p[ffkey] != 0
        if sel.sum() > 0:
            xs = p[ffkey][sel] / p[ffkey.replace("FIBERFLUX", "FLUX")][sel]
            prev_xs = tot2fibs[sel]
            sel2 = np.isfinite(prev_xs)
            if sel2.sum() > 0:
                assert np.abs(prev_xs[sel2] - xs[sel2]) < 1e-6
            tot2fibs[sel] = xs
    sel = ~np.isfinite(tot2fibs)
    tot2fibs[sel] = 0.782

    tractor2desifs_factors = tot2fibs / mean_psf_to_fiber_specfluxes

    if is_scalar:
        p, mean_psf_to_fiber_specfluxes = p[0], mean_psf_to_fiber_specfluxes[0]
        tractor2desifs_factors = tractor2desifs_factors[0]

    return tractor2desifs_factors


def get_phot_filt_props(p, zs, phot_bands):
    """
    Get the filter properties for a tractor catalog.

    Args:
        p: the "PHOTINFO" or "PHOTV2INFO" table for a single or several object(s)
        zs: redshifts (float or np.array() of floats)
        phot_bands: list of photometric bands used for the continuum estimation (list of strs)

    Returns:
        weffs: the effective wavelengths of the photometry (np.array of floats)
        wmins: the minimum wavelengths of the bands (np.array of floats)
        wmaxs: the maximum wavelengths of the bands (np.array of floats)
        islyas: is the lya line falling in the band? (np.array of bools)

    Notes:
        The min/max of the band is the FWHM boundaries, i.e. where
            response > 0.5 * max(response).
        Columns used from p:
            FLUX_{PHOT_BAND} and FLUX_IVAR_{PHOT_BAND}
            FLUX_* and FIBERFLUX_*.
        Output shapes:
            weffs, wmins, wmaxs, islyas: (Nband) if inputs are scalar, (Nrow, Nband) otherwise.
    """

    is_scalar = np.isscalar(p["TARGETID"])
    assert np.isscalar(zs) == is_scalar
    if is_scalar:
        p = Table(p)
        zs = np.array([zs])
    assert len(p) == len(zs)

    nrow = len(p)
    nband = len(phot_bands)

    fkeys = ["FLUX_{}".format(_) for _ in phot_bands]

    # AR all filter speclite curves
    global all_filts
    if all_filts is None:
        all_filts = get_speclite_all_filts()

    # AR broad-bands, narrow/medium bands
    allowed_bb_bands, allowed_not_bb_bands = get_allowed_phot_bands()
    bb_bands = [_ for _ in phot_bands if _ in allowed_bb_bands]
    not_bb_bands = [_ for _ in phot_bands if _ in allowed_not_bb_bands]
    assert np.all(np.isin(np.unique(phot_bands), np.unique(bb_bands + not_bb_bands)))

    # AR speclite filtname for each row/band
    speclite_filtnames = np.zeros((nrow, nband), dtype=object)
    unq_bb_imgs = np.unique(p["BB_IMG"])
    unq_bb_imgs = unq_bb_imgs[unq_bb_imgs != ""] # AR exclude those cases for bb_bands
    for j in range(nband):
        band = phot_bands[j]
        if band in bb_bands:
            sel = (np.isfinite(p[fkeys[j]])) & (p[fkeys[j]] != 0)
            for bb_img in unq_bb_imgs:
                sel2 = (sel) & (p["BB_IMG"] == bb_img)
                speclite_filtnames[sel2, j] = get_speclite_filtname(band, bb_img=bb_img)
        else:
            assert band in not_bb_bands
            speclite_filtnames[:, j] = get_speclite_filtname(band)

    # AR effective wavelengths
    weffs = np.nan + np.zeros((nrow, nband))
    wmins = np.nan + np.zeros((nrow, nband))
    wmaxs = np.nan + np.zeros((nrow, nband))
    for j in range(nband):
        unq_speclite_filtnames_j = np.unique(speclite_filtnames[:, j])
        unq_speclite_filtnames_j = unq_speclite_filtnames_j[
            unq_speclite_filtnames_j != 0
        ]
        for speclite_filtname in unq_speclite_filtnames_j:
            i_filt = [
                _
                for _ in range(len(all_filts.names))
                if all_filts.names[_] == speclite_filtname
            ][0]
            sel = speclite_filtnames[:, j] == speclite_filtname
            weffs[sel, j] = all_filts.effective_wavelengths[i_filt].value
            tmpws, tmprs = all_filts[i_filt].wavelength, all_filts[i_filt].response
            #tmpws = tmpws[tmprs > 0.01 * tmprs.max()]
            tmpws = tmpws[tmprs > 0.5 * tmprs.max()]
            wmins[sel, j], wmaxs[sel, j] = tmpws[0], tmpws[-1]

    # AR does the band cover lya? (we take a +/- 5A buffer)
    islyas = (wmins < wave_lya * (1 + zs) + 5) & (wmaxs > wave_lya * (1 + zs) - 5)

    if is_scalar:
        p, zs = p[0], zs[0]
        speclite_filtnames = speclite_filtnames[0]
        weffs, wmins, wmaxs, islyas = weffs[0], wmins[0], wmaxs[0], islyas[0]

    return speclite_filtnames, weffs, wmins, wmaxs, islyas


def get_nmgy2desi_factors(weffs):
    """
    Convert fluxes from nanomaggies to to 1e-17 * erg/cm2/s/A,
        i.e. DESI flux units, for one or more bands.

    Args:
        weffs: the effective wavelength of the band(s) (float or np.array())

    Returns:
        factors: the conversion factor (float, same format as weffs)

    Notes:
        https://en.wikipedia.org/wiki/AB_magnitude#Expression_in_terms_of_f%CE%BB
        first from nmgy to Jy:
                        f_Jy = 3.631 * 1e-6 * f_nmgy
        then from Jy to erg/s/cm2/A:
                        f_lam = 1 / (3.34 * 1e4 * w ** 2) * f_Jy
                              = 3.631 * 1e-6 / (3.34 * 1e4 * w ** 2) * f_nmgy
    """

    factors = 3.631 * 1e-6 / (3.34 * 1e4 * weffs ** 2) * 1e17
    return factors


def identify_bb_bands(phot_bands):
    """
    Identify which bands are broad-bands, and which are not (ie narrow/medium).

    Args:
        phot_bands: list of photometric bands used for the continuum estimation (list of strs)

    Returns:
        bb_bands: list of the broad-bands (list of strs)
        not_bb_bands: list of the not-broad-bands (list of strs)
    """
    allowed_bb_bands, allowed_not_bb_bands = get_allowed_phot_bands()
    bb_bands = [_ for _ in phot_bands if _ in allowed_bb_bands]
    not_bb_bands = [_ for _ in phot_bands if _ in allowed_not_bb_bands]
    #log.info("bb_bands = {}".format(", ".join(bb_bands)))
    #log.info("not_bb_bands = {}".format(", ".join(not_bb_bands)))
    assert np.all(np.isin(np.unique(phot_bands), np.unique(bb_bands + not_bb_bands)))
    return bb_bands, not_bb_bands


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

    # AR load all filters
    global all_filts
    if all_filts is None:
        all_filts = get_speclite_all_filts()

    # AR filter properties
    speclite_filtnames, weffs, wmins, wmaxs, islyas = get_phot_filt_props(p, z, phot_bands)

    # AR factor to convert from tractor total flux to desi psf-flux
    tractor2desi_factors = get_tractor2desi_factors(
        p,
        s["MEAN_PSF_TO_FIBER_SPECFLUX"]
    )

    fkeys = ["FLUX_{}".format(_) for _ in phot_bands]

    # AR tractor total fluxes (nband)
    phot_fs = np.array([p[k] for k in fkeys])
    phot_ivs = np.array([p[k.replace("FLUX", "FLUX_IVAR")] for k in fkeys])
    # AR tractor fiber fluxes
    phot_fs *= tractor2desi_factors
    phot_ivs /= tractor2desi_factors ** 2
    # AR now convert from nanonmaggies to 1e-17 * erg/cm2/s/A
    factors = get_nmgy2desi_factors(weffs)
    phot_fs *= factors
    phot_ivs /= factors**2

    # AR set ivar to zero for non-valid values
    sel = (~np.isfinite(weffs)) | (~np.isfinite(phot_fs))
    phot_ivs[sel] = 0.0

    # AR set ivar to zero for bands covering lya
    orig_phot_ivs = phot_ivs.copy()
    phot_ivs[islyas] = 0.0

    # AR fit
    coeff, beta, rchi2 = np.nan, np.nan, np.nan
    p0 = np.array([1.0, -2.0])
    bounds = ((0, -5), (100, 5))
    sel = (phot_ivs != 0) & (np.isfinite(phot_fs))
    if sel.sum() == 0:
        log.warning("no valid tractor flux for TARGETID={}".format(s["TARGETID"]))
    elif ~np.isfinite(z):
        log.warning("no finite z value for TARGETID={}".format(s["TARGETID"]))
    else:
        forfit_phot_fs = phot_fs[sel]
        forfit_phot_ivs = phot_ivs[sel]
        forfit_filt_indxs = []
        for filtname in speclite_filtnames[sel]:
            forfit_filt_indxs.append(
                [
                    _
                    for _ in range(len(all_filts.names))
                    if all_filts.names[_] == filtname
                ][0]
            )
        forfit_filt_indxs = np.array(forfit_filt_indxs)
        try:
            popt, pcov = curve_fit(
                get_powerlaw_desifs,
                (forfit_filt_indxs, np.array([z for _ in weffs[sel]])),
                forfit_phot_fs,
                maxfev=10000000,
                p0=p0,
                sigma=1.0 / np.sqrt(forfit_phot_ivs),
                bounds=bounds,
            )
            coeff, beta = popt[0], popt[1]
            fit_phot_fs = get_powerlaw_desifs(
                (forfit_filt_indxs, np.array([z for _ in weffs[sel]])),
                coeff,
                beta,
            )
            ndof = len(forfit_phot_fs) - 2
            rchi2 = ((forfit_phot_fs - fit_phot_fs) ** 2 * forfit_phot_ivs).sum() / ndof
        except ValueError:
            log.warning("fit failed for TARGETID={}".format(s["TARGETID"]))

    # AR now compute the rest-frame EW using the relevant band overlapping lya
    # AR - our continuum (with IGM) is: my_cont = igm * cont
    # AR - the tractor narrow/medium-band flux is: tractor_flux = igm * (cont + lya)
    # AR so the rest-frame EW is: (tractor_flux / my_cont - 1)/ (1 + z)
    # AR the relevant band is picked as follows:
    # AR - first restrict to not-broad-bands overlapping the lya
    # AR - then pick the one where the weff is the closest to the lya line
    phot_cont, phot_cont_and_lya, ew, ew_band = np.nan, np.nan, np.nan, ""
    if (np.isfinite(coeff)) & (np.isfinite(beta)) & (np.isfinite(z)):
        phot_ivs = orig_phot_ivs.copy() # AR remove the ivs=0 for band overlapping lya
        bb_bands, not_bb_bands = identify_bb_bands(phot_bands)
        sel = phot_ivs != 0                         # AR valid flux
        sel &= islyas                               # AR overlap lya
        sel &= np.isin(phot_bands, not_bb_bands)    # AR not-broad-band
        sel &= np.isfinite(weffs)                   # AR valid weff
        jj = np.where(sel)[0]
        if jj.size > 0:
            j = jj[np.abs(weffs[jj] - wave_lya * (1 + z)).argmin()]
            phot_cont_and_lya = phot_fs[j]
            filtname = speclite_filtnames[j]
            specliteindx = np.array([_ for _ in range(len(all_filts.names)) if all_filts.names[_] == filtname])
            specliteindx_z = (specliteindx, np.array([z]))
            phot_cont = get_powerlaw_desifs(specliteindx_z, coeff, beta)[0]
            band_widths = np.array([wmax - wmin for wmin, wmax in zip(wmins[jj], wmaxs[jj])])
            ew_band = phot_bands[j]
            #ew = (wmaxs[j] - wmins[j]) * (phot_cont_and_lya / phot_cont - 1.) / (1. + z)
            ew = 10 * (1+z) * (phot_cont_and_lya / phot_cont - 1.) / (1. + z)
            # print("{}\t{:.2f}\t{}\t{:.2f}\t{:.2f}\t{:.2f}".format(p["TARGETID"], z, ew_band, phot_cont, phot_cont_and_lya, ew))


    return coeff, beta, rchi2, weffs, wmins, wmaxs, islyas, phot_fs, phot_ivs, phot_cont, phot_cont_and_lya, ew, ew_band


def get_continuum_params(s, p, zs, phot_bands, numproc):
    """
    Estimate a power-law for the continuum, based on the tractor photometry.

    Args:
        s: the "SPECINFO" table
        p: the "PHOTINFO" or "PHOTV2INFO" table
        zs: redshifts (np.array of floats)
        phot_bands: list of photometric bands used for the continuum estimation (list of strs)
        numproc: number of parallel processes (int)

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

    # AR all filter speclite curves
    global all_filts
    if all_filts is None:
        all_filts = get_speclite_all_filts()

    # AR first build the Inoue14 grid
    build_igm_inoue14_zgrid(zs, numproc)

    # AR identify broad-bands, and narrow/medium-bands
    bb_bands, not_bb_bands = identify_bb_bands(phot_bands)

    ffkeys = [_ for _ in p.colnames if _[:9] == "FIBERFLUX" and _ != "FIBERFLUX_SYNTHG"]
    log.info("found these FIBERFLUX columns: {}".format(", ".join(ffkeys)))

    # AR launch fit on each spectrum
    myargs = [(s[i], p[i], zs[i], phot_bands) for i in range(nrow)]
    start = time()
    log.info("launch get_continuum_params_indiv() for {} spectra".format(nrow))
    pool = multiprocessing.Pool(processes=numproc)
    with pool:
        outs = pool.starmap(get_continuum_params_indiv, myargs)

    log.info(
        "done computing get_continuum_params_indiv() for {} spectra (took {:.1f}s)".format(
            nrow, time() - start
        )
    )

    coeffs = np.array([out[0] for out in outs])
    betas = np.array([out[1] for out in outs])
    rchi2s = np.array([out[2] for out in outs])
    weffs = np.vstack([out[3] for out in outs])
    wmins = np.vstack([out[4] for out in outs])
    wmaxs = np.vstack([out[5] for out in outs])
    islyas = np.vstack([out[6] for out in outs])
    phot_fs = np.vstack([out[7] for out in outs])
    phot_ivs = np.vstack([out[8] for out in outs])
    phot_conts = np.array([out[9] for out in outs])
    phot_cont_and_lyas = np.array([out[10] for out in outs])
    ews = np.array([out[11] for out in outs])
    ew_bands = np.array([out[12] for out in outs])

    return coeffs, betas, rchi2s, weffs, wmins, wmaxs, islyas, phot_fs, phot_ivs, phot_conts, phot_cont_and_lyas, ews, ew_bands


def get_continuum_params_ress_default_wlohi():
    """
    Get the default rest-frame wavelength range used in get_continuum_params_ress().

    Returns:
        rf_wlo 1300: the min. rest-frame wavelength to consider to compute the residuals (int)
        rf_whi = 1900: the max. rest-frame wavelength to consider to compute the residuals (int)
    """

    return 1300, 1900


def get_continuum_params_ress(coeffs, betas, ws, fs, ivs, zs, rf_wlo=None, rf_whi=None):
    """
    Compute the "residuals" of get_continuum_params() w.r.t. the spec. fluxes.

    Args:
        coeffs: the multiplicative coefficients, in 1e-17 erg/s/cm2/A (np.array of floats)
        betas: the slopes (np.array of floats)
        zs: redshifts (np.array of floats)
        ws: the DESI wavelengths (in A) (np.array of floats)
        fs: the DESI fluxes (in 1e-17 erg/s/cm2/A) (np.array of floats)
        ivs: the DESI inverse-variances (np.array of floats)
        rf_wlo (optional, defaults to 1300): the min. rest-frame wavelength to consider to compute the residuals (int)
        rf_whi (optional, defaults to 1900): the max. rest-frame wavelength to consider to compute the residuals (int)

    Returns:
        ress: the residuals, ie median(spec_flux - phot_cont).

    Notes:
        coeffs, betas, zs have shape = (Nrow).
        ws has shape = (Nwave).
        fs, ivs have shape = (Nrow, Nwave).
        We set ress=np.nan for rows where, z or coeff or beta are np.nan.
    """

    if rf_wlo is None:
        rf_wlo = get_continuum_params_ress_default_wlohi()[0]
    if rf_whi is None:
        rf_whi = get_continuum_params_ress_default_wlohi()[1]

    # AR scalar or array?
    if hasattr(coeffs, "__len__"):
        is_scalar = False
        assert hasattr(betas, "__len__")
        assert len(fs.shape) == 2
        assert len(ivs.shape) == 2
        assert hasattr(zs, "__len__")
        nrow, nwave = fs.shape
        assert ws.shape[0] == nwave
        assert ivs.shape == (nrow, nwave)
        assert len(coeffs) == nrow
        assert len(betas) == nrow
    else:
        is_scalar = True
        assert not hasattr(betas, "__len__")
        assert len(fs.shape) == 1
        assert len(ivs.shape) == 1
        nwave = len(ws)
        assert len(fs) == nwave
        assert len(ivs) == nwave

    if is_scalar:
        res = np.nan
        ok = (np.isfinite(zs)) & (np.isfinite(coeffs)) & (np.isfinite(betas))
        if (np.isfinite(zs)) & (np.isfinite(coeffs)) & (np.isfinite(betas)):
            conts = get_cont_powerlaw(ws, zs, coeffs, betas)
            sel = (ws / (1 + zs) > rf_wlo) & (ws / (1 + zs) < rf_whi)
            if sel.sum() > 10:
                ress = np.median(fs[sel] - conts[sel])
    else:
        sel = np.isfinite(zs)
        sel &= np.isfinite(coeffs)
        sel &= np.isfinite(betas)
        ii = np.where(sel)[0]
        ress = np.nan + np.zeros(len(zs))
        for i in ii:
            conts = get_cont_powerlaw(ws, zs[i], coeffs[i], betas[i])
            sel = (ws / (1 + zs[i]) > rf_wlo) & (ws / (1 + zs[i]) < rf_whi)
            if sel.sum() > 10:
                ress[i] = np.median(fs[i, sel] - conts[sel])

    return ress


def plot_continuum_params(
    outpdf,
    nplot,
    numproc,
    s,
    zs,
    ws,
    fs,
    ivs,
    phot_bands,
    weffs,
    wmins,
    wmaxs,
    phot_fs,
    phot_ivs,
    islyas,
    coeffs,
    betas,
    ress,
    photconts,
    photcontandlyas,
    ews,
    ewbands,
):
    """
    Make a diagnosis plot of the continuum estimation from the photometry, and compares with spectroscopy.

    Args:
        outpdf: output pdf file (str)
        nplot: number of indiv. plots (int)
        numproc: number of parallel processes (int)
        s: the desi-{img}.fits SPECINFO table
        zs: the redshift values (np.array of floats)
        ws: the DESI spectroscopy wavelengths (np.array of floats)
        fs: the DESI fluxes (np.array of floats)
        ivs: the DESI inverse-variances (np.array of floats)
        phot_bands: list of photometric bands possibly used for the continuum estimation (list of strs))
        weffs: the effective wavelengths for the phot_bands (np.array of floats)
        wmins: the minimimum wavelengths of the phot_bands (np.array of floats)
        wmaxs: the maximimum wavelengths of the phot_bands (np.array of floats)
        phot_fs: the photometric fluxes for phot_bands (np.array of floats)
        phot_ivs: the photometric inverse-variance for phot_bands (np.array of floats)
        islyas: are the phot_bands overlapping the lya line? (np.array of floats)
        coeffs: the multiplicative coefficients, in 1e-17 erg/s/cm2/A (np.array of floats)
        betas: the slopes (np.array of floats)
        ress: the spectra-phot_cont median residuals (np.array of floats)
        photconts: the phot. continuum flux (in 1e-17 erg/s/cm2/A) at the considered band (np.array of floats)
        photcontandlyas: the phot. continuum+lya flux (in 1e-17 erg/s/cm2/A) at the considered band (np.array of floats)
        ews: the phot. rest-frame EWs (np.array of floats)
        ewbands: the considered band for photconts and photcontandlyas (np.array of strs) 

    Notes:
        ws is a (Nrow) array, fs and ivs are (Nrow, Nwave) arrays.
        phot_bands is a (Nrow) array
        weffs, wmins, wmaxs, phot_fs, phot_ivs, islyas are (Nrow, Nband) arrays.
        coeffs, betas, ress are (Nrow) arrays.
    """

    tmpdir = tempfile.mkdtemp()
    np.random.seed(1234)
    if (len(s) < nplot) | (nplot is None):
        ii = np.arange(len(s))
    else:
        ii = np.random.choice(len(s), size=nplot, replace=False)
    outpngs = np.array(
        [os.path.join(tmpdir, "tmp-{:08d}.png".format(i)) for i in range(len(s))]
    )
    titles = [
        "TARGETID = {} (input_z = {:.2f})".format(tid, z)
        for tid, z in zip(s["TARGETID"], zs)
    ]
    txtss = np.array(
        [
            [
                "Phot. cont. band: {}".format(ewband),
                "Phot. cont. flux: {:.2f}".format(photcont),
                "Phot. cont+lya flux: {:.2f}".format(photcontandlya),
                "Phot. r.-f. EW: {:.1f}A".format(ew)
            ] for ewband, photcont, photcontandlya, ew in zip(
                ewbands, photconts, photcontandlyas, ews
            )
        ]
    )

    myargs = [
        (
            outpngs[i],
            s[i],
            zs[i],
            ws,
            fs[i],
            ivs[i],
            phot_bands,
            weffs[i],
            wmins[i],
            wmaxs[i],
            phot_fs[i],
            phot_ivs[i],
            islyas[i],
            coeffs[i],
            betas[i],
            ress[i],
            titles[i],
            txtss[i],
        )
        for i in ii
    ]
    start = time()
    log.info(
        "launch pool for {} calls of plot_continuum_params_indiv() with {} processors".format(
            len(myargs), numproc
        )
    )
    pool = multiprocessing.Pool(numproc)
    with pool:
        ds = pool.starmap(plot_continuum_params_indiv, myargs)
    log.info(
        "plot_continuum_params_indiv() on {} spectra done (took {:.1f}s)".format(
            len(myargs), time() - start
        )
    )

    start = time()
    os.system("convert {} {}".format(" ".join(outpngs[ii]), outpdf))
    log.info("{} done (took {:.1f}s)".format(outpdf, time() - start))
    for outpng in outpngs[ii]:
        if outpng is not None:
            os.remove(outpng)


def plot_continuum_params_indiv(
    outpng,
    s,
    z,
    ws,
    fs_i,
    ivs_i,
    phot_bands,
    weffs_i,
    wmins_i,
    wmaxs_i,
    phot_fs_i,
    phot_ivs_i,
    islyas_i,
    coeff,
    beta,
    res,
    title,
    txts,
):
    """
    Make a diagnosis plot of the continuum estimation from the photometry, and compares with spectroscopy.

    Args:
        outpng: output png file (str)
        s: the desi-{img}.fits SPECINFO table for a single row
        z: the redshift value (float)
        ws: the DESI spectroscopy wavelengths (np.array of floats)
        fs_i: the DESI flux for the considered object (np.array of floats)
        ivs_i: the DESI inverse-variance for the considered object (np.array of floats)
        phot_bands: list of photometric bands possibly used for the continuum estimation (list of strs)
        weffs_i: the effective wavelengths for the phot_bands (np.array of floats)
        wmins_i: the minimimum wavelengths of the phot_bands (np.array of floats)
        wmaxs_i: the maximimum wavelengths of the phot_bands (np.array of floats)
        phot_fs_i: the photometric fluxes for phot_bands (np.array of floats)
        phot_ivs_i: the photometric inverse-variance for phot_bands (np.array of floats)
        islyas_i: are the phot_bands overlapping the lya line? (np.array of floats)
        coeff: the multiplicative coefficient, in 1e-17 erg/s/cm2/A (floats)
        beta: the slope (floats)
        res: the spectra-phot_cont median residuals (float)
        title: plot title (str)
        txts: lines of text to print on the plot (list of str)

    Notes:
        phot_bands, weffs_i, wmins_i, wmaxs_i, phot_fs_i, phot_ivs_i, islyas_i are (Nband) arrays.
    """

    wcen = wave_lya * (1 + z)

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 2, hspace=0.2)
    for ip, (xlim, ylim, lw) in enumerate(
        zip(
            [(3600, 9800), (wcen - 250, wcen + 250)],
            [(-0.25, 0.50), (-0.25, 1.50)],
            [0.5, 2.0],
        )
    ):
        ax = plt.subplot(gs[ip])
        smf, _ = get_smooth(fs_i, ivs_i, 5)
        ax.scatter(
            weffs_i[~islyas_i],
            phot_fs_i[~islyas_i],
            c="g",
            zorder=2,
            label="Tractor photometry (used)",
        )
        ax.scatter(
            weffs_i[islyas_i],
            phot_fs_i[islyas_i],
            marker="x",
            c="r",
            zorder=2,
            label="Tractor photometry (not used)",
        )
        if ip == 0:
            y, dy = -0.05, -0.015
        if ip == 1:
            y, dy = -0.05, -0.030
        ii = weffs_i.argsort()
        for band, weff, wmin, wmax, islya in zip(
            phot_bands[ii], weffs_i[ii], wmins_i[ii], wmaxs_i[ii], islyas_i[ii]
        ):
            if islya:
                col = "r"
            else:
                col = "g"
            if (np.isfinite(weff)) & (wmax > xlim[0]) & (wmin < xlim[1]):
                ax.plot([wmin, wmax], [y, y], color=col, lw=3, alpha=0.5)
                if (weff > xlim[0]) & (weff < xlim[1]):
                    ax.text(weff, y, band, color=col, ha="center", va="center")
            y += dy
        sel = (np.isfinite(phot_ivs_i)) & (phot_ivs_i != 0)
        ax.errorbar(
            weffs_i[sel],
            phot_fs_i[sel],
            1.0 / np.sqrt(phot_ivs_i[sel]),
            color="none",
            ecolor="g",
            elinewidth=5,
            zorder=2,
        )
        conts = get_cont_powerlaw(ws, z, coeff, beta)
        ax.plot(ws, conts, zorder=3, label="Model Power Law")
        ax.plot(ws, smf, lw=lw, zorder=1, label="DESI spectrum")
        sel = (ws / (1 + z) > 1300) & (ws / (1 + z) < 1900)
        ax.plot(
            ws[sel], ws[sel] * 0 + 0.2, color="k", lw=3, alpha=0.5, label="Region for norm."
        )
        ax.plot(ws, conts + res, lw=1, zorder=3, color="k", label="Renorm. model Power Law")
        ax.axvline(wcen, color="k", lw=0.5, ls="--", zorder=-1)
        if ip == 1:
            ax.axvspan(wcen - 5 * (1 + z), wcen + 5 * (1 + z), color="k", alpha=0.1, zorder=-1)
            x = 0.05
            if ~np.isfinite(coeff):
                ax.text(x, 0.95, "Cont. power-law coeff=np.nan", transform=ax.transAxes)
            else:
                ax.text(
                    x,
                    0.95,
                    "Cont. power-law coeff={:.2f}".format(coeff),
                    transform=ax.transAxes,
                )
            if ~np.isfinite(beta):
                ax.text(x, 0.90, "Cont. power-law beta=np.nan", transform=ax.transAxes)
            else:
                ax.text(
                    x,
                    0.90,
                    "Cont. power-law beta={:.2f}".format(beta),
                    transform=ax.transAxes,
                )
            if ~np.isfinite(res):
                ax.text(x, 0.85, "Cont. power-law res=np.nan", transform=ax.transAxes)
            else:
                ax.text(
                    x,
                    0.85,
                    "(Cont. - spectrum) res.={:.3f}".format(res),
                    transform=ax.transAxes,
                )
            y, dy = 0.70, -0.05
            if txts is not None:
                for txt in txts:
                    ax.text(x, y, txt, transform=ax.transAxes)
                    y += dy
        if ip == 0:
            ax.legend(loc=2, fontsize=8)
        ax.set_title(title)
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("Obs. wavelength [A]")
        ax.set_ylabel("Flux [1e-17 erg/s/cm2/A]")
    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


def wrapper_continuum_params_indiv(outpng, fn, tid, zkey, title, txts):
    """
    Conveniency wrapper function to fit + plot the continuum for a target.

    Args:
        outpng: output png file (str)
        fn: full path to a desi-{img}.fits file (str)
        tid: TARGETID (int)
        zkey: the redshift key to use in the SPECINFO extension of fn (str)
        title: title for the plot (str)
        txts: list of texts to print (could be set to None) (list of strs)
    """

    h = fits.open(fn)
    s = Table(h["SPECINFO"].data)
    if "PHOTV2INFO" in h:
        p = Table(h["PHOTV2INFO"].data)
    else:
        p = Table(h["PHOTINFO"].data)
    ws, fs, ivs = h["BRZ_WAVE"].data, h["BRZ_FLUX"].data, h["BRZ_IVAR"].data

    ii = np.where(s["TARGETID"] == tid)[0]
    if ii.size == 0:
        log.warning(
            "no match for TARGETID={} in {}; no {} generated".format(tid, fn, outpng)
        )
        return None
    if ii.size > 1:
        log.warning(
            "{} matches found for TARGETID={} in {}; pick the first occurence".format(
                tid, fn
            )
        )
    i = ii[0]

    phot_bands = [
        _.replace("FLUX_IVAR_", "") for _ in p.colnames if _[:10] == "FLUX_IVAR_"
    ]
    phot_bands = np.array([_ for _ in phot_bands if _ != "SYNTHG"])

    (
        coeff,
        beta,
        weffs,
        wmins,
        wmaxs,
        islyas,
        phot_fs,
        phot_ivs,
    ) = get_continuum_params_indiv(s[i], p[i], s[zkey][i], phot_bands)

    res = get_continuum_params_ress(coeff, beta, ws, fs[i], ivs[i], s[zkey][i])

    plot_continuum_params_indiv(
        outpng,
        s[i],
        s[zkey][i],
        ws,
        fs[i],
        ivs[i],
        phot_bands,
        weffs,
        wmins,
        wmaxs,
        phot_fs,
        phot_ivs,
        islyas,
        coeff,
        beta,
        res,
        title,
        txts,
    )
