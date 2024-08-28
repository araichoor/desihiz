#!/usr/bin/env python

import os
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy import units
from astropy.convolution import convolve, Box1DKernel

#
from speclite.filters import load_filter
from redrock.utils import native_endian
from desiutil.log import get_logger
from desispec.io import read_frame, read_sky, read_flux_calibration
from desispec.io.util import get_tempfilename

#
import matplotlib.pyplot as plt
from matplotlib import gridspec

# AR settings
# np_rand_seed = 1234
# np.random.seed(np_rand_seed)
cameras = ["B", "R", "Z"]
fluxunits = 1e-17 * units.erg / units.s / units.cm**2 / units.Angstrom
_sky_coadd = None

allowed_noise_method = ["flux", "ivar", "ivarmed", "ivarmed2"]

log = get_logger()


# AR output file
def get_outfn(
    outdir,
    template_fn,
    template_row,
    efftime_min,
    noise_method,
    rescale_noise_cams,
    rescale_noise_elecs,
    mag_min,
    mag_max,
    mag_bin,
    z_min,
    z_max,
    z_bin,
):
    """
    Get a formatted file name.

    Args:
        outdir: output folder (str)
        template_fn: template filename; expected basename like "rrtemplate-lbg.fits" (str)
        template_row: row number of the template in template_fn (starting at 0) (int)
        efftime_min: EFFTIME_SPEC in minutes (int)
        noise_method: "flux", "ivar", "ivarmed", or "ivarmed2" (str)
        rescale_noise_cams: comma-separated list of the cameras to rescale the rd noise (str)
        rescale_noise_elecs: comma-separated list of the rd noise electrons to rescale to (str)
        mag_min: minimum magnitude (float)
        mag_max: maximum magnitude (float)
        mag_bin: magnitude binning (float)
        z_min: minimum redshift (float)
        z_max: maximum redshift (float)
        z_bin: redshift binning (float)

    Returns:
        outfn: standardize file name (str)

    """
    assert noise_method in allowed_noise_method
    template = (
        os.path.basename(template_fn).replace("rrtemplate-", "").replace(".fits", "")
    )
    outfn = os.path.join(
        outdir,
        "coadd-{}-{}-{}min-{}mag{}-dmag{}-{}z{}-dz{}.fits".format(
            template,
            template_row,
            efftime_min,
            mag_min,
            mag_max,
            mag_bin,
            z_min,
            z_max,
            z_bin,
        ),
    )
    if noise_method != "flux":
        outfn = outfn.replace(
            "coadd-{}-{}".format(template, template_row),
            "coadd-{}-{}-draw{}".format(template, template_row, noise_method),
        )
    if rescale_noise_cams is not None:
        tmpstr = "-".join(
            [
                "{}{}e".format(cam, noise_elec)
                for cam, noise_elec in zip(
                    rescale_noise_cams.split(","), rescale_noise_elecs.split(",")
                )
            ]
        )
        outfn = outfn.replace(
            "coadd-{}-{}".format(template, template_row),
            "coadd-{}-{}-{}".format(template, template_row, tmpstr),
        )
    return outfn


# AR read redrock template
def read_rrtemplate(fn, row):
    """
    Read a redrock-formatted template.

    Args:
        fn: template file name (str)
        row: zero-indexed row of the template to extract (int)

    Returns:
        ws: wavelengths (1d np array of floats)
        fs: fluxes (1d np array of floats)
    """
    fx = fits.open(fn)
    hdr = fx["BASIS_VECTORS"].header
    ws = np.asarray(
        hdr["CRVAL1"] + hdr["CDELT1"] * np.arange(hdr["NAXIS1"]), dtype=np.float64
    )
    if "LOGLAM" in hdr and hdr["LOGLAM"] != 0:
        ws = 10**ws
    fs = np.asarray(native_endian(fx["BASIS_VECTORS"].data), dtype=np.float64)
    fs = fs[row, :]
    return ws, fs


def get_rescale_var_vector(camera, goal_rd_elec, outpng=None):
    """
    Vector to rescale the read noise.

    Args:
        camera: "b", "r", or "z" (str)
        goal_rd_elec: goal read noise, in electron units (float)
        outpng (optional, defaults to None): file name for the control plot (str)

    Returns:
        wave: wavelength array (1d np array of floats)
        ratio: cvar_goalelec / cvar_std array (1d np array of floats)

    Notes:
        Discussion with Julien on Jan. 30th, 2023
        - https://desisurvey.slack.com/archives/D01MQB8D5JQ/p1675115375133969
        - we use here a typical dark exposure: NIGHT=20230128, EXPID=165078, EXPTIME=1051s, EFFTIME_SPEC=572s, SPEED=1.2
        - the median speed for the dark survey so far is ~1.2
        - for now, we hard-code the use of rows=250-500, where both amplifiers have ~consistent read noise for a given camera
    """
    log.info("camera={}, goal_rd_elec={}".format(camera, goal_rd_elec))
    # AR read
    night, expid, petal = 20230128, 165078, 0
    expdir = "/global/cfs/cdirs/desi/spectro/redux/daily/exposures/{}/{:08d}".format(
        night, expid
    )
    sframe = read_frame(
        os.path.join(expdir, "sframe-{}{}-{:08d}.fits.gz".format(camera, petal, expid))
    )
    sky = read_sky(
        os.path.join(expdir, "sky-{}{}-{:08d}.fits.gz".format(camera, petal, expid))
    )
    fluxcal = read_flux_calibration(
        os.path.join(
            expdir, "fluxcalib-{}{}-{:08d}.fits.gz".format(camera, petal, expid)
        )
    )
    cframe = read_frame(
        os.path.join(expdir, "cframe-{}{}-{:08d}.fits.gz".format(camera, petal, expid))
    )
    # AR using AMPC, AMPD
    rowmin, rowmax = 250, 500
    obsrdn_c, obsrdn_d = sframe.meta["OBSRDNC"], sframe.meta["OBSRDND"]
    std_rdnoise = np.round(0.5 * (obsrdn_c + obsrdn_d), 1)
    log.info(
        "obsrdn_c={:.1f}, obsrdn_d={:.1f} -> using std_rdnoise={}".format(
            obsrdn_c, obsrdn_d, std_rdnoise
        )
    )
    # AR select sky fibers
    ii = np.where(cframe.fibermap["OBJTYPE"] == "SKY")[0]
    ii = ii[(ii >= rowmin) & (ii <= rowmax)]
    #
    svar_true = np.median(1 / (sframe.ivar[ii] + (sframe.ivar[ii] == 0)), axis=0)
    skyflux = np.median(sky.flux[ii], axis=0)
    p1 = 1 / 0.8
    sel = (svar_true > 0) & (skyflux > 0) & (skyflux < 400) & (svar_true < 400)
    p0 = np.median((svar_true - p1 * skyflux)[sel]).round(0)
    log.info("p0 = {}".format(p0))
    #
    # p0jguy = {"b":138, "r":66, "z":98}[camera]
    cvar_true = np.median(
        1 / (cframe.ivar[rowmin:rowmax] + (cframe.ivar[rowmin:rowmax] == 0)), axis=0
    )
    cvar_std = np.median(
        (p0 + p1 * sky.flux[rowmin:rowmax]) / fluxcal.calib[rowmin:rowmax] ** 2, axis=0
    )
    # cvar_jguy = np.median((p0jguy + p1 * sky.flux[rowmin:rowmax]) / fluxcal.calib[rowmin:rowmax] ** 2, axis=0)
    cvar_goalelec = np.median(
        (p0 * (goal_rd_elec / std_rdnoise) ** 2 + p1 * sky.flux[rowmin:rowmax])
        / fluxcal.calib[rowmin:rowmax] ** 2,
        axis=0,
    )
    # cvar_goalelec_jguy = np.median((p0jguy * (goal_rd_elec / std_rdnoise) ** 2 + p1 * sky.flux[rowmin:rowmax]) / fluxcal.calib[rowmin:rowmax] ** 2, axis=0)
    #
    if outpng is not None:
        fig = plt.subplots(figsize=(20, 5))
        gs = gridspec.GridSpec(1, 3, wspace=0.3)
        # AR calibrate_variance_model
        ax = plt.subplot(gs[0])
        ax.plot(skyflux, svar_true, ".", alpha=0.1)
        ax.plot(
            skyflux,
            p0 + p1 * skyflux,
            color="orange",
            label="y = {} + {} * x".format(p0, p1),
        )
        # ax.plot(skyflux, p0jguy + p1 * skyflux, label="y = {} + {} * x".format(p0jguy, p1), color="g")
        ax.set_title(
            "{}{}-{:08d}: calibrate_variance_model".format(camera, petal, expid)
        )
        ax.set_xlabel("Median sky flux")
        ax.set_ylabel("Median sframe var")
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 2000)
        ax.grid()
        ax.legend(loc=2)
        # AR variance_model_vs_data
        ax = plt.subplot(gs[1])
        ax.plot(cframe.wave, cvar_true, label="variance in data")
        ax.plot(
            cframe.wave,
            cvar_std,
            color="orange",
            alpha=0.5,
            label="variance model (rdnoise+sky)",
        )
        # ax.plot(cframe.wave, cvar_jguy, alpha=0.5, label="variance model (rdnoise+sky) with p0={}".format(p0jguy), color="g")
        ax.set_title("{}{}-{:08d}: variance_model_vs_data".format(camera, petal, expid))
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylabel("Variance  [e^2/A]")
        ax.grid()
        ax.legend(loc=1)
        # AR variance_ratio_vs_rdnoise
        ax = plt.subplot(gs[2])
        ax.plot(
            cframe.wave,
            cvar_goalelec / cvar_std,
            color="orange",
            label="readnoise from {} to {} elec/pix".format(std_rdnoise, goal_rd_elec),
        )
        # ax.plot(cframe.wave, cvar_goalelec_jguy / cvar_std, alpha=0.5, label="readnoise from {} to {} elec/pix with p0={}".format(std_rdnoise, goal_rd_elec, p0jguy), color="g")
        ax.set_title(
            "{}{}-{:08d}: variance_ratio_vs_rdnoise".format(camera, petal, expid)
        )
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylabel("Variance (or required exposure time) ratio")
        ax.grid()
        ax.legend(loc=2)
        #
        plt.savefig(outpng, bbox_inches="tight")
        plt.close()
    #
    return cframe.wave, cvar_goalelec / cvar_std


def read_sky_coadd(fn):
    """
    Read the (healpix) sky spectra concatenation file.

    Args:
        fn: full path to the file (str)

    Returns:
        _sky_coadd: fits.open(fn)

    Notes:
        fn typically is args.sky_coadd_fn from desi_simcoadd
        (current default is skies-jura.fits, a concatenation of TARGETID>0 sky fibers from healpix files)
    """
    global _sky_coadd
    if _sky_coadd is not None:
        log.info("using cached {}".format(fn))
        return _sky_coadd
    else:
        log.info("reading {}".format(fn))
        _sky_coadd = fits.open(fn)
    return _sky_coadd


# AR select sky fibers in the correct efftime
def get_skies(
    sky_coadd_fn,
    efftime_min,
    relative_tolerance=0.1,
    rescale_noise_cams=None,
    rescale_noise_elecs=None,
):
    """
    Extract sky fibers for a given effective time, and optionally rescale the noise.

    Args:
        sky_coadd_fn: full path to the sky concatenation file (str)
        efftime_min: effective time, in minutes (float)
        relative_tolerance (optional, defaults to 0.1): select with (efftime / efftime_min - 1) < relative_tolerance (float)
        rescale_noise_cams (optional, defaults to None): array of cameras for which rescale the noise (list of str)
        rescale_noise_elecs (optional, defaults to None): array of goal read noise, in electron units (list of float)

    Returns:
        sky, a dictionary with the following keys:
            TARGETID, TSNR2_{BGS,LRG,ELG,LYA}: propagated from sky_coadd_fn["SCORES"]
            {camera}_WAVELENGTH: copied from sky_coadd_fn
            {camera}_FLUX: sky_coadd_fn[{camera}_FLUX] * {camera}_RESCALE_VAR ** 0.5
            {camera}_IVAR: sky_coadd_fn[{camera}_IVAR] / {camera}_RESCALE_VAR
            {camera}_RESOLUTION: copied from sky_coadd_fn
            {camera}_RESCALE_VAR: vector used to rescale {camera}_VAR
            SKY_{camera}_MEDIAN_FLUX, SKY_{camera}_BLUE_MEDIAN_FLUX, SKY_{camera}_RED_MEDIAN_FLUX: median values of the *rescaled* sky flux over the camera, the half-bluest pixels of the camera, and the half-reddest pixels of the camera

    Notes:
        We use for the effective time 12.15 * TSNR2_LRG / 60.
    """
    h = read_sky_coadd(sky_coadd_fn)
    sel = (
        np.abs(12.15 * h["SCORES"].data["TSNR2_LRG"] / 60 / efftime_min - 1)
        < relative_tolerance
    )
    txt = "gathered {} sky fibers with abs(efftime_min/{} - 1) < {}".format(
        sel.sum(), efftime_min, relative_tolerance
    )
    if sel.sum() == 0:
        log.warning(txt)
    else:
        log.info(txt)
    sky = {}
    for key in [
        "TARGETID",
        "TSNR2_BGS",
        "TSNR2_LRG",
        "TSNR2_ELG",
        "TSNR2_LYA",
    ]:
        sky[key] = h["SCORES"].data[key][sel]
    for camera in cameras:
        sky["{}_WAVELENGTH".format(camera)] = h["{}_WAVELENGTH".format(camera)].data
        for ext in [
            "{}_FLUX".format(camera),
            "{}_IVAR".format(camera),
            "{}_RESOLUTION".format(camera),
        ]:
            sky[ext] = h[ext].data[sel]
    # AR sky: rescale?
    # AR TODO : if noise_method!=flux, the rescaling may not be relevant
    for camera in cameras:
        sky["{}_RESCALE_VAR".format(camera)] = np.ones_like(
            sky["{}_WAVELENGTH".format(camera)]
        )
    if rescale_noise_cams is not None:
        for camera, goal_rd_elec in zip(
            rescale_noise_cams.split(","),
            rescale_noise_elecs.split(","),
        ):
            (
                rescale_ws,
                sky["{}_RESCALE_VAR".format(camera.upper())],
            ) = get_rescale_var_vector(camera, int(goal_rd_elec))
            assert np.all(rescale_ws == sky["{}_WAVELENGTH".format(camera.upper())])
    for camera in cameras:
        sky["{}_FLUX".format(camera)] *= sky["{}_RESCALE_VAR".format(camera)] ** 0.5
        sky["{}_IVAR".format(camera)] /= sky["{}_RESCALE_VAR".format(camera)]
    # AR after potentially rescaling, now compute the median values
    # AR    setting flux=nan for ivar=0, then using np.nanmedian()...
    for camera in cameras:
        nw = len(sky["{}_WAVELENGTH".format(camera)])
        fluxes, ivars = (
            sky["{}_FLUX".format(camera)].copy(),
            sky["{}_IVAR".format(camera)].copy(),
        )
        fluxes[ivars == 0] = np.nan
        for key, imin, imax in zip(
            [
                "SKY_{}_MEDIAN_FLUX".format(camera),
                "SKY_{}_BLUE_MEDIAN_FLUX".format(camera),
                "SKY_{}_RED_MEDIAN_FLUX".format(camera),
            ],
            [0, 0, nw // 2],
            [nw, nw // 2, nw],
        ):
            sky[key] = np.nanmedian(fluxes[:, imin:imax], axis=1)
    #
    return sky


def rescale_template2mag(ws, fs, mag, mag_band, verbose=False):
    """
    Return spectra normalized to input magnitudes.

    Args:
        ws: wavelengths (1d array of floats)
        fs: fluxes (1d array of floats)
        mag: desired magnitude (float)
        band: a band loaded with speclite.filters.load_filter() (str)
        verbose (optional, defaults to False): print infos on the prompt (bool)

    Returns:
        rescale_fs: rescaled fluxes

    Notes:
        Similar to desispec.fluxcalibration.normalize_templates,
            but handles padding and possibility to use other filter..
    """
    # AR filter
    filter_response = load_filter(mag_band)
    # AR zero-padding spectrum so that it covers the filter response
    pad_ws, pad_fs = ws.copy(), fs.copy()
    if (pad_ws.min() > filter_response.wavelength.min()) | (
        pad_ws.max() < filter_response.wavelength.max()
    ):
        pad_ws, pad_fs = filter_response.pad_spectrum(pad_fs, pad_ws, method="zero")
    # AR now scale
    orig_mag = filter_response.get_ab_magnitude(pad_fs * fluxunits, pad_ws)
    scalefac = 10 ** ((orig_mag - mag) / 2.5)
    rescale_fs = fs * scalefac
    if verbose:
        log.info(
            "orig_mag={:.1f}\tmag={:.1f}\tscalefac={:.2f}".format(
                orig_mag, mag, scalefac
            )
        )
    return rescale_fs


def get_lsst_bands():
    """
    Get list list of lsst filters.

    Args:
        None

    Returns:
        ["u", "g", "r", "i", "z", "y"]
    """
    return ["u", "g", "r", "i", "z", "y"]


def get_lsst_mags(ws, fs, band, year=2023, np_round=2):
    """
    Compute the magnitudes for a given lsst band.

    Args:
        ws: wavelengths (1d array of floats)
        fs: fluxes (1d or 2d array of floats)
            if 1d-array, should be (nwave)
            if 2d-array, should be (nspec, nwave)
        band: lsst bands (str)
        year (optional, defaults to 2023): speclite version of lsst filters (int)
        np_round (optional, defaults to 2): round mags to np_round digits (int)

    Returns:
        mags: the lsst magnitudes (list of floats, same length as bands)
    """

    filter_response = load_filter("lsst{}-{}".format(year, band))

    # AR zero-padding spectrum so that it covers the filter response
    pad_ws, pad_fs = ws.copy(), fs.copy()

    # AR get the mag
    if len(fs.shape) == 1:
        pad_fs, pad_ws = filter_response.pad_spectrum(fs, ws, method="zero")
        mags = filter_response.get_ab_magnitude(pad_fs * fluxunits, pad_ws)
    elif len(fs.shape) == 2:
        assert fs.shape[1] == len(ws)
        pad_fs, pad_ws = filter_response.pad_spectrum(fs, ws, method="zero", axis=1)
        mags = filter_response.get_ab_magnitude(pad_fs * fluxunits, pad_ws)
    else:
        msg = "unexpected fs.shape = {}; it should be either (nwave) or (nspec, nwave); exit".format(
            fs.shape
        )
        log.error(msg)
        raise ValueError(msg)

    mags = mags.round(np_round)

    return mags


def template_rf2z(rf_ws, rf_fs, cameras_ws, z, mag, mag_band):
    """
    Redshift + rescale a template spectrum.

    Args:
        rf_ws: rest-frame wavelengths (1d array of floats)
        rf_fs: rest-frame fluxes (1d array of floats)
        cameras_ws: list of cameras (list of strs)
        z: redshift to redshift to
        mag: requested magnitude (float)
        mag_band: band for the requested magnitude (str)

    Returns:
        cameras_fs: dictionary, with the redshifted+rescaled flux
            (1d array of floats) for each camera (dictionary)
    """
    # AR first rescale to the requested magnitude
    ws = (1 + z) * rf_ws
    fs = rescale_template2mag(ws, rf_fs, mag, mag_band)
    # AR then interpolate for each camera
    cameras_fs = {}
    for camera in cameras_ws:
        cameras_fs[camera] = np.interp(cameras_ws[camera], ws, fs, left=0, right=0)
    return cameras_fs


def get_tsnr2_truez(
    myd, zs, lya_rf_wmin=1215.7 - 5, lya_rf_wmax=1215.7 + 5, wdelta=0.8, smooth=100
):
    """
    Compute the TSNR2-like values around / excluding the Lya line.

    Args:
        myd: dictionary with at least thse columns:
            "{camera}_WAVELENGTH": (nwave)
            "{camera}_FLUX": (nspec, nwave)
            "{camera}_IVAR": (nspec, nwave)
        zs: redshifts for the nspec spectra (array of floats)
        lya_rf_wmin (optional, defaults to 1215.7 - 5): wavelength (A) of the blueward
            side of the Lya region (float)
        lya_rf_wmax (optional, defaults to 1215.7 + 5): wavelength (A) of the redward
            side of the Lya region (float)
        wdelta (optional, defaults to 0.8): wavelength bin (float)
        smooth (optional, defaults to 100): smoothing scale for dF=<F - smooth(F)>

    Returns:
        mydout: dictionary with the following keys:
            "TSNR2_{camera}_TRUEZLYA": per-camera TSNR2 over the Lya region (nspec)
            "TSNR2_{camera}_TRUEZNOLYA": per-camera TSNR2 excluding the Lya region (nspec)
            "TSNR2_TRUEZLYA" and "TSNR2_TRUEZNOLYA": sum over the cameras (nspec)
    Notes:
        We define TSNR2 = (<F - smooth(F)>) ** 2 * IVAR.
        See desispec.tsnr
    """
    #
    smoothing = np.ceil(smooth / wdelta).astype(int)
    #
    nspec = len(zs)
    mydout = {}
    for suffix in ["TRUEZLYA", "TRUEZNOLYA"]:
        keys = ["TSNR2_{}".format(suffix)]
        keys += ["TSNR2_{}_{}".format(suffix, camera) for camera in cameras]
        for key in keys:
            mydout[key] = np.zeros(nspec)
    for camera in cameras:
        for i in range(nspec):
            # AR nulling the lya region for computing the smoothed spectrum
            lya_wmin, lya_wmax = (1 + zs[i]) * lya_rf_wmin, (1 + zs[i]) * lya_rf_wmax
            ws = myd["{}_WAVELENGTH".format(camera)]
            sel = (ws > lya_wmin) & (ws < lya_wmax)
            fs_i = myd["{}_FLUX".format(camera)][i]
            fs_nolya_i = fs_i.copy()
            fs_nolya_i[sel] = np.nan
            smfs_i = convolve(fs_nolya_i, Box1DKernel(smoothing), boundary="extend")
            dfs_i = fs_i - smfs_i
            if sel.sum() > 0:
                mydout["TSNR2_TRUEZLYA_{}".format(camera)][i] = (
                    dfs_i[sel] ** 2 * myd["{}_IVAR".format(camera)][i, sel]
                ).mean()
            mydout["TSNR2_TRUEZNOLYA_{}".format(camera)][i] = (
                dfs_i[~sel] ** 2 * myd["{}_IVAR".format(camera)][i, ~sel]
            ).mean()
        for suffix in ["TRUEZLYA", "TRUEZNOLYA"]:
            mydout["TSNR2_{}".format(suffix)] += mydout[
                "TSNR2_{}_{}".format(suffix, camera)
            ]
    return mydout


def get_sim(
    rf_ws, rf_fs, sky, zs, mags, mag_band, lsst_bands, np_rand_seed, noise_method
):
    """
    Create a redshifted+rescaled template with realistic noise.

    Args:
        rf_ws: rest-frame wavelengths (1d array of floats)
        rf_fs: rest-frame fluxes (1d array of floats)
        sky: output of get_skies() (Table())
        zs: redshifts (array of floats)
        mags: requested magnitudes for each redshift (array of floats)
        mag_band: band for the requested magnitude (str)
        lsst_bands: list of lsst bands (list of str)
        np_rand_seed: seed to initialize np.random.seed() (int)
        noise_method: "flux", "ivar", "ivarmed", or "ivarmed2" (str)

    Returns:
        myd: dictionary with various entries:
            FIBERMAP: a fibermap-like Table, with the following columns:
                TRUE_Z: input zs
                COADD_FIBERSTATUS: 0
                OBJTYPE: "TGT"
                TARGET_RA, TARGET_DEC: 206.56, 57.07 (coordinate with a low EBV)
                MAG_{band} for band in lsstbands: magnitudes in the lsst bands
                SKY_TARGETID: TARGETID of the sky coadd used for adding noise
                SKY_{camera}_MEDIAN_FLUX, SKY_{camera}_BLUE_MEDIAN_FLUX, SKY_{camera}_RED_MEDIAN_FLUX: median values of the *rescaled* sky flux over the camera, the half-bluest pixels of the camera, and the half-reddest pixels of the camera
            SCORES: a scores-like Table, with the following columns:
                RANDSEED: numpy random seed
                TSNR2_{BGS,LRG,ELG,LYA}: values propagated from the sky coadd
                TSNR2_TRUEZLYA, TSNR2_TRUEZLYA_{B,R,Z}: output from get_tsnr2_truez()
                TSNR2_TRUEZNOLYA, TSNR2_TRUEZNOLYA_{B,R,Z}: output from get_tsnr2_truez()
            {camera}_WAVELENGTH: wavelength array (nwave)
            {camera}_IVAR: ivar array (nsim, nwave), from the sky input
            {camera}_RESOLUTION: copy of the values from the sky input
            {camera}_MASK: zeros array (nsim, nwave)
            {camera}_FLUX_NO_NOISE: redshifted+rescaled template_fs with no noise (nsim, nwave)
            {camera}_FLUX: redshifed+rescaled template_fs, with noise (nsim, nwave)
            {camera}_RESCALE_VAR: propagated from the sky input
    """

    assert noise_method in allowed_noise_method

    zs = np.atleast_1d(zs)
    mags = np.atleast_1d(mags)

    assert len(zs.shape) == 1
    assert len(mags.shape) == 1
    nsim = len(zs)
    assert len(mags) == nsim

    np.random.seed(np_rand_seed)
    nsky = sky["{}_FLUX".format(cameras[0])].shape[0]
    ii_sky = np.random.choice(nsky, size=nsim, replace=True)

    #
    myd = {}
    myd["FIBERMAP"] = Table()
    myd["FIBERMAP"].meta["EXTNAME"] = "FIBERMAP"
    myd["FIBERMAP"]["TRUE_Z"] = zs
    # AR add columns required by redrock
    myd["FIBERMAP"]["COADD_FIBERSTATUS"] = 0
    myd["FIBERMAP"]["OBJTYPE"] = "TGT"
    # AR add columns required by emlinefit
    # AR pick a location with low ebv (=0.0024)
    # AR so that when dust is removed, it is negligible
    # AR (rather than touching the code...)
    myd["FIBERMAP"]["TARGET_RA"], myd["FIBERMAP"]["TARGET_DEC"] = 206.56, 57.07
    # AR scores
    myd["SCORES"] = Table()
    myd["SCORES"].meta["EXTNAME"] = "SCORES"
    myd["SCORES"]["RANDSEED"] = np_rand_seed + np.zeros(nsim, dtype=int)

    # AR template: redshift + mag-rescale
    cameras_ws = {camera: sky["{}_WAVELENGTH".format(camera)] for camera in cameras}
    nws = {camera: cameras_ws[camera].size for camera in cameras}
    template_fs = {camera: np.zeros(0).reshape(0, nws[camera]) for camera in cameras}
    for z, mag in zip(zs, mags):
        tmp_fs = template_rf2z(rf_ws, rf_fs, cameras_ws, z, mag, mag_band)
        for camera in cameras:
            template_fs[camera] = np.append(
                template_fs[camera], tmp_fs[camera].reshape(1, nws[camera]), axis=0
            )

    # AR lsst mags
    myd["FIBERMAP"].meta["FILTERS"] = ",".join(lsst_bands)
    for band in lsst_bands:
        # re-constitute a single array for each spectrum...
        if band == lsst_bands[0]:
            tmp_ws = np.hstack([cameras_ws[camera] for camera in cameras])
            tmp_ws, ii = np.unique(tmp_ws, return_index=True)
        tmp_fs = np.hstack([template_fs[camera] for camera in cameras])
        tmp_fs = tmp_fs[:, ii]
        myd["FIBERMAP"]["MAG_{}".format(band.upper())] = get_lsst_mags(
            tmp_ws, tmp_fs, band
        )

    # AR template: add noise from a randomly picked sky fiber
    myd["FIBERMAP"]["SKY_TARGETID"] = sky["TARGETID"][ii_sky]
    for camera in cameras:
        for key in [
            "SKY_{}_MEDIAN_FLUX".format(camera),
            "SKY_{}_BLUE_MEDIAN_FLUX".format(camera),
            "SKY_{}_RED_MEDIAN_FLUX".format(camera),
        ]:
            myd["FIBERMAP"][key] = sky[key][ii_sky]

    # AR columns
    for key in [
        "TSNR2_BGS",
        "TSNR2_LRG",
        "TSNR2_ELG",
        "TSNR2_LYA",
    ]:
        myd["SCORES"][key] = sky[key][ii_sky]

    for camera in cameras:
        # AR wavelengths
        myd["{}_WAVELENGTH".format(camera)] = sky["{}_WAVELENGTH".format(camera)]
        nw = sky["{}_WAVELENGTH".format(camera)].size
        # AR ivar, resolution (simple copies)
        for ext in [
            "{}_IVAR".format(camera),
            "{}_RESOLUTION".format(camera),
        ]:
            myd[ext] = sky[ext][ii_sky]
        # AR mask
        myd["{}_MASK".format(camera)] = np.zeros((nsim, nw))
        # AR flux with no noise
        myd["{}_FLUX_NO_NOISE".format(camera)] = np.broadcast_to(
            template_fs[camera], (nsim, nw)
        )
        # AR flux with noise
        if noise_method == "flux":
            efs = sky["{}_FLUX".format(camera)][ii_sky]
        # AR
        elif noise_method in ["ivar", "ivarmed", "ivarmed2"]:
            efs = np.zeros_like(sky["{}_FLUX".format(camera)][ii_sky])
            for i, i_sky in enumerate(ii_sky):
                flux_i = sky["{}_FLUX".format(camera)][i_sky]
                ivar_i = sky["{}_IVAR".format(camera)][
                    i_sky
                ]  # AR same as myd["{}_IVAR".format(camera)][i]..
                # AR pick a value from IVAR
                # AR set to 0 where IVAR=0..
                err_i = 0.0 * ivar_i
                sel = ivar_i > 0
                err_i[sel] = ivar_i[sel] ** -0.5
                efs[i] = np.random.normal(size=nw) * err_i
                # AR add an offset, using median value
                # AR maybe it can be done faster with doing that outside of the for i loop...
                if noise_method == "ivarmed":
                    efs[i] += myd["FIBERMAP"]["SKY_{}_MEDIAN_FLUX".format(camera)][i]
                if noise_method == "ivarmed2":
                    for key, imin, imax in zip(
                        [
                            "SKY_{}_BLUE_MEDIAN_FLUX".format(camera),
                            "SKY_{}_RED_MEDIAN_FLUX".format(camera),
                        ],
                        [0, nw // 2],
                        [nw // 2, nw],
                    ):
                        efs[i, imin:imax] += myd["FIBERMAP"][key][i]
        else:
            msg = "noise_method={} not allowed".format(noise_method)
            log.error(msg)
            raise ValueError(msg)
        myd["{}_FLUX".format(camera)] = myd["{}_FLUX_NO_NOISE".format(camera)] + efs
        myd["{}_RESCALE_VAR".format(camera)] = sky["{}_RESCALE_VAR".format(camera)]

    # AR
    tsnr2 = get_tsnr2_truez(myd, zs)
    for key in tsnr2.keys():
        myd["SCORES"][key] = tsnr2[key]
    return myd


def create_rrtemplate_with_z(template_fn, outdir, zmin, zmax):
    """
    Create a new redrock template file, with a redshift grid stored in a REDSHIFTS extension.

    Args:
        template_fn: input template file name (str)
        outdir: output folder name (str)
        zmin: minimum redshift (float)
        zmax: maximum redshift (float)

    Returns:
        Nothing.

    Notes:
        Output file is written to {outdir}/{basename(template_fn)}.
        If outdir does not exist, it is created.
        Redshift grid is 10 ** np.arange(np.log10(1 + zmin), np.log10(1 + zmax), 5e-4) - 1,
            same way as https://github.com/desihub/redrock/blob/0ba968f7c438c8fbd413e953e02f6d02dd5822e6/py/redrock/templates.py#L106
    """
    log.info("read {}".format(template_fn))
    # AR open template (with no REDSHIFTS extension)
    h = fits.open(template_fn)
    # AR add a redshift grid
    zs = 10 ** np.arange(np.log10(1 + zmin), np.log10(1 + zmax), 5e-4) - 1
    log.info("redshift grid: {} values within {} and {}".format(zs.size, zmin, zmax))
    h.append(fits.ImageHDU(data=zs))
    h[-1].header["EXTNAME"] = "REDSHIFTS"
    # AR create folder
    if not os.path.isdir(outdir):
        log.info("create {}".format(outdir))
        os.makedirs(outdir, exist_ok=True)
    # AR write
    outfn = os.path.join(outdir, os.path.basename(template_fn))
    log.info("write to {}".format(outfn))
    h.writeto(outfn, overwrite=True)
