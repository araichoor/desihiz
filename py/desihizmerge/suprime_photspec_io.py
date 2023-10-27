#!/usr/bin/env python

import os
import fitsio
from glob import glob
import tempfile
from speclite import filters as speclite_filters
import numpy as np
from astropy.table import Table, vstack, hstack
from astropy.io import fits
from astropy import units
from desispec.io import read_spectra, write_spectra
from desispec.coaddition import coadd_cameras, coadd_fibermap
from desispec.spectra import stack as spectra_stack
from desiutil.log import get_logger
from desihizmerge.hizmerge_io import match_coord


log = get_logger()


def get_tractor_match(d, tractorfn):

    t = Table(fitsio.read(tractorfn))

    for key in t.colnames:

        t[key].name = key.upper()

    # no duplicates here..
    iid, iit, _, _, _ = match_coord(
        d["TARGET_RA"], d["TARGET_DEC"], t["RA"], t["DEC"], search_radius=1.0
    )
    log.info(
        "match {}/{} unique TARGETIDs".format(iid.size, np.unique(d["TARGETID"]).size)
    )

    # handle duplicated TARGETIDs
    sel = np.in1d(d["TARGETID"], d["TARGETID"][iid])
    match_d = d[sel]
    match_t = Table()

    for key in t.colnames:

        if len(t[key].shape) == 1:

            match_t[key] = np.zeros_like(t[key], shape=(sel.sum(),))

        elif len(t[key].shape) == 2:

            match_t[key] = np.zeros_like(t[key], shape=(sel.sum(), t[key].shape[1]))

        else:

            msg = "unexpected t[{}].shape={}".format(key, t[key].shape)
            log.error(msg)
            raise ValueError(msg)

    for i, it in zip(iid, iit):

        sel = match_d["TARGETID"] == d["TARGETID"][i]

        for key in t.colnames:

            if len(t[key].shape) == 1:

                match_t[key][sel] = t[key][it]

            if len(t[key].shape) == 2:

                match_t[key][sel, :] = t[key][it, :]

    assert np.all(match_t["RA"] != 0.0)
    log.info(
        "returning tables with {} rows ({} unique TARGETIDs)".format(
            len(match_d), iid.size
        )
    )

    return match_d, match_t


def get_filts(bands, waves):

    filtdir = os.path.join(
        os.getenv("DESI_ROOT"), "users", "raichoor", "laelbg", "suprime", "filt"
    )
    myfilts = {}

    for band in bands:

        filtfn = os.path.join(
            filtdir, "Subaru_Suprime.{}.dat".format(band.replace("I", "IB"))
        )
        d = Table.read(filtfn, format="ascii.commented_header")
        # d["WAVE"] is in nm
        # let s convert to A, and interpolate on the DESI waves
        myfilts[band] = {}
        myfilts[band]["TRANS"] = np.interp(
            waves, d["WAVE"], d["TRANSMISSION"], left=0, right=0
        )

    return myfilts


def get_speclite_filts(bands, waves):

    myfilts = get_filts(bands, waves)

    # use band.lower() because speclite doesn t like e.g. "I427"...
    tmp_speclite_dir = tempfile.mkdtemp()

    for band in bands:

        tmp_filt = speclite_filters.FilterResponse(
            wavelength=waves * units.Angstrom,
            response=myfilts[band]["TRANS"],
            meta=dict(group_name="suprime", band_name=band.lower()),
        )
        tmp_name = tmp_filt.save(tmp_speclite_dir)

    # now read them in
    speclite_filts = {
        band: speclite_filters.load_filter("suprime-{}".format(band.lower()))
        for band in bands
    }

    return speclite_filts


# d: CUSTOM
def get_spectro_flux(bands, d, waves, fluxs):

    speclite_filts = get_speclite_filts(bands, waves)

    nspec = fluxs.shape[0]
    spec_mags = {band: np.zeros(nspec) for band in bands}

    for i in range(nspec):
        tmpfs = fluxs[i].copy()
        tmpfs *= 1e-17
        tmpfs *= d["MEAN_PSF_TO_FIBER_SPECFLUX"][i]

        for band in bands:

            spec_mags[band][i] = speclite_filts[band].get_ab_magnitude(
                tmpfs * units.erg / (units.cm**2 * units.s * units.Angstrom),
                waves * units.Angstrom,
            )

    spec_fluxs = {band: 10 ** (-0.4 * (spec_mags[band] - 22.5)) for band in bands}

    return spec_fluxs


def process_fn(d, t, rrfn, bands):

    cofn = rrfn.replace("redrock", "coadd")
    s = read_spectra(cofn)
    fm = s.fibermap
    sel = d["FN"] == rrfn
    log.info("{}\t{}".format(sel.sum(), rrfn))

    if sel.sum() == 0:

        msg = "{}\tsel.sum() = 0; this shouldn t happen".format(rrfn)
        log.error(msg)
        raise ValueError(msg)

    s = s.select(targets=d["TARGETID"][sel])
    assert np.all(s.fibermap["TARGETID"] == d["TARGETID"][sel])
    s = coadd_cameras(s)
    myd, myt = d[sel], t[sel]

    spec_fluxs = get_spectro_flux(bands, d, s.wave["brz"], s.flux["brz"])

    for band in bands:

        myd["SPECTRO_FIBERTOTFLUX_{}".format(band)] = spec_fluxs[band]

    return (s, myd, myt)


# d: CUSTOM table
# t: PHOTINFO table
def get_unq_spectra(d, t, waves, fluxs, ivars, bands):

    orig_d, orig_t = d.copy(), t.copy()
    orig_fluxs, orig_ivars = fluxs.copy(), ivars.copy()

    # cut on COADD_FIBERSTATUS = 0
    sel = d["COADD_FIBERSTATUS"] == 0
    d, t = d[sel], t[sel]
    fluxs, ivars = fluxs[sel, :], ivars[sel, :]

    #
    unq_tids, ii, counts = np.unique(
        d["TARGETID"], return_index=True, return_counts=True
    )
    ntid = unq_tids.size

    unq_d = Table()

    for key in ["TARGETID", "TARGET_RA", "TARGET_DEC", "COADD_FIBERSTATUS"]:

        unq_d[key] = d[key][ii]

    unq_d["NSPEC"] = counts
    unq_d["MEAN_PSF_TO_FIBER_SPECFLUX"] = 0.0
    tsnr2_keys = [key for key in d.colnames if "TSNR2" in key]

    for key in tsnr2_keys:

        unq_d[key] = 0.0

    unq_t = Table()

    for key in t.colnames:

        if len(t[key].shape) == 1:

            unq_t[key] = t[key][ii]

        if len(t[key].shape) == 2:

            unq_t[key] = t[key][ii, :]

    nwave = waves.size
    unq_fluxs = np.zeros((ntid, nwave), dtype=float)
    unq_ivars = np.zeros((ntid, nwave), dtype=float)

    # rows with counts = 1
    sel = counts == 1
    unq_fluxs[sel, :] = fluxs[ii[sel], :]
    unq_ivars[sel, :] = ivars[ii[sel], :]
    unq_d["MEAN_PSF_TO_FIBER_SPECFLUX"][sel] = d["MEAN_PSF_TO_FIBER_SPECFLUX"][ii[sel]]

    for key in tsnr2_keys:

        unq_d[key][sel] = d[key][ii[sel]]

    # rows with counts > 1
    # tsnr2: summing

    for i in np.where(counts > 1)[0]:

        jj = np.where(d["TARGETID"] == unq_d["TARGETID"][i])[0]
        log.info("{}\t{}".format(jj.size, unq_d["TARGETID"][i]))

        for j in jj:

            unq_fluxs[i] += fluxs[j] * ivars[j]
            unq_ivars[i] += ivars[j]

        unq_fluxs[i] /= unq_ivars[i]
        unq_d["MEAN_PSF_TO_FIBER_SPECFLUX"][i] = d["MEAN_PSF_TO_FIBER_SPECFLUX"][
            jj
        ].mean()

        for key in tsnr2_keys:

            unq_d[key][i] = d[key][jj].sum()

    # SPECTRO_FIBERTOTFLUX
    spec_fluxs = get_spectro_flux(bands, unq_d, waves, unq_fluxs)

    for band in bands:

        unq_d["SPECTRO_FIBERTOTFLUX_{}".format(band)] = spec_fluxs[band]

    d, t = orig_d, orig_t
    fluxs, ivars = orig_fluxs, orig_ivars

    return unq_fluxs, unq_ivars, unq_d, unq_t


def build_hs(s, d, t):

    hs = fits.HDUList()

    # header
    h = fits.PrimaryHDU()
    hs.append(h)

    # images
    for ext, bunit in zip(
        ["wave", "flux", "ivar"],
        [
            "Angstrom",
            "10**-17 erg/(s cm2 Angstrom)",
            "10**+34 (s2 cm4 Angstrom2) / erg2",
        ],
    ):

        h = fits.ImageHDU(name="BRZ_{}".format(ext.upper()))

        if bunit is not None:

            h.header["BUNIT"] = bunit

        h.data = eval("s.{}['brz']".format(ext))
        hs.append(h)

    # tables
    for extd, extname in zip(
        [s.fibermap, s.scores, d, t],
        ["FIBERMAP", "SCORES", "CUSTOM", "PHOTINFO"],
    ):

        h = fits.convenience.table_to_hdu(extd)
        h.header["EXTNAME"] = extname
        hs.append(h)

    return hs


def build_unq_hs(waves, unq_fluxs, unq_ivars, unq_d, unq_t):

    hs = fits.HDUList()

    # header
    h = fits.PrimaryHDU()
    hs.append(h)

    # images
    for ext, extd, bunit in zip(
        ["wave", "flux", "ivar"],
        [waves, unq_fluxs, unq_ivars],
        [
            "Angstrom",
            "10**-17 erg/(s cm2 Angstrom)",
            "10**+34 (s2 cm4 Angstrom2) / erg2",
        ],
    ):

        h = fits.ImageHDU(name="BRZ_{}".format(ext.upper()))

        if bunit is not None:

            h.header["BUNIT"] = bunit

        h.data = extd
        hs.append(h)

    # tables
    for extd, extname in zip(
        [unq_d, unq_t],
        ["CUSTOM", "PHOTINFO"],
    ):

        h = fits.convenience.table_to_hdu(extd)
        h.header["EXTNAME"] = extname
        hs.append(h)

    return hs
