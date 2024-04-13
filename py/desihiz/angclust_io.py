#!/usr/bin/env python


import os
import fitsio
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from desitarget.geomask import match
from desiutil.log import get_logger
from desihiz.hizmerge_io import (
    allowed_imgs,
    get_img_bands,
    get_phot_fns,
    get_ext_coeffs,
    match_coord,
)

log = get_logger()

allowed_fields = ["cosmos", "xmmlss"]


def get_angclust_targs(img, band):
    """
    Create the catalog for angular clustering used by MJW

    Args:
        img: element from allowed_imgs (str)
        band: filter to consider (str)

    Returns:
        d: Table() with the selected photometric targets (astropy Table())

    Notes:
        ODIN/N501:
            - reproducing AD "Selection3" sample
            - tested on Nov. 16, 2023 against:
                fn = "/pscratch/sd/m/mwhite/AnalyzeLAE/odin/odin_lae_cosmos_N501.fits"
                d = Table.read(fn)
                d = d[d["Selection3"] == 1]
        ODIN/N419 (COSMOS):
            - reproducing AD "Selection3" sample
            - there are two bb img sources, HSC and LS: for simplicity, for targets selected
                by both imaging, we only report the HSC photometry.

    """

    assert img in allowed_imgs
    assert band in get_img_bands(img)

    # odin/n419/cosmos
    if (img == "odin") & (band == "N419"):

        # for odin/n419, we only use cosmos_yr2 so far (not xmmlss_yr2)
        # we have two tractor catalogs: HSC and LSDR10
        # note that the {brickname,objid} are the same, as the HSC/LSDR10
        #   difference is for the forced photometry

        # first read the targets
        fadir = os.path.join(
            os.getenv("DESI_ROOT"), "survey", "fiberassign", "special", "tertiary", "0026"
        )
        fn = os.path.join(fadir, "inputcats", "COSMOS_LAE_Candidates_2023apr04v2.fits.gz")
        t = Table(fitsio.read(fn))
        sel = np.array(["LAE N419" in _ for _ in t["SELECTION"]])
        t = t[sel]

        # now loop on the HSC/LSDR10 catalogs
        fns = get_phot_fns(img, "cosmos_yr2", band)
        ds = {}

        for fn in fns:

            # HSC or LS ?
            if os.path.basename(fn) == "ODIN_N419_tractor_HSC_forced_all.fits.gz":
                bb_img = "HSC"
            elif os.path.basename(fn) == "ODIN_N419_tractor_DR10_forced_all.fits.gz":
                bb_img = "LS"
            else:
                msg = "unrecognized fn = {}".format(fn)
                log.error(msg)
                raise ValueError(msg)

            # cut t on bb_img
            sel = np.array(["LAE N419 ODIN+{}".format(bb_img) in _ for _ in t["SELECTION"]])
            tcut = t[sel]
            log.info("select {} targets for {}".format(sel.sum(), bb_img))

            # read the tractor file
            d = Table(fitsio.read(fn))
            for key in d.colnames:
                d[key].name = key.upper()
            d["UNQID"] = ["{}-{}".format(b, o) for b, o in zip(d["BRICKNAME"], d["OBJID"])]
            log.info("read {} rows from {}".format(len(d), fn))

            # cross-match with the targets
            # we want an exact match for all targets..
            iid, iit, _, _, d2d = match_coord(d["RA"], d["DEC"], tcut["RA"], tcut["DEC"], search_radius=1.0)
            assert iid.size == len(tcut)
            assert np.all(d2d == 0.)
            ds[bb_img] = d[iid]

        # AR sanity check
        assert np.unique(ds["HSC"]["UNQID"]).size == len(ds["HSC"])
        assert np.unique(ds["LS"]["UNQID"]).size == len(ds["LS"])
        iih, iil = match(ds["HSC"]["UNQID"], ds["LS"]["UNQID"])
        assert np.abs(ds["HSC"]["RA"][iih] - ds["LS"]["RA"][iil]).max() == 0
        assert np.abs(ds["HSC"]["DEC"][iih] - ds["LS"]["DEC"][iil]).max() == 0

        # AR add dummy columns for r2, i2 for LS
        assert len([key for key in ds["LS"].colnames if key not in ds["HSC"].colnames]) == 0
        keys = [key for key in ds["HSC"].colnames if key not in ds["LS"].colnames]
        log.info("add following keys to ds['LS']: {}".format(",".join(keys)))
        for key in keys:
            shape = list(ds["HSC"][key].shape)
            shape[0] = len(ds["LS"])
            shape = tuple(shape)
            ds["LS"][key] = np.zeros_like(ds["HSC"][key], shape=shape)

        # AR same key ordering, for vstack
        ds["LS"] = ds["LS"][ds["HSC"].colnames]

        # AR stack, start with HSC
        # AR consequence: targets selected both by HSC and LS will
        # AR    have the HSC photometry reported
        d = ds["HSC"].copy()
        d["HSC"], d["LS"] = True, False
        sel = np.in1d(d["UNQID"], ds["LS"]["UNQID"])
        d["LS"][sel] = True
        sel = ~np.in1d(ds["LS"]["UNQID"], d["UNQID"])
        d2 = ds["LS"][sel]
        d2["HSC"], d2["LS"] = False, True
        d = vstack([d, d2])

        # mags
        ext_coeffs = get_ext_coeffs(img)
        n419_mags = (
            22.5
            - 2.5 * np.log10(d["FLUX_N419"])
            - ext_coeffs["ODIN"]["N419"] * d["EBV"]
        )
        n501_mags = (
            22.5
            - 2.5 * np.log10(d["FLUX_N501"])
            - ext_coeffs["ODIN"]["N501"] * d["EBV"]
        )

        # cut on AD "Selection3" [email from 3/15/24, 1:34 PM]
        minmag = 19.0
        excess = -0.75
        maglim = 24.906 # corresponding to 5e-17 erg/s/cm^2 line flux
        cc = [-60.5, 5.5, -0.125]
        c12 = n419_mags - n501_mags
        isel_a = (n419_mags <= maglim) & (n419_mags >= minmag)
        isel_b = (d["FLUX_N501"] > 0) & (c12 <= excess)
        isel_c = (d["FLUX_N501"] > 0) & (c12 <= cc[0] + cc[1] * n419_mags + cc[2] * n419_mags ** 2 - 0.4)
        isel_d = (d["FLUX_N501"] > 0) & (c12 <= -16.375 + 0.6875 * n419_mags)
        isel_e = (d["FLUX_N501"] > 0) & (c12 <= 17.27 - 0.75 * n419_mags)

        sel = (isel_a) & (isel_b) & (isel_d) & (isel_e)
        sel |= d["FLUX_N501"] <= 0

        d = d[sel]

        log.info("select {}/{} rows".format(sel.sum(), sel.size))

    # odin/n501
    if (img == "odin") & (band == "N501"):

        # for odin/n501, cosmos_yr1 and cosmos_yr2 used the same target catalog
        fn = get_phot_fns(img, "cosmos_yr1", band)[0]
        d = Table(fitsio.read(fn))
        log.info("read {} rows from {}".format(len(d), fn))

        # mags
        ext_coeffs = get_ext_coeffs(img)
        n501_mags = (
            22.5
            - 2.5 * np.log10(d["FLUX_N501"])
            - ext_coeffs["ODIN"]["N501"] * d["EBV"]
        )
        n673_mags = (
            22.5
            - 2.5 * np.log10(d["FLUX_N673"])
            - ext_coeffs["ODIN"]["N673"] * d["EBV"]
        )

        # cut on AD "Selection3"
        cc = [-60.5, 5.5, -0.125]
        cc2 = [-16.375, 0.6875]
        sel = d["PRIORITY"] == 1
        sel &= n501_mags >= 18
        sel &= (
            n501_mags - n673_mags <= cc[0] + cc[1] * n501_mags + cc[2] * n501_mags**2
        )
        sel &= n501_mags - n673_mags <= cc2[0] + cc2[1] * n501_mags

        d = d[sel]

        log.info("select {}/{} rows".format(sel.sum(), sel.size))

    return d


def add_vi(d, img, band, mergefn):
    """
    Adds the VI informations.

    Args:
        d: Table() with the selected photometric targets (astropy Table())
        img: element from allowed_imgs (str)
        band: filter to consider (str)
        mergefn: full path to the desi-{img}.fits file (str)

    Returns:
        d: same as input, but with additional VI columns
            (VI, VI_Z, VI_QUALITY, VI_SPECTYPE)

    Notes:
        A MERGEFN keyword is also added to d.meta.
    """

    assert img in allowed_imgs
    assert band in get_img_bands(img)

    # unqids: for matching
    # pextname: for suprime, we will likely use PHOTV2INFO
    if img == "odin":
        pextname = "PHOTINFO"
        unqids = np.array(
            ["{}-{}".format(b, o) for b, o in zip(d["BRICKNAME"], d["OBJID"])]
        )

    # read vi info
    s = Table(fitsio.read(mergefn, "SPECINFO"))
    p = Table(fitsio.read(mergefn, pextname))
    log.info("read {} rows from {}".format(len(s), mergefn))

    # cut on VI-ed rows + some per-img custom cuts
    # + rename s["TARGETID"] => s["VI_TARGETID"]
    sel = s["VI"]
    sel &= s[band]
    if img == "odin":
        sel &= np.in1d(s["CASE"].astype(str), ["cosmos_yr1", "cosmos_yr2"])
    s, p = s[sel], p[sel]
    s["TARGETID"].name = "VI_TARGETID"
    log.info(
        "cut on {}/{} rows with VI + some per-img custom cuts".format(
            sel.sum(), sel.size
        )
    )

    # unqids
    if img == "odin":
        p["UNQID"] = np.array(
            ["{}-{}".format(b, o) for b, o in zip(p["BRICKNAME"], p["OBJID"])]
        )

    # initialize columns to fill
    keys = ["VI", "VI_TARGETID", "VI_Z", "VI_QUALITY", "VI_SPECTYPE", "EFFTIME_SPEC"]
    for key in keys:
        d[key] = np.zeros_like(s[key], shape=(len(d),))

    # fill (not many rows, so just brutal loop, no fancy method...)
    for i, unqid in enumerate(unqids):
        jj = np.where(p["UNQID"] == unqid)[0]
        assert jj.size <= 1
        if jj.size == 1:
            j = jj[0]
            for key in keys:
                d[key][i] = s[key][j]
    log.info("filled {}/{} rows with VI infos".format(d["VI"].sum(), len(d)))

    # add info in header
    d.meta["MERGEFN"] = mergefn

    return d
