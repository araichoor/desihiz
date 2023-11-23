#!/usr/bin/env python


import os
import fitsio
import numpy as np
from astropy.io import fits
from astropy.table import Table
from desiutil.log import get_logger
from desihiz.hizmerge_io import (
    allowed_imgs,
    get_img_bands,
    get_phot_fns,
    get_ext_coeffs,
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
    """

    assert img in allowed_imgs
    assert band in get_img_bands(img)

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
