#!/usr/bin/env python

"""
# all cuts:  see David s email from 10/11/23, 11:37 AM pacific

# in the script, we use isbug=True,False to identify:
# - either the original, bugged, photometry used for target selection
# - or the correct photometry (from rerunning tractor with fixing the bugs)
"""

import os
import numpy as np
from astropy.table import Table
from desiutil.log import get_logger
from desihiz.hizmerge_io import get_img_bands
from desihiz.suprime_analysis import (
    get_radecbox_area,
    get_tractor_band,
    get_bug2ok_mags,
    get_selection_selbands,
    get_selection_maglims,
    get_selection_keys_thresholds,
)

log = get_logger()


djs_selbands = get_selection_selbands("djs")


def get_djscol_blue_red(djscol):

    if djscol == "DJS_COL2":

        return "I427", "I464"

    if djscol == "DJS_COL3":

        return "I464", "I484"

    if djscol == "DJS_COL4":

        return "I484", "I505"

    if djscol == "DJS_COL5":

        return "I505", "I527"


# use here original offsets (applied at the ts level
#   on the "bugged" catalogs),
#   plus an additional offset to mimick the
#   "bugged" offsets
# see header of Subaru_tractor_forced_all-redux-20231025-djs.fits
# DJS_COL = BLUE - RED + OFFSET
def get_djscol_offset(djscol, isbug):

    if djscol == "DJS_COL2":

        offset = +0.56

    if djscol == "DJS_COL3":

        offset = -0.01

    if djscol == "DJS_COL4":

        offset = -0.035

    if djscol == "DJS_COL5":

        offset = +0.013

    if not isbug:

        blue, red = get_djscol_blue_red(djscol)
        bug2ok_mags = get_bug2ok_mags()
        offset -= bug2ok_mags[blue] - bug2ok_mags[red]

    offset = np.round(offset, 3)
    log.info("(djscol, isbug, offset) = ({}, {}, {:.3f})".format(djscol, isbug, offset))

    return offset


def add_djscols(d, isbug):

    d["RI_COL"] = d["MAG_R"] - d["MAG_I"]

    for djscol in ["DJS_COL2", "DJS_COL3", "DJS_COL4", "DJS_COL5"]:

        blue, red = get_djscol_blue_red(djscol)
        d[djscol] = d["MAG_{}".format(blue)] - d["MAG_{}".format(red)]
        zp = get_djscol_offset(djscol, isbug)
        d[djscol] += zp

        if zp > 0:

            d.meta[djscol] = "MAG_{} - MAG_{} + {}".format(blue, red, zp)

        else:

            d.meta[djscol] = "MAG_{} - MAG_{} - {}".format(blue, red, -zp)

    # colors of djscols...
    for ij in [
        "3M2",
        "4M3",
        "4M2",
        "5M4",
        "5M3",
        "5M2",
    ]:

        i, j = ij.split("M")
        d["DJS_COL{}".format(ij)] = d["DJS_COL{}".format(i)] - d["DJS_COL{}".format(j)]

    return d


# 5-sigma psfdepths
def get_djs_depthlims():

    return {
        "I427": 24.9,
        "I464": 24.9,
        "I484": 25.8,
        "I505": 25.2,
        "I527": 26.0,
        "R": 26.0,
        "I": 26.0,
    }


def add_djs_quality_msk(d, maxsize=1.0, rands=False):

    ramin, ramax, decmin, decmax, _ = get_radecbox_area()

    isqual = (d["RA"] > ramin) & (d["RA"] < ramax)
    isqual &= (d["DEC"] > decmin) & (d["DEC"] < decmax)
    isqual &= d["MASKBITS"] == 0

    depthlims = get_djs_depthlims()
    suprime_bands = get_img_bands("suprime")

    for band in suprime_bands + ["R", "I"]:

        isqual &= d["DEPTH_{}".format(band)] > depthlims[band]

    if not rands:

        isqual &= d["SHAPE_R"] < maxsize

    d["DJS_QUALITY_MSK"] = isqual

    return d


def add_djs_parents(d, isbug, mrmin=23.0, snrmin=8.0, maxsize=1.0):

    d = add_djs_quality_msk(d, maxsize=maxsize)
    maglims = get_selection_maglims("djs", isbug)

    for band in djs_selbands:

        tband = get_tractor_band(band)

        # parent sample
        isparent = d["DJS_QUALITY_MSK"].copy()
        isparent &= d["FRACFLUX_{}".format(tband)] < 0.15
        isparent &= d["MAG_{}".format(band)] < maglims[band]
        isparent &= d["FIBMAG_{}".format(band)] < maglims[band] + 0.25
        isparent &= d["MAG_R"] > mrmin
        isparent &= d["SNR_{}".format(band)] > snrmin
        isparent &= (d["RCHISQ_{}".format(tband)] > 0) & (
            d["RCHISQ_{}".format(tband)] < 2.0
        )
        d["PARENT_{}".format(band)] = isparent

        log.info(
            "isbug={}\tPARENT_{}={}".format(
                isbug,
                band,
                d["PARENT_{}".format(band)].sum(),
            )
        )

    return d


def add_djs_sels(d, isbug, mrmin=23.0, snrmin=8.0, maxsize=1.0):

    for band in djs_selbands:

        # selection
        keys, thresholds = get_selection_keys_thresholds("djs", band)
        log.info("band={}\tkeys={}\tthresholds={}".format(band, keys, thresholds))
        d["SEL_{}".format(band)] = d["PARENT_{}".format(band)].copy()

        for key, threshold in zip(keys, thresholds):

            d["SEL_{}".format(band)] &= d[key] > threshold

        if band == "I464":

            d["SEL_{}".format(band)] &= d["RI_COL"] < -0.1 + 0.4 * d["DJS_COL2"]

        # ri cut
        d["SEL_{}".format(band)] &= d["RI_COL"] > -0.4

        if band == "I464":

            rimax = 0.40

        else:

            rimax = 0.45

        d["SEL_{}".format(band)] &= d["RI_COL"] < rimax

        log.info(
            "isbug={}\tSEL_{}={}".format(
                isbug,
                band,
                d["SEL_{}".format(band)].sum(),
            )
        )

    return d
