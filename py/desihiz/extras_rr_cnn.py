#!/usr/bin/env python

import os
from time import time
from glob import glob
import multiprocessing
import fitsio
import numpy as np
from astropy.table import Table, vstack

from desitarget.geomask import match, match_to
from redrock.results import read_zscan
from desiutil.log import get_logger

from desihiz.extras_utils import wave_lya, wave_oii

log = get_logger()

wave_lya = 1215.67
wave_oii = 0.5 * (3726.1 + 3728.8)


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
        np.abs((1 + d["ALL_Z"][:, 0]) * wave_oii / wave_lya - 1 - d["ALL_Z"][:, 1])
        < 0.01
    )
    d["Z_01_BEST"][sel] = d["ALL_Z"][:, 1][sel]

    return d


def get_cnnkeys():

    keys = [
        "Z",
        "CL",
    ]

    return keys


def get_cnn(tids, cnnfn):

    d = Table()
    d["CL"] = -99.0 + np.zeros(len(tids))
    d["Z"] = -99.0

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
    log.info(
        "{}/{} TARGETIDs are in the {} {} files".format(
            ii.size, len(d), len(fns), pattern
        )
    )
    d["Z"][ii] = rr["Z"][iirr]

    return d
