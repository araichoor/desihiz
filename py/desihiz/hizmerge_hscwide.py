#!/usr/bin/env python


import os
from glob import glob
import fitsio
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units
from desitarget.targetmask import desi_mask
from desiutil.log import get_logger
from desihiz.hizmerge_io import (
    get_img_dir,
    match_coord,
    get_init_infos,
    get_phot_fns,
)

log = get_logger()


def get_hscwide_cosmos_yr3_infos():
    """
    Get the minimal photometric infos for HSC targets from tertiary37

    Args:
        None

    Returns:
        mydict: dictionary with {keys: arrays},
            with keys: TARGETID, TERTIARY_TARGET, PHOT_RA, PHOT_DEC, PHOT_SELECTION
    """
    #
    fadir = os.path.join(
        os.getenv("DESI_ROOT"), "survey", "fiberassign", "special", "tertiary", "0037"
    )

    # first read the tertiary37 file
    fn = os.path.join(fadir, "tertiary-targets-0037-assign.fits")
    d = Table.read(fn)

    # cut on hsc targets
    sel = (d["LBG_HSC_NEW"]) | (d["LBG_HSC_REOBS"])
    d = d[sel]

    # get the row in the target file
    assert np.all((d["LBG_HSC_NEW_ROW"] == -99) | (d["LBG_HSC_REOBS_ROW"] == -99))
    rows = d["LBG_HSC_NEW_ROW"].copy()
    sel = d["LBG_HSC_NEW_ROW"] == -99
    rows[sel] = d["LBG_HSC_REOBS_ROW"][sel]

    # now read targets file
    fn = os.path.join(fadir, "inputcats", "cosmos_yr3-lbg-hsc.fits")
    t = Table.read(fn)

    # match t to d
    t = t[rows]

    # sanity check
    # - for TERTIARY_TARGET=LBG_HSC_NEW,LBG_HSC_REOBS: (ra, dec) should be exactly the same
    # - for TERTIARY_TARGET=LBG_SUPRIME_NEW,LBG_SUPRIME_2H_NEW : (ra, dec) are those from higher priority catalog
    #       and should be within 1 arcsec
    sel = (d["TERTIARY_TARGET"] == "LBG_HSC_NEW") | (
        d["TERTIARY_TARGET"] == "LBG_HSC_REOBS"
    )
    assert np.all(
        (d["TERTIARY_TARGET"][~sel] == "LBG_SUPRIME_NEW")
        | (d["TERTIARY_TARGET"][~sel] == "LBG_SUPRIME_2H_NEW")
    )
    dcs = SkyCoord(d["RA"] * units.degree, d["DEC"] * units.degree, frame="icrs")
    tcs = SkyCoord(t["RA"] * units.degree, t["DEC"] * units.degree, frame="icrs")
    seps = dcs.separation(tcs).to("arcsec").value
    assert np.all(seps[sel] == 0)
    assert np.all(seps[~sel] < 1)

    # now, t and d are row-matched
    nrows = [len(t)]
    mydict = get_init_infos("hscwide", nrows)

    for band in ["GRIZ"]:

        mydict[band]["TARGETID"] = d["TARGETID"]
        mydict[band]["TERTIARY_TARGET"] = d["TERTIARY_TARGET"]
        mydict[band]["PHOT_RA"] = t["RA"]
        mydict[band]["PHOT_DEC"] = t["DEC"]
        mydict[band]["PHOT_SELECTION"] = t["SAMPLE"].astype(
            mydict[band]["PHOT_SELECTION"].dtype
        )

    return mydict


# get photometry infos (targetid, brickname, objid)
# this is for hscwide targets only
# sky/std will have dummy values
def get_hscwide_phot_infos(case, d, photdir=None):
    """
    Get the photometric information (TARGETID, BRICKNAME, OBJID) for a given case

    Args:
        case: round of DESI observation (str)
        d: output of the get_spec_table() function
        photdir (optional, defaults to $DESI_ROOT/users/raichoor/laelbg/{img}/phot):
            folder where the files are
    """
    if photdir is None:

        photdir = os.path.join(get_img_dir("odin"), "phot")

    # initialize columns we will fill
    objids = np.zeros(len(d), dtype=int)
    targfns = np.zeros(len(d), dtype="S100")

    # now get the per-band phot. infos
    for band in ["GRIZ"]:

        ii_band = np.where(d[band])[0]
        fns = get_phot_fns("hscwide", case, band, photdir=photdir)
        log.info("{}\t{}\t{}\t{}".format(case, band, ii_band.size, fns))

        # is that band relevant for that case?
        if fns is None:

            continue

        for fn in fns:

            # indexes:
            # - targets selected with that band
            # - not dealt with yet (by previous fn)
            sel = (d[band]) & (objids == 0)

            ii_band = np.where(sel)[0]
            log.info(
                "{}\t{}\t{}\t{}/{} targets not dealt with yet".format(
                    case,
                    band,
                    os.path.basename(fn),
                    ii_band.size,
                    d[band].sum(),
                )
            )

            t = Table.read(fn, unit_parse_strict="silent")

            for key in t.colnames:

                t[key].name = t[key].name.upper()

            iid, iit, d2d, _, _ = match_coord(
                d["PHOT_RA"][ii_band],
                d["PHOT_DEC"][ii_band],
                t["RA"],
                t["DEC"],
                search_radius=1.0,
                verbose=True,
            )
            log.info(
                "{}\t{}\t{:04d}/{:04d}\t{}\t{}".format(
                    case,
                    band,
                    iid.size,
                    ii_band.size,
                    (d2d != 0).sum(),
                    os.path.basename(fn),
                )
            )

            # verify we have a "perfect" match, except for targets
            #   which also are in higher-priority samples
            #   but have ra,dec from another imaging than hscwide
            sel_diff = d2d != 0

            if sel_diff.sum() != 0:

                log.info(
                    "{}\t{}\tlooking at {}/{} rows with d2d!=0".format(
                        case, band, sel_diff.sum(), d2d.size
                    )
                )
                tertiary_targets = d["TERTIARY_TARGET"][ii_band][iid][sel_diff].astype(
                    str
                )
                unq_tertiary_targets, counts = np.unique(
                    tertiary_targets, return_counts=True
                )
                txt = ", ".join(
                    [
                        "{}={}".format(unq_tertiary_target, count)
                        for unq_tertiary_target, count in zip(
                            unq_tertiary_targets, counts
                        )
                    ]
                )
                log.info("{}\t{}\tthose come from: {}".format(case, band, txt))

                hip_targs = {
                    "COSMOS_YR3_GRIZ": [
                        "LBG_SUPRIME_NEW",
                        "LBG_SUPRIME_2H_NEW",
                    ],
                }

                photnames = d["PHOT_SELECTION"][ii_band][iid][sel_diff].astype(str)
                ischecked = np.zeros(len(tertiary_targets), dtype=bool)
                for photname in list(hip_targs.keys()):
                    sel = np.array([photname in _ for _ in photnames])
                    log.info(
                        "{}\t{}\t{} in PHOT_SELECTION\thip_targs={}\tnp.unique(tertiary_targets)={}".format(
                            case,
                            band,
                            photname,
                            hip_targs[photname],
                            np.unique(tertiary_targets[sel]).tolist(),
                        )
                    )
                    assert np.all(np.in1d(tertiary_targets[sel], hip_targs[photname]))
                    ischecked[sel] = True
                assert np.all(ischecked)
                log.info(
                    "{}\t{}\tall the {}/{} d2d!=0 are due to higher-priority targets".format(
                        case, band, sel_diff.sum(), d2d.size
                    )
                )
            # fill the values
            iid = ii_band[iid]
            objids[iid] = t["OBJECT_ID"][iit]
            targfns[iid] = fn

        # verify all objects are matched
        assert ((d[band]) & (objids == 0)).sum() == 0

    return objids, targfns
