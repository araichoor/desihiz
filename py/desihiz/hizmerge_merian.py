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
from desitarget.geomask import match, match_to
from desiutil.log import get_logger
from desihiz.hizmerge_io import (
    get_img_dir,
    get_img_bands,
    match_coord,
    get_init_infos,
    get_phot_fns,
)

log = get_logger()


def get_merian_cosmos_yr2_infos():
    """
    Get the minimal photometric infos for MERIAN N540 targets from tertiary23

    Args:
        None

    Returns:
        mydict: dictionary with {keys: arrays},
            with keys: TARGETID, TERTIARY_TARGET, PHOT_RA, PHOT_DEC, PHOT_SELECTION
    """
    #
    fadir = os.path.join(
        os.getenv("DESI_ROOT"), "survey", "fiberassign", "special", "tertiary", "0023"
    )
    bands = get_img_bands("merian")

    # first read the tertiary23 file
    fn = os.path.join(fadir, "tertiary-targets-0023-assign.fits")
    d = Table.read(fn)

    # cut on MS3 targets
    sel = d["TERTIARY_TARGET"] == "MERIAN_MS3"
    d = d[sel]
    assert np.all(d["ORIG_FN"] == "inputcats/COSMOS_Merian_MS3.txt")

    # all the MS3 targets are N540
    d["N540"] = True

    # get the row in the target file
    rows = d["ORIG_ROW"].copy()

    # now read targets file
    fn = os.path.join(fadir, "inputcats", "COSMOS_Merian_MS3.txt")
    t = Table.read(fn, format="ascii.commented_header")

    # match t to d
    t = t[rows]

    # sanity check:
    assert np.all(d["RA"] == t["ALPHA_J2000"])
    assert np.all(d["DEC"] == t["DELTA_J2000"])

    # initialize (with grabbing correct datamodel)
    tmpdict = get_init_infos("merian", [len(t)])[bands[0]]

    for key in ["PHOT_RA", "PHOT_DEC", "PHOT_SELECTION"]:

        d[key] = tmpdict[key]

    # get merian ra, dec
    d["PHOT_RA"], d["PHOT_DEC"] = t["ALPHA_J2000"], t["DELTA_J2000"]

    # get band + selection
    d["PHOT_SELECTION"] = "MERIAN_MS3"

    # sanity check
    ## all rows are filled
    assert np.all(d["TERTIARY_TARGET"] != 0)
    assert np.all(d["PHOT_RA"] != 0)
    assert np.all(d["PHOT_DEC"] != 0)

    #
    mydict = get_init_infos("merian", [d[band].sum() for band in bands])

    for band in bands:

        sel = d[band]
        mydict[band]["TARGETID"] = d["TARGETID"][sel]
        mydict[band]["TERTIARY_TARGET"] = d["TERTIARY_TARGET"][sel]
        mydict[band]["PHOT_RA"] = d["RA"][sel]
        mydict[band]["PHOT_DEC"] = d["DEC"][sel]
        mydict[band]["PHOT_SELECTION"] = d["PHOT_SELECTION"][sel].astype(
            mydict[band]["PHOT_SELECTION"].dtype
        )

    ## check
    for band in bands:

        names, counts = np.unique(d["TERTIARY_TARGET"][d[band]], return_counts=True)
        log.info(
            "{} ({}):\t{}".format(
                band,
                d[band].sum(),
                ", ".join(
                    ["{}={}".format(name, count) for name, count in zip(names, counts)]
                ),
            )
        )
    return mydict


# get photometry infos (targetid, index_lae)
# this is for hscwide targets only
# sky/std will have dummy values
def get_merian_phot_infos(case, d, photdir=None):
    """
    Get the photometric information (TARGETID, INDEX_LAE) for a given case

    Args:
        case: round of DESI observation (str)
        d: output of the get_spec_table() function
        photdir (optional, defaults to $DESI_ROOT/users/raichoor/laelbg/{img}/phot):
            folder where the files are
    """
    if photdir is None:

        photdir = os.path.join(get_img_dir("odin"), "phot")

    # AR add INDEX_LAE to d
    # AR kind of duplicating what s in get_merian_cosmos_yr2_infos()...
    if case == "cosmos_yr2":
        bands = get_img_bands("merian")
        assert len(bands) == 1
        band = bands[0]
        #
        fadir = os.path.join(
            os.getenv("DESI_ROOT"), "survey", "fiberassign", "special", "tertiary", "0023"
        )
        fn = os.path.join(fadir, "tertiary-targets-0023-assign.fits")
        t = Table.read(fn)
        # AR cut on MS3 targets
        sel = t["TERTIARY_TARGET"] == "MERIAN_MS3"
        t = t[sel]
        assert np.all(t["ORIG_FN"] == "inputcats/COSMOS_Merian_MS3.txt")
        rows = t["ORIG_ROW"]
        # AR now read targets file
        fn = os.path.join(fadir, "inputcats", "COSMOS_Merian_MS3.txt")
        t2 = Table.read(fn, format="ascii.commented_header")
        # AR match t2 to t, to have the TARGETID<=>INDEX_LAE
        t2 = t2[rows]
        assert np.all(t["RA"] == t2["ALPHA_J2000"])
        assert np.all(t["DEC"] == t2["DELTA_J2000"])
        t["INDEX_LAE"] = t2["ID"]
        # AR now add INDEX_LAE
        iit = match_to(t["TARGETID"], d["TARGETID"])
        assert np.all(t["TARGETID"][iit] == d["TARGETID"])
        merianids = t["INDEX_LAE"][iit]
        targfns = np.array([get_phot_fns("merian", case, band, photdir=photdir)], dtype="S100")

    # verify all objects are matched
    assert ((d[band]) & (merianids == 0)).sum() == 0

    merianids = merianids.astype(str)

    return merianids, targfns
