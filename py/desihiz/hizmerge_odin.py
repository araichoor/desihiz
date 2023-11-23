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


def get_odin_cosmos_yr1_infos():
    """
    Get the minimal photometric infos for ODIN cosmos_yr1 (TILEID=82636)

    Args:
        None

    Returns:
        mydict: dictionary with {keys: arrays},
            with keys: TARGETID, TERTIARY_TARGET, PHOT_RA, PHOT_DEC, PHOT_SELECTION
    """
    #
    fadir = os.path.join(
        os.getenv("DESI_ROOT"), "survey", "fiberassign", "special", "20220324"
    )
    t501 = Table.read(os.path.join(fadir, "LAE_Candidates_NB501_v1_targeting.fits.gz"))
    t673 = Table.read(os.path.join(fadir, "LAE_Candidates_NB673_v0_targeting.fits.gz"))
    d = Table.read(os.path.join(fadir, "ToO.ecsv"))

    # n419 (no targets)

    # n501
    iit501, iid501, d2d501, _, _ = match_coord(
        t501["RA"], t501["DEC"], d["RA"], d["DEC"], search_radius=1.0
    )
    assert (iid501.size == len(t501)) & (d2d501.max() == 0)

    # n673
    iit673, iid673, d2d673, _, _ = match_coord(
        t673["RA"], t673["DEC"], d["RA"], d["DEC"], search_radius=1.0
    )
    assert (iid673.size == len(t673) - 373) & (
        d2d673.max() == 0
    )  # 373 rows are not matched..

    nrows = [0, iit501.size, iit673.size]
    mydict = get_init_infos("odin", nrows)

    for band, t, iid, iit in zip(
        ["N501", "N673"],
        [t501, t673],
        [iid501, iid673],
        [iit501, iit673],
    ):

        mydict[band]["TARGETID"] = d["TARGETID"][iid]
        mydict[band]["TERTIARY_TARGET"] = np.array(["LAE_ODIN" for _ in iid])  # custom
        mydict[band]["PHOT_RA"] = t["RA"][iit]
        mydict[band]["PHOT_DEC"] = t["DEC"][iit]
        mydict[band]["PHOT_SELECTION"] = t["SELECTION"][iit].astype(
            mydict[band]["PHOT_SELECTION"].dtype
        )

    return mydict


def get_odin_xmmlss_yr2_infos():
    """
    Get the minimal photometric infos for ODIN xmmlss_yr2 (tertiary18)

    Args:
        None

    Returns:
        mydict: dictionary with {keys: arrays},
            with keys: TARGETID, TERTIARY_TARGET, PHOT_RA, PHOT_DEC, PHOT_SELECTION
    """
    # n501, n673 (no targets)

    # n419
    fadir = os.path.join(
        os.getenv("DESI_ROOT"), "survey", "fiberassign", "special", "tertiary", "0018"
    )
    fn = os.path.join(fadir, "tertiary-targets-0018.fits")
    d = Table.read(fn)
    sel = d["TERTIARY_TARGET"] == "ODIN_BRIGHT"
    sel |= d["TERTIARY_TARGET"] == "ODIN_FAINT"
    nrows = [sel.sum(), 0, 0]
    mydict = get_init_infos("odin", nrows)
    mydict["N419"]["TARGETID"] = d["TARGETID"][sel]
    mydict["N419"]["TERTIARY_TARGET"] = d["TERTIARY_TARGET"][sel]
    mydict["N419"]["PHOT_RA"] = d["RA"][sel]
    mydict["N419"]["PHOT_DEC"] = d["DEC"][sel]
    mydict["N419"]["PHOT_SELECTION"] = np.array(
        ["N419+HSC" for _ in range(sel.sum())],
        dtype=mydict["N419"]["PHOT_SELECTION"].dtype,
    )

    return mydict


def get_odin_cosmos_yr2_infos():
    """
    Get the minimal photometric infos for ODIN cosmos_yr2 (tertiary26)

    Args:
        None

    Returns:
        mydict: dictionary with {keys: arrays},
            with keys: TARGETID, TERTIARY_TARGET, PHOT_RA, PHOT_DEC, PHOT_SELECTION
    """
    #
    fadir = os.path.join(
        os.getenv("DESI_ROOT"), "survey", "fiberassign", "special", "tertiary", "0026"
    )

    # first read the tertiary26 file
    fn = os.path.join(fadir, "tertiary-targets-0026-assign.fits")
    d = Table.read(fn)

    # cut on odin targets
    # remark:
    #   by construction, d["LAE_ODI4OBS"] has to belong to one of the three subsamples
    #       d["LAE_ODIN419"], d["LAE_ODIN501"], d["LAE_ODIN673"]
    sel = (d["LAE_ODIN419"]) | (d["LAE_ODIN501"]) | (d["LAE_ODIN673"])
    assert ((d["LAE_ODI4OBS"]) & (~sel)).sum() == 0

    # also include lae_subaru targets, as there can be some overlap..
    sel |= d["LAE_SUBARU"]

    d = d[sel]

    # get the row in Arjun s target file
    # with a sanity check along the way
    # remark:
    #   N673 have 35 repeats in Arjun s file
    #   4/35 of those repeats are in d
    #   that s why we grab the rows, and do not perform a sphere-match...
    rows = np.zeros(len(d), dtype=int)

    for i in range(len(d)):

        x = [
            d["LAE_ODIN419_ROW"][i],
            d["LAE_ODIN501_ROW"][i],
            d["LAE_ODIN673_ROW"][i],
            d["LAE_SUBARU_ROW"][i],
        ]
        x = np.unique([_ for _ in x if _ != -99])
        assert x.size == 1
        rows[i] = x[0]

    # now read Arjun s file
    # remark:
    #   tsel_n419 and tsel_n501 have 9 rows in common
    #   6/9 made it through into tertiary-targets-0026.fits
    #   TARGETIDs = 39089837499749125, 39089837499750560, 39089837499750713,
    #               39089837499750737, 39089837499750854, 39089837499752288
    #   and those 6/9 have been assigned
    fn = os.path.join(fadir, "inputcats", "COSMOS_LAE_Candidates_2023apr04v2.fits.gz")
    t = Table.read(fn)

    # define columns to identify the odin selections
    t["N419"] = np.array(["LAE N419" in _ for _ in t["CANDTYPE"]])
    t["N501"] = np.array(["LAE_501" in _ for _ in t["CANDTYPE"]])
    t["N673"] = np.array(["LAE_673" in _ for _ in t["CANDTYPE"]])
    t["N673"] |= np.array(["LAE_673b" in _ for _ in t["CANDTYPE"]])
    t["N673"] |= np.array(["LBG_673" in _ for _ in t["CANDTYPE"]])

    # match t to d
    t = t[rows]

    # sanity check
    # - for TERTIARY_TARGET=*ODI*: (ra, dec) should be exactly the same
    # - for TERTIARY_TARGET!=*ODI*: (ra, dec) are those from higher priority catalog
    #       and should be within 1 arcsec
    sel = np.array(["ODIN" in _ or "ODI4OBS" in _ for _ in d["TERTIARY_TARGET"]])
    dcs = SkyCoord(d["RA"] * units.degree, d["DEC"] * units.degree, frame="icrs")
    tcs = SkyCoord(t["RA"] * units.degree, t["DEC"] * units.degree, frame="icrs")
    seps = dcs.separation(tcs).to("arcsec").value
    assert np.all(seps[sel] == 0)
    assert np.all(seps[~sel] < 1)

    # now, t and d are row-matched
    nrows = [t["N419"].sum(), t["N501"].sum(), t["N673"].sum()]
    mydict = get_init_infos("odin", nrows)

    for band in ["N419", "N501", "N673"]:

        mydict[band]["TARGETID"] = d["TARGETID"][t[band]]
        mydict[band]["TERTIARY_TARGET"] = d["TERTIARY_TARGET"][t[band]]
        mydict[band]["PHOT_RA"] = t["RA"][t[band]]
        mydict[band]["PHOT_DEC"] = t["DEC"][t[band]]
        mydict[band]["PHOT_SELECTION"] = t["SELECTION"][t[band]].astype(
            mydict[band]["PHOT_SELECTION"].dtype
        )

    ## check
    for band in ["N419", "N501", "N673"]:

        names, counts = np.unique(d["TERTIARY_TARGET"][t[band]], return_counts=True)
        log.info(
            "{} ({}):\t{}".format(
                band,
                t[band].sum(),
                ", ".join(
                    ["{}={}".format(name, count) for name, count in zip(names, counts)]
                ),
            )
        )

    return mydict


# get photometry infos (targetid, brickname, objid)
# this is for odin targets only
# sky/std will have dummy values
def get_odin_phot_infos(case, d, photdir=None):
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
    bricknames = np.zeros(len(d), dtype="S8")
    objids = np.zeros(len(d), dtype=int)
    targfns = np.zeros(len(d), dtype="S100")

    # now get the per-band phot. infos
    for band in ["N419", "N501", "N673"]:

        ii_band = np.where(d[band])[0]
        fns = get_phot_fns("odin", case, band, photdir=photdir)
        log.info("{}\t{}\t{}\t{}".format(case, band, ii_band.size, fns))

        # is that band relevant for that case?
        if fns is None:

            continue

        for fn in fns:

            # indexes:
            # - targets selected with that band
            # - not dealt with yet (by previous fn)
            sel = (d[band]) & (bricknames == np.zeros(1, dtype="S8")[0])

            # special handling of cosmos_yr2, n419, which has two sets of phot (LS and HSC)
            # we want to only consider the targets passing the considered selection
            if (case == "cosmos_yr2") & (band == "N419"):

                if os.path.basename(fn) == "ODIN_N419_tractor_HSC_forced_all.fits.gz":

                    sel &= np.array(
                        ["LAE N419 ODIN+HSC" in _ for _ in d["PHOT_SELECTION"]]
                    )
                    log.info(
                        "{}\t{}\t{}:\tfurther request 'LAE N419 ODIN+HSC' in PHOT_SELECTION".format(
                            case, band, os.path.basename(fn)
                        )
                    )

                elif (
                    os.path.basename(fn) == "ODIN_N419_tractor_DR10_forced_all.fits.gz"
                ):

                    sel &= np.array(
                        ["LAE N419 ODIN+LSDR10" in _ for _ in d["PHOT_SELECTION"]]
                    )
                    log.info(
                        "{}\t{}\t{}:\tfurther request 'LAE N419 ODIN+DR10' in PHOT_SELECTION".format(
                            case, band, os.path.basename(fn)
                        )
                    )

                else:

                    msg = "{}\t{}: unexpected fn={}".format(case, band, fn)
                    log.error(msg)
                    raise ValueError(msg)

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

            # LAE_Candidates_NB673_v0_targeting.fits.gz has 373 duplicates
            # we remove those beforehand...
            if os.path.basename(fn) == "LAE_Candidates_NB673_v0_targeting.fits.gz":

                unqids = np.array(
                    [
                        "{}-{}".format(brn, oid)
                        for brn, oid in zip(t["BRICKNAME"], t["OBJID"])
                    ]
                )
                _, ii = np.unique(unqids, return_index=True)
                t = t[ii]

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

            # verify we have a "perfect" match
            # except for cosmos_2yr,N501, where there are 6 rows for targets also selected with n419,
            #   and which have the n419 ra, dec
            if (case == "cosmos_yr2") & (band == "N501"):

                assert (d2d != 0).sum() == np.array(
                    ["501" not in _.split(";")[0] for _ in d["PHOT_SELECTION"][ii_band]]
                ).sum()

            else:

                assert (d2d != 0).sum() == 0

            # fill the values
            iid = ii_band[iid]
            bricknames[iid] = t["BRICKNAME"][iit]
            objids[iid] = t["OBJID"][iit]
            targfns[iid] = fn

            # for n673:
            #   d has 4 duplicates (same targets, but different TARGETIDs)
            #   so those 4 rows will be empty here
            #   we fill them
            #   for each case, it s the second occurence of the TARGETID which is empty
            if (case == "cosmos_yr2") & (band == "N673"):

                ii_miss = np.where(
                    (d[band]) & (bricknames == np.zeros(1, dtype="S8")[0])
                )[0]
                assert ii_miss.size == 4
                log.info(
                    "{}\t{}\tdeal with {} duplicates...".format(
                        case, band, ii_miss.size
                    )
                )

                for i_miss in ii_miss:

                    ii = np.where(
                        (d["PHOT_RA"] == d["PHOT_RA"][i_miss])
                        & (d["PHOT_DEC"] == d["PHOT_DEC"][i_miss])
                    )[0]
                    assert ii.size == 2
                    assert ii[1] == i_miss
                    i_fill = ii[0]
                    bricknames[i_miss] = bricknames[i_fill]
                    objids[i_miss] = objids[i_fill]
                    targfns[i_miss] = targfns[i_fill]

        # verify all objects are matched
        assert ((d[band]) & (bricknames == "")).sum() == 0

    return bricknames, objids, targfns
