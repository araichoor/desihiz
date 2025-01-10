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
from desitarget.geomask import match
from desiutil.log import get_logger
from desihiz.hizmerge_io import (
    get_img_dir,
    get_img_bands,
    match_coord,
    get_init_infos,
    get_phot_fns,
)

log = get_logger()


def get_maskbits_fn():

    photdir = os.path.join(get_img_dir("ibis"), "phot")
    fn = os.path.join(photdir, "ibis3-xmm.fits")

    return fn


# [decam-chatter 18711] Suggested MASKBITS for IBIS
# 2024-12-20
# few bricks have no forced-hsc-wide photometry
def swap_maskbits(d):

    maskbitsfn = get_maskbits_fn()

    d2 = Table(fitsio.read(maskbitsfn, columns=["BRICKNAME", "OBJID", "MASKBITS"]))
    d2 = d2[np.in1d(d2["BRICKNAME"], d["BRICKNAME"])]
    d2["UNQID"] = ["{}-{}".format(b, o) for b, o in zip(d2["BRICKNAME"], d2["OBJID"])]
    ii, ii2 = match(d["UNQID"], d2["UNQID"])
    assert np.all(d["UNQID"][ii] == d2["UNQID"][ii2])
    assert ii.size == len(d)
    d["MASKBITS"][ii] = d2["MASKBITS"][ii2]
    return d


# https://desisurvey.slack.com/archives/C027M1AF31C/p1734823315418679?thread_ts=1734806556.969119&cid=C027M1AF31C
# 2024-12-21
def add_synthg(d):

    bands = get_img_bands("ibis")
    # AR remove fake added band
    bands = [_ for _ in bands if _ != "M541"]

    cs = np.array([-0.004, 0.120, 0.262, 0.229, 0.359])
    assert len(cs) == len(bands)

    d["FLUX_SYNTHG"], d["FLUX_IVAR_SYNTHG"], d["FIBERFLUX_SYNTHG"] = 0.0, 0.0, 0.0
    vs = np.zeros(len(d))
    for quant in ["FLUX", "FIBERFLUX"]:
        fkey = "{}_SYNTHG".format(quant)
        for c, band in zip(cs, bands):
            d[fkey] += c * d["{}_{}".format(quant, band)]
            if quant == "FLUX":
                vs += c ** 2 / d["FLUX_IVAR_{}".format(band)]
        d["FLUX_IVAR_SYNTHG"] = vs ** -1.0

    d.meta["SYNTGCS"] = ",".join(cs.astype(str))

    return d


def get_ibis_xmmlss_yr4_phot_selections(unqids):

    fadir = os.path.join(
        os.getenv("DESI_ROOT"), "survey", "fiberassign", "special", "tertiary", "0044"
    )

    phot_sels = np.zeros(len(unqids), dtype="<U100")
    empty_selection = phot_sels[0]

    print(unqids.size, np.unique(unqids).size)
    for checker in ["AR", "DJS", "HE"]:

        if checker == "AR":

            basename = "ibis-xmm-ar-targets-v3.fits"
            names = [
                "AR_LAE_M411",
                "AR_LAE_M438",
                "AR_LBG_M438",
                "AR_LAE_M464",
                "AR_LBG_M464",
                "AR_LAE_M490",
                "AR_LBG_M490",
                "AR_LAE_M517",
                "AR_LBG_M517"
            ]

        if checker == "DJS":

            basename = "xmm-targets-all.fits"
            names = [
                "DJS_LAE_M411",
                "DJS_LBG_M411",
                "DJS_LAE_M438",
                "DJS_LBG_M438",
                "DJS_LAE_M464",
                "DJS_LBG_M464",
                "DJS_LAE_M490",
                "DJS_LBG_M490",
                "DJS_LAE_M517",
                "DJS_LAE_M541",
            ]

        if checker == "HE":

            basename = "ibis3-xmm-ebv-clauds-he-targets.fits"
            names = [
                "HE_M411",
                "HE_M438",
                "HE_M464",
                "HE_M490",
                "HE_M517"
            ]

        # AR it is expected that some targets are not in unqids
        t = Table(fitsio.read(os.path.join(fadir, "inputcats", basename)))
        if "UNQID" not in t.colnames:
            t["UNQID"] = ["{}-{}".format(b, o) for b, o in zip(t["BRICKNAME"], t["OBJID"])]
        ii, iit = match(unqids, t["UNQID"])
        print(checker, ii.size, len(t))
        #assert np.all(unqids[ii] == t["UNQID"])

        for name in names:

            # AR
            if checker == "AR":
                key = "_".join(name.split("_")[1:])
                jj = np.where(t[key][iit])[0]
            # DJS
            if checker == "DJS":
                val = "".join(name.split("_")[1:])
                for tmpi, tmpband in zip(["1", "2", "3" ,"4", "5", "6"], ["M411", "M438", "M464", "M490", "M517", "M541"]):
                    val = val.replace(tmpband, tmpi)
                jj = np.where([val in _ for _ in t["SAMPLE"][iit]])[0]
            # HE
            if checker == "HE":
                key = name.replace("HE", "SEL")
                jj = np.where(t[key][iit])[0]

            for i in ii[jj]:
                if phot_sels[i] == empty_selection:
                    phot_sels[i] = name
                else:
                    phot_sels[i] += "; {}".format(name)

    return phot_sels


def get_ibis_xmmlss_yr4_infos():
    """
    Get the minimal photometric infos for IBIS targets from tertiary44

    Args:
        None

    Returns:
        mydict: dictionary with {keys: arrays},
            with keys: TARGETID, TERTIARY_TARGET, PHOT_RA, PHOT_DEC, PHOT_SELECTION
    """
    #
    fadir = os.path.join(
        os.getenv("DESI_ROOT"), "survey", "fiberassign", "special", "tertiary", "0044"
    )
    bands = get_img_bands("ibis")

    # first read the tertiary44 file
    fn = os.path.join(fadir, "tertiary-targets-0044-assign.fits")
    d = Table.read(fn)

    # cut on ibis targets
    # all HIZ1H_IBIS also are HIZ_IBIS
    sel = d["HIZ_IBIS"].copy()
    d = d[sel]

    # get the row in the target file
    rows = d["HIZ_IBIS_ROW"].copy()

    # now read targets file
    fn = os.path.join(fadir, "inputcats", "ibis-xmm-ar-djs-he.fits")
    t = Table.read(fn)

    # match t to d
    t = t[rows]

    # sanity check:
    # - IBIS targets are top-priority, so we should have an exact (ra, dec) match
    dcs = SkyCoord(d["RA"] * units.degree, d["DEC"] * units.degree, frame="icrs")
    tcs = SkyCoord(t["RA"] * units.degree, t["DEC"] * units.degree, frame="icrs")
    seps = dcs.separation(tcs).to("arcsec").value
    assert np.all(seps == 0)

    # initialize (with grabbing correct datamodel)
    tmpdict = get_init_infos("ibis", [len(t), 0, 0, 0, 0, 0])[bands[0]]

    for key in ["PHOT_RA", "PHOT_DEC", "PHOT_SELECTION"]:

        d[key] = tmpdict[key]

    # get ibis_ra, ibis_dec
    d["PHOT_RA"], d["PHOT_DEC"] = t["RA"], t["DEC"]

    # get band + selection
    unqids = np.array(["{}-{}".format(b, o) for b, o in zip(t["BRICKNAME"], t["OBJID"])])
    d["PHOT_SELECTION"] = get_ibis_xmmlss_yr4_phot_selections(unqids)
    for band in bands:
        d[band] = False
        sel = np.array([band in _ for _ in d["PHOT_SELECTION"]])
        d[band][sel] = True

    # sanity check
    ## all rows are filled
    assert np.all(d["TERTIARY_TARGET"] != 0)
    assert np.all(d["PHOT_RA"] != 0)
    assert np.all(d["PHOT_DEC"] != 0)
    ## IBIS targets are top-priority, (ra, dec) match should be exact
    fa_cs = SkyCoord(d["RA"] * units.degree, d["DEC"] * units.degree, frame="icrs")
    ibis_cs = SkyCoord(
        d["PHOT_RA"] * units.degree, d["PHOT_DEC"] * units.degree, frame="icrs"
    )
    seps = fa_cs.separation(ibis_cs).to("arcsec").value
    assert np.all(seps == 0)

    #
    mydict = get_init_infos("ibis", [d[band].sum() for band in bands])

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


# get photometry infos (targetid, brickname, objid)
# this is for ibis targets only
# sky/std will have dummy values
def get_ibis_phot_infos(case, d, photdir=None):
    """
    Get the photometric information (TARGETID, BRICKNAME, OBJID) for a given case


    Args:
        case: round of DESI observation (str)
        d: output of the get_spec_table() function
        photdir (optional, defaults to $DESI_ROOT/users/raichoor/laelbg/{img}/phot):
            folder where the files are
    """
    if photdir is None:

        photdir = os.path.join(get_img_dir("ibis"), "phot")

    # initialize columns we will fill
    bricknames = np.zeros(len(d), dtype="S8")
    objids = np.zeros(len(d), dtype=int)
    targfns = np.zeros(len(d), dtype="S150")

    empty_brickname = bricknames[0]

    # now get the per-band phot. infos
    bands = get_img_bands("ibis")

    for band in bands:

        ii_band = np.where(d[band])[0]
        fns = get_phot_fns("ibis", case, band, photdir=photdir)
        log.info("{}\t{}\t{}\t{}".format(case, band, ii_band.size, fns))

        # is that band relevant for that case?
        if fns is None:

            continue

        for fn in fns:

            # indexes:
            # - targets selected with that band
            # - not dealt with yet (by previous fn)
            sel = (d[band]) & (bricknames == empty_brickname)
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

            # fill the values
            iid = ii_band[iid]
            bricknames[iid] = t["BRICKNAME"][iit]
            objids[iid] = t["OBJID"][iit]
            targfns[iid] = fn

        assert ((d[band]) & (bricknames == empty_brickname)).sum() == 0

    return bricknames, objids, targfns
