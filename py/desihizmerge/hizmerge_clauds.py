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
from desitarget.geomask import match_to
from desiutil.log import get_logger
from desihizmerge.hizmerge_io import (
    default_photdir,
    get_img_bands,
    match_coord,
    get_init_infos,
    get_phot_fns,
    get_clauds_fn,
)

log = get_logger()


def get_clauds_cosmos_yr1_infos():
    """
    Get the minimal photometric infos for CLAUDS cosmos_yr1 (TILEID=80871,80872)

    Args:
        None

    Returns:
        mydict: dictionary with {keys: arrays},
            with keys: TARGETID, TERTIARY_TARGET, PHOT_RA, PHOT_DEC, PHOT_SELECTION

    Notes:
        At least DESILBG_BXU_FINAL.fits and DESILBG_TMG_FINAL.fits have objects in common:
            those have different TARGETIDs but are the same underlying CLAUDS objects
        LBG_TOMOG_COSMOS_FINAL.fits has some duplicated targetids (maybe that s expected, don t know),
            but none of those made it into 80871,80872; so we simplify things downstream
            with retaining only one row for each
    """
    #
    photdtype = get_init_infos("clauds", [0, 0, 0])["UGR"]["PHOT_SELECTION"].dtype
    # those four catalogs have zero overlap in TARGETIDs between them
    #   so we can just vstack them, and take a single PHOT_SELECTION value
    myds = {}
    mydir = os.path.join(
        os.getenv("DESI_TARGET"), "secondary", "sv1", "outdata", "0.52.0", "dark"
    )

    # UGR (BXU, TMG, TOMOG_COSMOS)
    fns = [
        os.path.join(mydir, "DESILBG_BXU_FINAL.fits"),
        os.path.join(mydir, "DESILBG_TMG_FINAL.fits"),
        os.path.join(mydir, "LBG_TOMOG_COSMOS_FINAL.fits"),
    ]
    photnames = [
        "COSMOS_YR1_BXU",
        "COSMOS_YR1_TMG",
        "COSMOS_YR1_TOMOG_COSMOS",
    ]
    ds = []

    for fn, photname in zip(fns, photnames):

        d = Table(fitsio.read(fn))

        # duplicates in LBG_TOMOG_COSMOS_FINAL.fits
        if os.path.basename(fn) == "LBG_TOMOG_COSMOS_FINAL.fits":

            _, ii = np.unique(d["TARGETID"], return_index=True)
            log.info(
                "{}\t: remove {} duplicates in TARGETID".format(
                    os.path.basename(fn), len(d) - ii.size
                )
            )
            d = d[ii]

        d["TERTIARY_TARGET"] = os.path.basename(fn).split(os.path.extsep)[0]
        d["PHOT_SELECTION"] = np.zeros(len(d), dtype=photdtype)
        d["PHOT_SELECTION"] = photname
        ds.append(d)

    myds["UGR"] = vstack(ds)

    # USGR: none

    # GRI (G)
    fn = os.path.join(mydir, "DESILBG_G_FINAL.fits")
    photname = "COSMOS_YR1_GDROP"
    d = Table(fitsio.read(fn))
    d["TERTIARY_TARGET"] = os.path.basename(fn).split(os.path.extsep)[0]
    d["PHOT_SELECTION"] = np.zeros(len(d), dtype=photdtype)
    d["PHOT_SELECTION"] = photname
    myds["GRI"] = d

    # store
    nrows = [len(myds["UGR"]), 0, len(myds["GRI"])]
    mydict = get_init_infos("clauds", nrows)

    for band in ["UGR", "GRI"]:

        assert np.unique(myds[band]["TARGETID"]).size == len(myds[band])
        mydict[band]["TARGETID"] = myds[band]["TARGETID"]
        mydict[band]["TERTIARY_TARGET"] = myds[band]["TERTIARY_TARGET"]
        mydict[band]["PHOT_RA"] = myds[band]["RA"]
        mydict[band]["PHOT_DEC"] = myds[band]["DEC"]
        mydict[band]["PHOT_SELECTION"] = myds[band]["PHOT_SELECTION"]

    return mydict


def get_clauds_xmmlss_yr2_infos():
    """
    Get the minimal photometric infos for CLAUDS xmmlss_yr2 (tertiary15)

    Args:
        None

    Returns:
        mydict: dictionary with {keys: arrays},
            with keys: TARGETID, TERTIARY_TARGET, PHOT_RA, PHOT_DEC, PHOT_SELECTION
    """

    fadir = os.path.join(
        os.getenv("DESI_ROOT"), "survey", "fiberassign", "special", "tertiary", "0015"
    )
    # first get the TARGETID, TERTIARY_TARGET, etc of the considered targets
    fn = os.path.join(fadir, "tertiary-targets-0015.fits")
    d = Table.read(fn)
    sel = np.zeros(len(d), dtype=bool)
    for name in ["LBG", "LBG_EXTBX", "LBG_EXTFAINT", "LBG_HIP"]:
        sel |= d["TERTIARY_TARGET"] == name
    d = d[sel]

    # then add if it belongs to LBG, LBG_EXTBX, or LBG_EXTFAINT
    #   for that we row-match the target catalog to d
    #   and will use these columns:
    #   - LBG      => LBG_HIP, LBG
    #   - LBGFAINT => LBG_EXTFAINT
    #   - BX       => LBG_EXTBX
    fn = os.path.join(fadir, "inputcats", "lbg-xmmlss-fall2022-hipr24.2-extsub1.0.fits")
    t = Table.read(fn)
    t["ID"] = t["ID"].astype(str)
    assert np.unique(d["ORIG_ID"]).size == len(d)
    assert np.unique(t["ID"]).size == len(t)
    iit = match_to(t["ID"], d["ORIG_ID"])
    t = t[iit]
    assert np.all(t["ID"] == d["ORIG_ID"])

    # PHOT_SELECTION column
    # a bit hacky... want a np array of empty lists
    # https://stackoverflow.com/questions/43483663/how-do-i-make-a-grid-of-empty-lists-in-numpy-that-accepts-appending
    arr_phot_selections = np.empty((len(d),), dtype=object)
    for i in np.ndindex(arr_phot_selections.shape):
        arr_phot_selections[i] = []
    for photname, tname in zip(
        ["XMMLSS_YR2_USGR_LBG", "XMMLSS_YR2_USGR_EXTBX", "XMMLSS_YR2_USGR_EXTFAINT"],
        ["LBG", "BX", "LBGFAINT"],
    ):
        sel = t[tname]
        for i in np.where(sel)[0]:
            arr_phot_selections[i].append(photname)
    photdtype = get_init_infos("clauds", [0, 0, 0])["UGR"]["PHOT_SELECTION"].dtype
    d["PHOT_SELECTION"] = np.array(
        ["; ".join(_) for _ in arr_phot_selections], dtype=photdtype
    )

    # USGR only (i.e. no UGR nor GRI)
    nrows = [0, 0, len(d)]
    mydict = get_init_infos("clauds", nrows)
    band = "USGR"
    mydict[band]["TARGETID"] = d["TARGETID"]
    mydict[band]["TERTIARY_TARGET"] = d["TERTIARY_TARGET"]
    mydict[band]["PHOT_RA"] = d["RA"]
    mydict[band]["PHOT_DEC"] = d["DEC"]
    mydict[band]["PHOT_SELECTION"] = d["PHOT_SELECTION"]

    return mydict


def get_clauds_cosmos_yr2_lbgnew_u_or_us_tids():
    """

    Notes:
        Cuts: see email from Christophe from 3/16/23, 2:51 AM pacific
    """
    # extract the LBG_NEW targets
    fadir = os.path.join(
        os.getenv("DESI_ROOT"), "survey", "fiberassign", "special", "tertiary", "0026"
    )
    fn = os.path.join(fadir, "tertiary-targets-0026.fits")
    d = Table.read(fn)
    sel = d["LBG_NEW"]
    d = d[sel]
    log.info("found {} LBG_NEW targets in {}".format(len(d), os.path.basename(fn)))
    mydict = {}

    for uband in ["u", "uS"]:

        # read phot. catalog + cut on the ugr or uSgr targets
        pfn = get_clauds_fn("cosmos_yr2", v2=True, uband=uband)
        p = fitsio.read(pfn)
        # mimick what was done for clauds-sext-cosmos-{u,uS}gr-r25.fits
        sel = p["MASK"] == 0
        sel &= p["ST_TRAIL"] == 0
        if uband == "u":
            sel &= p["FLAG_FIELD_BINARY"][:, 1]
        elif uband == "uS":
            sel &= p["FLAG_FIELD_BINARY"][:, 2]
        sel &= p["FLAG_FIELD_BINARY"][:, 0]
        for band in [uband, "g", "r"]:
            sel &= (p[band] > 0) & (p[band] < 40)
            sel &= (p["{}_err".format(band)] > 0) & (p["{}_err".format(band)] < 100)
        p = p[sel]
        #
        ugs, grs = p[uband] - p["g"], p["g"] - p["r"]
        if uband == "u":
            sel = (ugs > 0.3) & ((ugs > 2.2 * grs + 0.32) | (ugs > 1.6 * grs + 0.75))
        if uband == "uS":
            sel = (ugs > 0.3) & ((ugs > 2.0 * grs + 0.42) | (ugs > 1.6 * grs + 0.55))
        p = p[sel]

        # match d and p
        iid, iip, d2d, _, _ = match_coord(
            d["RA"], d["DEC"], p["RA"], p["DEC"], search_radius=1.0
        )
        log.info(
            "matched {}/{} of LBG_NEW targets with {}".format(
                iid.size, len(d), os.path.basename(pfn)
            )
        )

        # assert non-exact matches are only for non-CLAUDS, higher-priority targets
        sel = d2d != 0
        sel2 = (sel) & ((d["SUPRIME"][iid]) | (d["LAE_SUBARU"][iid]))
        assert np.all(sel == sel2)
        log.info(
            "\tfound {} d2d!=0 matches, but all come from either SUPRIME or LAE_SUBARU, so ok".format(
                sel.sum()
            )
        )

        mydict[uband] = d["TARGETID"][iid]

        # TODO: understand this...
        # hacky fix: 8 TARGETIDs are missing
        # - from debugging, their uSg and gr colors fall at <0.01 mag away from the uSgr selection cuts
        # - I m not sure why is that, i.e. it s larger than numerical precision...
        # - but still, it is extremely likely those belong to the uSgr selection
        # => so I manually add those to the uSgr selection
        if uband == "uS":
            extra_tids = np.array(
                [
                    39089837499747308,
                    39089837499747361,
                    39089837499747773,
                    39089837499747886,
                    39089837499747970,
                    39089837499747974,
                    39089837499747977,
                    39089837499748056,
                ]
            )
            assert np.all(~np.in1d(extra_tids, mydict[uband]))
            mydict[uband] = np.append(mydict[uband], extra_tids)
            log.info(
                "manually add {} TARGETIDs to the uSgr selection (see code for details): {}".format(
                    extra_tids.size, ", ".join(extra_tids.astype(str))
                )
            )

    # simple (dummy) sanity check
    sel = np.in1d(d["TARGETID"], mydict["u"].tolist() + mydict["uS"].tolist())
    assert np.all(np.in1d(d["TARGETID"], mydict["u"].tolist() + mydict["uS"].tolist()))

    return mydict["u"], mydict["uS"]


def get_clauds_cosmos_yr2_infos():
    """
    Get the minimal photometric infos for CLAUDS cosmos_yr2 (tertiary26)

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

    # cut on lbg targets (discard LBG_HYP, LBG_HYPHQ, as not selected with CLAUDS...)
    sel = np.zeros(len(d), dtype=bool)
    for name in ["LBG_Z2", "LBG_NEW", "LBG_EXT", "LBG_REOBS"]:
        sel |= d[name]
    d = d[sel]

    # LBG_Z2, LBG_EXT, LBG_REOBS: selected with u
    # LBG_NEW: mix of u and uS, see email Christophe 3/16/23, 2:51 AM pacific
    # - identify which targets come from a u or uS selection
    # - fill a PHOT_SELECTION column
    lbgnew_u_tids, lbgnew_uS_tids = get_clauds_cosmos_yr2_lbgnew_u_or_us_tids()
    # a bit hacky... want a np array of empty lists
    # https://stackoverflow.com/questions/43483663/how-do-i-make-a-grid-of-empty-lists-in-numpy-that-accepts-appending
    arr_phot_selections = np.empty((len(d),), dtype=object)

    for i in np.ndindex(arr_phot_selections.shape):

        arr_phot_selections[i] = []

    mysel = {}

    # UGR: LBG_NEW
    sel = np.in1d(d["TARGETID"], lbgnew_u_tids)

    for i in np.where(sel)[0]:

        arr_phot_selections[i].append("COSMOS_YR2_LBG_NEW_UGR")

    mysel["UGR"] = sel.copy()

    # UGR: LBG_Z2, LBG_EXT, LBG_REOBS
    for name in ["LBG_Z2", "LBG_EXT", "LBG_REOBS"]:

        name_sel = d[name]

        for i in np.where(name_sel)[0]:

            arr_phot_selections[i].append("COSMOS_YR2_{}".format(name))

        mysel["UGR"] |= name_sel

    # USGR: LBG_NEW
    sel = np.in1d(d["TARGETID"], lbgnew_uS_tids)

    for i in np.where(sel)[0]:

        arr_phot_selections[i].append("COSMOS_YR2_LBG_NEW_USGR")

    mysel["USGR"] = sel.copy()

    assert np.all((mysel["UGR"]) | (mysel["USGR"]))
    photdtype = get_init_infos("clauds", [0, 0, 0])["UGR"]["PHOT_SELECTION"].dtype
    d["PHOT_SELECTION"] = np.array(
        ["; ".join(_) for _ in arr_phot_selections], dtype=photdtype
    )

    nrows = [mysel["UGR"].sum(), mysel["USGR"].sum(), 0]
    mydict = get_init_infos("clauds", nrows)

    for band in ["UGR", "USGR"]:

        mydict[band]["TARGETID"] = d["TARGETID"][mysel[band]]
        mydict[band]["TERTIARY_TARGET"] = d["TERTIARY_TARGET"][mysel[band]]
        mydict[band]["PHOT_RA"] = d["RA"][mysel[band]]
        mydict[band]["PHOT_DEC"] = d["DEC"][mysel[band]]
        mydict[band]["PHOT_SELECTION"] = d["PHOT_SELECTION"][mysel[band]]

    return mydict


# get photometry infos (clauds_id)
# this is for clauds targets only
# sky/std will have dummy values
def get_clauds_phot_infos(case, d, photdir=None, v2=None):
    """
    Get the photometric information (TARGETID, ID) for a given case

    Args:
        case: round of DESI observation (str)
        d: output of the get_spec_table() function
        photdir (optional, defaults to $DESI_ROOT/users/raichoor/laelbg/{img}/phot):
            folder where the files are
        v2 (optional, defaults to False): for img=clauds, if True, use custom catalogs
            with per-HSC pointing photometric offset on the Desprez+23 catalogs,
            (see https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=7493)
            (bool)

    Returns:
        claudsids: the CLAUDS ID (np.array of int)
        targfns:
    """
    if photdir == None:

        photdir = default_photdir

    # initialize columns we will fill
    claudsids = np.zeros(len(d), dtype=int)
    targfns = np.zeros(len(d), dtype="S150")

    # now get the per-band phot. infos
    bands = get_img_bands("clauds")

    for band in bands:

        ii_band = np.where(d[band])[0]
        fns = get_phot_fns("clauds", case, band, photdir=photdir, v2=v2)
        log.info("{}\t{}\t{}\t{}".format(case, band, ii_band.size, fns))

        # is that band relevant for that case?
        if fns is None:

            continue

        for fn in fns:

            # indexes:
            # - targets selected with that band
            # - not dealt with yet (by previous fn)
            sel = (d[band]) & (claudsids == 0)

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
            #   but have ra,dec from another imaging than clauds
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

                # remark: in tertiary-targets-0026.fits, when the ra, dec merging is done,
                #   the lower-priority LAE_ODI4OBS and LAE_SUB4OBS classes are not defined yet
                #   targets from LAE_ODI4OBS and LAE_SUB4OBS inherit the ra, dec from the
                #   higher priority LAE_ODIN and LAE_SUBARU classes, which take precedence
                #   over some LBG targets;
                #   hence, for a considered LBG target class, if LAE_ODIN or LAE_SUBARU are higher-priority
                #   targets, we also add LAE_ODI4OBS and LAE_SUB4OBS
                #
                hip_targs = {
                    "COSMOS_YR2_LBG_Z2": ["LBG_HYPHQ", "LBG_HYP"],
                    "COSMOS_YR2_LBG_NEW_UGR": [
                        "LBG_HYPHQ",
                        "LBG_HYP",
                        "SUPRIME",
                        "LAE_SUBARU",
                        "LAE_SUB4OBS",
                    ],
                    "COSMOS_YR2_LBG_NEW_USGR": [
                        "LBG_HYPHQ",
                        "LBG_HYP",
                        "SUPRIME",
                        "LAE_SUBARU",
                        "LAE_SUB4OBS",
                    ],
                    "COSMOS_YR2_LBG_EXT": [
                        "LBG_HYPHQ",
                        "LBG_HYP",
                        "SUPRIME",
                        "LAE_SUBARU",
                        "LAE_SUB4OBS",
                    ],
                    "COSMOS_YR2_LBG_REOBS": [
                        "LBG_HYPHQ",
                        "LBG_HYP",
                        "SUPRIME",
                        "LAE_SUBARU",
                        "LAE_ODIN419",
                        "LAE_ODIN501",
                        "LAE_ODIN673",
                        "LBG_UDROP",
                        "LAE_SUB4OBS",
                        "LAE_ODI4OBS",
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
            claudsids[iid] = t["ID"][iit]
            targfns[iid] = fn

            # cosmos_yr1: at least DESILBG_TMG_FINAL and DESILBG_BXU_FINAL
            #   have objects in common, but those have different TARGETIDs
            #   such duplicates are excluded with the approach above
            #   we here handle them
            if case == "cosmos_yr1":

                radecs = np.array(
                    [
                        "{}-{}".format(ra, dec)
                        for ra, dec in zip(d["PHOT_RA"], d["PHOT_DEC"])
                    ]
                )
                unqradecs, counts = np.unique(
                    radecs[d[band]], return_counts=True
                )  # restrict to the considered "band"
                unqradecs = unqradecs[counts > 1]  # cut on repeats
                unqradecs = unqradecs[
                    np.in1d(unqradecs, radecs[iid])
                ]  # cut on matched rows for this band
                log.info(
                    "{}\t{}\thandle {} (PHOT_RA,PHOT_DEC) which have repeats".format(
                        case, band, unqradecs.size
                    )
                )
                fill_counts = 0
                empty_targfn = np.zeros_like(targfns, shape=(1,))[0]

                for unqradec in unqradecs:

                    ii = np.where((d[band]) & (radecs == unqradec))[0]
                    log.info(
                        "{}\t{}\tii={}\tTARGETID={}".format(
                            band, unqradec, ii.tolist(), d["TARGETID"][ii].tolist()
                        )
                    )
                    tmpids, tmptargfns = claudsids[ii], targfns[ii]
                    assert ii.size > 1
                    assert (tmpids != 0).sum() == 1
                    assert (tmptargfns != empty_targfn).sum() == 1
                    tmpid = [_ for _ in tmpids if _ != 0][0]
                    tmptargfn = [_ for _ in tmptargfns if _ != empty_targfn][0]
                    # overwrite the already recorded (matched) value, but ok, simpler code-wise
                    claudsids[ii] = tmpid
                    targfns[ii] = tmptargfn
                    fill_counts += ii.size - 1  # because one value was already filled

                log.info(
                    "{}\t{}\tfill an additional {}/{} rows with handling duplicates".format(
                        case, band, fill_counts, ii_band.size
                    )
                )

        # verify all objects are matched
        assert ((d[band]) & (claudsids == 0)).sum() == 0

    return claudsids, targfns
