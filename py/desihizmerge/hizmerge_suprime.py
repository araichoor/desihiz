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
from desihizmerge.hizmerge_io import (
    default_photdir,
    match_coord,
    get_img_bands,
    get_init_infos,
    get_phot_fns,
)

log = get_logger()


def convert_suprime_cosmos_yr2_djssel_to_band(djs_sel):
    """
    Convert DJS notation to SUPRIME band

    Args:
        djs_sel: int (1, 2, 4, or 8)

    Returns:
        SUPRIME band

    Notes:
        This is from Suprime-LBG-Schlegel.fits
    """
    assert djs_sel in [1, 2, 4, 8]

    if djs_sel == 1:

        return "I464"

    elif djs_sel == 2:

        return "I484"

    elif djs_sel == 4:

        return "I505"

    elif djs_sel == 8:

        return "I527"

    else:

        msg = "unexpected djs_sel={}".format(djs_sel)
        log.error(msg)
        raise ValueError(msg)


def get_suprime_cosmos_yr2_infos():
    """
    Get the minimal photometric infos for SUPRIME cosmos_yr2 (tertiary26)

    Args:
        None

    Returns:
        mydict: dictionary with {keys: arrays},
            with keys: TARGETID, TERTIARY_TARGET, PHOT_RA, PHOT_DEC, PHOT_SELECTION
    """
    fadir = os.path.join(
        os.getenv("DESI_ROOT"), "survey", "fiberassign", "special", "tertiary", "0026"
    )

    bands = get_img_bands("suprime")

    # first read the tertiary26 file
    fn = os.path.join(fadir, "tertiary-targets-0026-assign.fits")
    d = Table.read(fn)

    # cut on suprime targets
    # by construction, all LAE_SUB4OBS have to also be LAE_SUBARU
    sel = (d["SUPRIME"]) | (d["LAE_SUBARU"])
    assert ((d["LAE_SUB4OBS"]) & (~sel)).sum() == 0
    d = d[sel]

    # initialize (with grabbing correct datamodel)
    tmpdict = get_init_infos("suprime", [len(d), 0, 0, 0, 0])[bands[0]]

    for key in ["PHOT_RA", "PHOT_DEC", "PHOT_SELECTION"]:

        d[key] = tmpdict[key]

    for band in bands:

        d[band] = False

    # to handle bytes / str differences downstream...
    empty_suprime_selection = d["PHOT_SELECTION"][0]

    log.info("# CHECKER BAND NTARG NTARG_ALSO_OTHER_SEL")

    # David s targets file
    ii_djs = np.where(d["SUPRIME"])[0]
    fn = os.path.join(fadir, "inputcats", "Suprime-LBG-Schlegel.fits")
    t = Table(fitsio.read(fn))

    # row-match it to d[ii_djs]
    t = t[d["SUPRIME_ROW"][ii_djs]]

    # get suprime_ra, suprime_dec
    d["PHOT_RA"][ii_djs], d["PHOT_DEC"][ii_djs] = t["RA"], t["DEC"]

    # get band + selection
    for djs_sel in np.unique(t["LBG_SELECTION"]):

        ii_t = np.where(t["LBG_SELECTION"] == djs_sel)[0]

        if ii_t.size == 0:

            continue

        ii_d = ii_djs[ii_t]
        band = convert_suprime_cosmos_yr2_djssel_to_band(djs_sel)
        d[band][ii_d] = True
        selname = "DJS_SEL{}".format(djs_sel)  # e.g. "DJS_SEL1"
        count = 0

        for i_d in ii_d:

            if d["PHOT_SELECTION"][i_d] == empty_suprime_selection:

                d["PHOT_SELECTION"][i_d] = selname

            else:

                d["PHOT_SELECTION"][i_d] += "; {}".format(selname)
                count += 1

        log.info("DJS\t{}\t{}\t{}".format(band, ii_t.size, count))

    # Arjun s targets file
    ii_ad = np.where(d["LAE_SUBARU"])[0]
    fn = os.path.join(fadir, "inputcats", "COSMOS_LAE_Candidates_2023apr04v2.fits.gz")
    t = Table.read(fn)

    # row-match it to d[ii_ad]
    t = t[d["LAE_SUBARU_ROW"][ii_ad]]

    # get suprime_ra, suprime_dec
    prev_ras, prev_decs = d["PHOT_RA"], d["PHOT_DEC"]
    d["PHOT_RA"][ii_ad], d["PHOT_DEC"][ii_ad] = t["RA"], t["DEC"]

    # assert that the ra, dec are the same for the djs x ad overlap
    sel = prev_ras != 0.0
    assert np.all(prev_ras[sel] == d["PHOT_RA"][sel])
    sel = prev_decs != 0.0
    assert np.all(prev_decs[sel] == d["PHOT_DEC"][sel])

    # get band + selection
    for band in bands:

        tmpname = "LAE {}".format(band.replace("I", "IA"))
        ii_t = np.where([tmpname in _ for _ in t["SELECTION"]])[0]

        if ii_t.size == 0:

            continue

        ii_d = ii_ad[ii_t]
        d[band][ii_d] = True
        count = 0

        for i_t, i_d in zip(ii_t, ii_d):

            if d["PHOT_SELECTION"][i_d] == empty_suprime_selection:

                d["PHOT_SELECTION"][i_d] = t["SELECTION"][i_t]

            else:

                d["PHOT_SELECTION"][i_d] += "; {}".format(t["SELECTION"][i_t])
                count += 1

        log.info("AD\t{}\t{}\t{}".format(band, ii_t.size, count))

    # sanity check

    ## all rows are filled
    assert np.all(d["TERTIARY_TARGET"] != 0)
    assert np.all(d["PHOT_RA"] != 0)
    assert np.all(d["PHOT_DEC"] != 0)
    assert np.all(d["PHOT_SELECTION"] != empty_suprime_selection)
    ## - if TERTIARY_TARGET=SUPRIME,LAE_SUBARU,LAE_SUB4OBS:
    ##       (ra, dec) should be exactly the same
    ##   else:
    ##       (ra, dec) are those from higher priority catalog
    ##       and should be within 1 arcsec
    sel = np.array(
        [
            "SUPRIME" in _ or "LAE_SUBARU" in _ or "SUB4OBS" in _
            for _ in d["TERTIARY_TARGET"]
        ]
    )
    fa_cs = SkyCoord(d["RA"] * units.degree, d["DEC"] * units.degree, frame="icrs")
    suprime_cs = SkyCoord(
        d["PHOT_RA"] * units.degree, d["PHOT_DEC"] * units.degree, frame="icrs"
    )
    seps = fa_cs.separation(suprime_cs).to("arcsec").value
    assert np.all(seps[sel] == 0)
    assert np.all(seps[~sel] < 1)

    #
    mydict = get_init_infos("suprime", [d[band].sum() for band in bands])

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
# this is for suprime targets only
# sky/std will have dummy values
def get_suprime_phot_infos(case, d, photdir=None,v2=False):
    """
    Get the photometric information (TARGETID, BRICKNAME, OBJID) for a given case

    Args:
        case: round of DESI observation (str)
        d: output of the get_spec_table() function
        photdir (optional, defaults to $DESI_ROOT/users/raichoor/laelbg/{img}/phot):
            folder where the files are
        v2 (optional, default to False): if True, then use Dustin's rerun from 20231025
            (bool)
    """
    if photdir == None:

        photdir = default_photdir

    # initialize columns we will fill
    bricknames = np.zeros(len(d), dtype="S8")
    objids = np.zeros(len(d), dtype=int)
    targfns = np.zeros(len(d), dtype="S150")

    empty_brickname = bricknames[0]

    # now get the per-band phot. infos
    bands = get_img_bands("suprime")

    for band in bands:

        ii_band = np.where(d[band])[0]
        fns = get_phot_fns("suprime", case, band, photdir=photdir, v2=v2)
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

            # if v2=False:
            # we expect to have a "perfect" match, except when:
            # - djs+ad targets: a higher-priority tertiary26 sample drove the (ra, dec)
            #   at the fiberassign tertiary step (identified with TERTIARY_TARGET)
            # - ad targets: when a ad higher-priority sample drove the (ra, dec)
            #   at the input target catalog creation (identified with PHOT_SELECTION_
            # - djs targets: those are "pure" suprime tractor (ra, dec)
            # if v2=True:
            # we are comparing to rerun tractor catalogs, so the (ra, dec) will not
            #   be an exact matching
            # in addition some targets are not matched:
            # - I427: 6/260
            # - I464: 27/67
            # - I484: 9/420
            # - I505: 18/407
            # - I527: 0/98
            # for those, we let the bricknames, objids, targfns empty

            if not v2:

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

                    phot_sels = d["PHOT_SELECTION"][ii_band][iid][sel_diff].astype(str)
                    phot_sels = np.array(
                        [_.split(";")[0].strip() for _ in phot_sels]
                    )  # keep the first selection

                    is_lowerprio_fa = (
                        (tertiary_targets == "LBG_HYPHQ")
                        | (tertiary_targets == "LBG_HYP")
                        | (tertiary_targets == "LBG_Z2")
                    )

                    is_lowerprio_ad = np.array([_[:4] != "DJS_" for _ in phot_sels]) & (
                        phot_sels != "LAE IA{}".format(band)
                    )

                    log.info("tertiary_targets\t= {}".format(", ".join(tertiary_targets)))
                    log.info(
                        "is_lowerprio_fa\t= {}".format(
                            ", ".join(is_lowerprio_fa.astype(str))
                        )
                    )
                    log.info("phot_sels\t= {}".format(", ".join(phot_sels)))
                    log.info(
                        "is_lowerprio_ad= {}".format(", ".join(is_lowerprio_ad.astype(str)))
                    )
                    log.info(
                        "{}\t{}\tlowerprio_fa={}\tlowerprio_ad={}\tlowerprio_faxtlowerprio_ad={}".format(
                            case,
                            band,
                            is_lowerprio_fa.sum(),
                            is_lowerprio_ad.sum(),
                            ((is_lowerprio_fa) & (is_lowerprio_ad)).sum(),
                        )
                    )
                    assert np.all((is_lowerprio_fa) | (is_lowerprio_ad))

            # fill the values
            iid = ii_band[iid]
            bricknames[iid] = t["BRICKNAME"][iit]
            objids[iid] = t["OBJID"][iit]
            targfns[iid] = fn

        # not_v2 : verify all objects are matched
        # v2 : verify the expected non-matched

        if v2:

            n_nomatch = {
                "I427" : 6,
                "I464" : 27,
                "I484" : 9,
                "I505" : 18,
                "I527" : 0,
            }[band]

        else:

            n_nomatch = 0

        log.info(
            "{}\t{}\tv2={}\texpected no_match: {}/{}".format(
                case,
                band,
                v2,
                n_nomatch,
                d[band].sum(),
            )
        )
        assert (
            (d[band]) & (bricknames == empty_brickname)
        ).sum() == n_nomatch

    return bricknames, objids, targfns
