#!/usr/bin/env python


import os
import subprocess
from glob import glob
from datetime import datetime
import fitsio
import numpy as np

from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.time import Time

from desitarget.targetmask import desi_mask
from desispec.io import read_spectra, write_spectra
from desispec.coaddition import coadd_cameras, coadd_fibermap
from desispec.spectra import stack as spectra_stack
from desispec.fiberbitmasking import (
    get_all_fiberbitmask_with_amp,
    get_all_nonamp_fiberbitmask_val,
    get_justamps_fiberbitmask,
)
from desispec.tsnr import get_ensemble
from desiutil.log import get_logger

log = get_logger()

# allowed values
allowed_imgs = ["odin", "suprime", "clauds"]
allowed_img_cases = {
    "odin": ["cosmos_yr1", "xmmlss_yr2", "cosmos_yr2"],
    "suprime": ["cosmos_yr2", "cosmos_yr3"],
    "clauds": ["cosmos_yr1", "xmmlss_yr2", "cosmos_yr2"],
}
allowed_cases = []
for img in allowed_img_cases:
    allowed_cases += allowed_img_cases[img]
allowed_cases = np.unique(allowed_cases).tolist()


def print_config_infos():
    """
    Print various configuration informations
    """
    # machine
    log.info("NERSC_HOST={}".format(os.getenv("NERSC_HOST")))
    log.info("HOSTNAME={}".format(os.getenv("HOSTNAME")))

    # desispec, desihiz code version/path
    for name in ["desispec", "desihiz"]:
        exec("import {}".format(name))
        log.info(
            "running with {} code version: {}".format(
                name, eval("{}.__version__".format(name))
            )
        )
        log.info(
            "running with {} code path: {}".format(
                name, eval("{}.__path__".format(name))
            )
        )

    #
    log.info("spec_rootdir: {}".format(get_spec_rootdir()))


def get_img_dir(img):
    """
    Get the "root" folder for an imaging survey.

    Args:
        img: element from allowed_imgs (str)

    Returns:
        imgdir: folder path (str)
    """
    assert img in allowed_imgs
    imgdir = os.path.join(os.getenv("DESI_ROOT"), "users", "raichoor", "laelbg", img)
    return imgdir


def get_img_bands(img):
    """
    Get bands for an imaging survey.

    Args:
        img: element from allowed_imgs (str)

    Returns:
        list of bands

    Notes:
        For odin/suprime, it is the selection bands.
        For clauds, it is the set of bands used for the selection.
    """
    assert img in allowed_imgs

    if img == "odin":

        bands = ["N419", "N501", "N673"]

    if img == "suprime":

        bands = ["I427", "I464", "I484", "I505", "I527"]

    if img == "clauds":

        bands = ["UGR", "USGR", "GRI"]

    return bands


def get_img_cases(img):
    """
    Returns the cases (i.e. round of DESI observations) of an imaging survey

    Args:
        img: element from allowed_imgs (str)

    Returns:
        cases: list of the cases (list of str)
    """
    assert img in allowed_imgs
    cases = allowed_img_cases[img]
    return cases


def get_bb_img(fn):
    """
    Get the broad-band imaging for a given target catalog

    Args:
        fn: target catalog full path (str)

    Returns:
        bb_img: imaging string ("LS", "HSC") (str)
    """
    if os.path.basename(fn) in [
        "LAE_Candidates_NB501_v1_targeting.fits.gz",
        "LAE_Candidates_NB673_v0_targeting.fits.gz",
        "ODIN_N419_tractor_DR10_forced_all.fits.gz",
    ]:

        bb_img = "LS"

    elif os.path.basename(fn) in [
        "tractor-xmm-N419-hsc-forced.fits",
        "ODIN_N419_tractor_HSC_forced_all.fits.gz",
        "Subaru_tractor_forced_all.fits.gz",
        "Subaru_tractor_forced_all-redux-20231025.fits",
    ]:

        bb_img = "HSC"

    elif os.path.basename(fn) in [
        "COSMOS_11bands-SExtractor-Lephare.fits",
        "COSMOS_11bands-SExtractor-Lephare-offset.fits",
        "XMMLSS_11bands-SExtractor-Lephare.fits",
        "XMMLSS_11bands-SExtractor-Lephare-offset.fits",
    ]:

        bb_img = "CLAUDS"

    else:

        msg = "unexpected fn: {}".format(fn)
        log.error(msg)
        raise ValueError(msg)

    return bb_img


def get_spec_rootdir():
    """
    Get the root folder for the spectroscopic healpix reductions

    Args:
        None

    Returns:
        folder name (str)
    """
    return os.path.join(os.getenv("DESI_ROOT"), "users", "raichoor", "laelbg")


def get_specprod(case):
    """
    Get the spectroscopic reduction (e.g., daily, iron) for a given case

    Args:
        case: round of DESI observation (str)

    Returns:
        specprod: the spectroscopic production (str)

    Notes:
        Nov. 2024: switch everything to loa
    """
    assert case in allowed_cases

    specprod = "loa"

    return specprod


# same for odin or suprime
def get_specdirs(img, case):
    """
    Get the folder(s) with the spectroscopic (healpix) reduction

    Args:
        img: element from allowed_imgs (str)
        case: round of DESI observation (str)

    Returns:
        specdirs: list the folder full path (list of str)
    """
    assert img in allowed_imgs
    assert case in allowed_img_cases[img]

    spec_rootdir = get_spec_rootdir()
    specprod = get_specprod(case)

    if img == "odin":

        if case == "cosmos_yr1":

            casedirs = ["tileid82636-thru20220324-loa"]

        if case == "xmmlss_yr2":

            casedirs = ["tertiary18-thru20230112-loa"]

        if case == "cosmos_yr2":

            casedirs = ["tertiary26-thru20230416-loa"]

    if img == "suprime":

        if case == "cosmos_yr2":

            casedirs = ["tertiary26-thru20230416-loa"]

        if case == "cosmos_yr3":

            casedirs = ["tertiary37-thru20240309-loa"]

    if img == "clauds":

        if case == "cosmos_yr1":

            casedirs = [
                "tileid80871-80872-thru20210512-loa",
                "tileid82636-thru20220324-loa",
            ]

        if case == "xmmlss_yr2":

            casedirs = ["tertiary15-thru20221216-loa"]

        if case == "cosmos_yr2":

            casedirs = ["tertiary26-thru20230416-loa"]

    specdirs = [
        os.path.join(spec_rootdir, specprod, "healpix", casedir) for casedir in casedirs
    ]

    return specdirs


# ODIN: https://github.com/moustakas/fastspecfit-projects/blob/main/tertiary/deep-photometry.ipynb
# LS:   https://www.legacysurvey.org/dr9/catalogs/#galactic-extinction-coefficients
# HSC:  https://hsc-release.mtk.nao.ac.jp/schema/#pdr3.pdr3_wide.forced
def get_ext_coeffs(img):
    """
    Get the per-filter Alam extinction coefficients to correct for Galactic dust.

    Args:
        img: element from allowed_imgs (str)

    Returns:
        mydict: dictionary with a structure like {band: coeff} (dictionary)

    Notes:
        ODIN: https://github.com/moustakas/fastspecfit-projects/blob/main/tertiary/deep-photometry.ipynb
        LS: https://www.legacysurvey.org/dr9/catalogs/#galactic-extinction-coefficients
        HSC: https://hsc-release.mtk.nao.ac.jp/schema/#pdr3.pdr3_wide.forced
        SUPRIME: /global/cfs/cdirs/desi/users/raichoor/laelbg/suprime/phot/Subaru_COSMOS_all.ipynb
        CLAUDS:
            u, uS: inferred
            filter curves from http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse:
            see email from Eddie from 03/24/2023
            hsc: https://hsc-release.mtk.nao.ac.jp/schema/#pdr3.pdr3_dud_rev.forced
    """
    assert img in allowed_imgs

    tmpdict = {
        "ODIN": {"N419": 4.324, "N501": 3.540, "N673": 2.438},
        "HSC": {
            "G": 3.240,
            "R": 2.276,
            "R2": 2.276,
            "I": 1.633,
            "I2": 1.633,
            "Z": 1.263,
        },
        "LS": {"G": 3.214, "R": 2.165, "I": 1.592, "Z": 1.211},
        "SUPRIME": {
            "I427": 4.202,
            "I464": 3.894,
            "I484": 3.694,
            "I505": 3.490,
            "I527": 3.304,
        },
        "CLAUDS": {
            "U": 4.12,
            "US": 4.01,
            "G": 3.24,
            "R": 2.276,
            "I": 1.633,
            "Z": 1.263,
            "Y": 1.075,
        },
    }

    if img == "odin":

        mydict = {_: tmpdict[_] for _ in ["ODIN", "HSC", "LS"]}

    if img == "suprime":

        mydict = {_: tmpdict[_] for _ in ["SUPRIME", "HSC"]}

    if img == "clauds":

        mydict = {_: tmpdict[_] for _ in ["CLAUDS"]}

    return mydict


def get_cosmos2020_fn(case):
    """
    Get the Weaver+22 COSMOS2020 FARMER catalog full path

    Args:
        case: round of DESI observation (str)

    Returns:
        fn: full path of the catalog (str)
    """
    assert case in allowed_cases
    fn = None

    if case[:6] == "cosmos":

        fn = os.path.join(
            os.getenv("DESI_ROOT"),
            "users",
            "raichoor",
            "cosmos",
            "COSMOS2020_FARMER_R1_v2.0.fits",
        )

    return fn


def get_clauds_fn(case, v2=False, uband="u"):
    """
    Get the Desprez+23 CLAUDS SExtractor catalog full path

    Args:
        case: round of DESI observation (str)
        v2 (optional, defaults to False): if True, use custom catalogs
            with per-HSC pointing photometric offset on the Desprez+23 catalogs,
            (see https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=7493)
            (bool)
        uband (optional, defaults to "u"): required if offset=True;
            can be either "u" or "uS" (str)

    Returns:
        fn: full path of the catalog (str)
    """
    assert case in allowed_cases

    fn = None
    claudsdir = os.path.join(get_img_dir("clauds"), "phot")

    field = case[:6]

    if v2:

        v2_str = "-offset"

    else:

        v2_str = ""

    fn = os.path.join(
        claudsdir, "{}_11bands-SExtractor-Lephare{}.fits".format(field.upper(), v2_str)
    )

    return fn


def get_vi_fns(img):
    """
    Get the Visual Inspection catalogs for a given imaging

    Args:
        img: element from allowed_imgs (str)

    Returns:
        fns: list of the full paths to catalogs (list of str)
    """
    assert img in allowed_imgs

    fn = None

    if img == "odin":

        mydir = os.path.join(get_img_dir("odin"), "vi")
        fns = [
            os.path.join(mydir, "FINAL_VI_ODIN_N501_20231012.fits"),
            os.path.join(mydir, "FINAL_VI_ODIN_N419_20231129.fits"),
            os.path.join(mydir, "FINAL_VI_ODIN_N673_20240507.fits"),
        ]

    if img == "suprime":

        mydir = os.path.join(get_img_dir("suprime"), "vi")
        fns = [
            os.path.join(mydir, "FINAL_VI_Subaru_COSMOS_v20230803.fits.gz"),
            os.path.join(mydir, "all_VI_Subaru_COSMOS_tertiary37.fits.gz"),
            os.path.join(mydir, "all_VI_COSMOS_LBG_CLAUDS.fits.gz"),
        ]

    if img == "clauds":

        mydir = os.path.join(get_img_dir("clauds"), "vi")
        fns = [os.path.join(mydir, "desi-vi-truth-table_fuji_V4.ecsv")]

    return fns


def read_vi_fn(fn):
    """
    Read a VI catalog, and execute few commands to clean/homogenize.

    Args:
        fn: full path to the VI file (str)

    Returns:
        d: astropy Table()
    """

    basename = os.path.basename(fn)

    if basename.split(os.path.extsep)[-1] == "ecsv":
        d = Table.read(fn)
    else:
        d = Table(fitsio.read(fn))
    log.info("{}\t: read {} rows".format(basename, len(d)))

    # expected file names per img
    basenames = {
        img: [os.path.basename(_) for _ in get_vi_fns(img)] for img in allowed_imgs
    }

    # odin, suprime
    # - suprime: add dummy VI_SPECTYPE_FINAL
    # - rename columns
    if (basename in basenames["odin"]) | (basename in basenames["suprime"]):
        if basename in [
            "all_VI_Subaru_COSMOS_tertiary37.fits.gz",
            "all_VI_COSMOS_LBG_CLAUDS.fits.gz",
        ]:
            key_olds = ["VI_TARGETID", "VI_COMMENT"]
            key_news = ["TARGETID", "VI_COMMENTS"]
        else:
            key_olds = [
                "VI_Z_FINAL",
                "VI_QUALITY_FINAL",
                "VI_SPECTYPE_FINAL",
                "VI_COMMENTS_FINAL",
            ]
            key_news = ["VI_Z", "VI_QUALITY", "VI_SPECTYPE", "VI_COMMENTS"]
            if basename in basenames["odin"]:
                key_olds = ["VI_TARGETID"] + key_olds
                key_news = ["TARGETID"] + key_news
            if basename in basenames["suprime"]:
                d["VI_SPECTYPE_FINAL"] = np.zeros(len(d), dtype="<U10")
                log.info("{}\t: add dummy VI_SPECTYPE_FINAL".format(basename))
        for key_old, key_new in zip(key_olds, key_news):
            d[key_old].name = key_new
            log.info("{}\t: rename {} to {}".format(basename, key_old, key_new))

    # clauds
    # - remove duplicates
    # - rename columns
    if basename in basenames["clauds"]:
        sel = d["DUPL"] != "d"
        log.info("{}\t: remove {} duplicates".format(basename, (~sel).sum()))
        d = d[sel]
        for key_old, key_new in zip(
            ["VI_quality", "VI_z", "VI_spectype", "VI_comment"],
            ["VI_QUALITY", "VI_Z", "VI_SPECTYPE", "VI_COMMENTS"],
        ):
            d[key_old].name = key_new
            log.info("{}\t: rename {} to {}".format(basename, key_old, key_new))

    # we want no duplicates at this point
    assert np.unique(d["TARGETID"]).size == len(d)

    return d


def mtime_infos(fn):
    """
    Get the timestamp info of a file (and from the source file if it is a symlink)

    Args:
        fn: file name (str)

    Returns:
        fn_mtime: timestamp of fn (datetime object)
        src_fn: source file path, if fn a symlink; None if fn is not a symlink (str)
        src_mtime: timestamp of the source fileif fn a symlink; None if fn is not a symlink (datetime object)
    """
    outfmt = "%Y-%m-%dT%H:%M:%S"

    if os.path.islink(fn):

        fn_mtime = datetime.fromtimestamp(os.lstat(fn).st_mtime)
        src_fn = os.readlink(fn)
        src_mtime = datetime.fromtimestamp(os.path.getmtime(src_fn))

    else:

        fn_mtime = datetime.fromtimestamp(os.path.getmtime(fn))
        src_fn = None
        src_mtime = None

    return fn_mtime, src_fn, src_mtime


def check_coaddfn_cframes_timestamp(case, coaddfn):
    """
    Compares the coadd timestamps with the exposure (cframe) timestamps.
    Issues a log.warning() in case the exposures are more recent than the coadd
        (which means that the coadd is based from deprecated exposures)

    Args:
        case: round of DESI observation (str)
        coaddfn: full path to the coadd filename (str)

    Returns:
        -

    Notes:
        Typical scheme we try to catch here is:
        - we create some coadds
        - then some exposures are reprocessed because of some issues
            identified in https://github.com/desihub/desisurveyops/issues
    """
    assert case in allowed_cases

    spec_rootdir, specprod = get_spec_rootdir(), get_specprod(case)
    outfmt = "%Y-%m-%dT%H:%M:%S"

    spectrafn = coaddfn.replace("coadd", "spectra")
    hdr = fits.getheader(spectrafn, "PRIMARY")
    cframefns = [
        hdr["INFIL{:03d}".format(_)]
        for _ in range(1000)
        if "INFIL{:03d}".format(_) in hdr
    ]

    for fn in cframefns:

        # the recorded fn is like SPECPROD/exposures/..., in which case
        #   I guess one is supposed to use the DESI_SPECTRO_REDUX and SPECPROD
        #   keywords to get the full path
        # though I m not 100% sure of the correctness of that for custom runs
        #   as for here
        # so I assume that the exposures come from spec_rootdir/specprod/exposures/...
        if fn[:8] == "SPECPROD":
            fn = os.path.join(
                spec_rootdir, specprod, os.path.join(*fn.split(os.path.sep)[1:])
            )

        fn_mtime, src_fn, src_mtime = mtime_infos(fn)

        if src_fn is None:
            mtime = fn_mtime
        # if link, check if source cframe is older/newer than the coadd (should be older...)
        else:
            fn, mtime = src_fn, src_mtime

        spectra_mtime = datetime.fromtimestamp(os.path.getmtime(spectrafn))

        if Time(mtime).mjd > Time(spectra_mtime).mjd:

            log.warning(
                "{}={},\t{}\t{}".format(
                    spectrafn,
                    spectra_mtime.strftime(outfmt),
                    fn,
                    mtime.strftime(outfmt),
                )
            )


def get_coaddfns(img, case):
    """
    Get the (per-healpix) coadd filenames for a given {img, case}

    Args:
        img: element from allowed_imgs (str)
        case: round of DESI observation (str)

    Returns:
        coaddfns: list of filenames (list of str)

    Notes:
        This uses get_specdirs(), i.e. gets coadds from custom reductions,
            as the daily pipeline does not produce healpix reductions
    """
    assert img in allowed_imgs
    assert case in allowed_img_cases[img]

    specdirs = get_specdirs(img, case)

    if "cosmos" in case:
        pattern = "coadd-27???.fits"
    elif "xmmlss" in case:
        pattern = "coadd-17???.fits"
    else:
        msg = "unexpected case = {}".format(case)
        log.error(msg)
        raise ValueError(msg)

    coaddfns = np.hstack(
        [sorted(glob(os.path.join(specdir, pattern))) for specdir in specdirs]
    )

    for coaddfn in coaddfns:

        log.info(coaddfn)
        check_coaddfn_cframes_timestamp(case, coaddfn)

    return coaddfns


def get_init_infos(img, nrows):
    """
    Get the dictionary with zero-like value arrays for the columns used downstream
    This ensures we are using the same datamodel across the code

    Args:
        img: element from allowed_imgs (str)
        nrows: nband-element array with the number of objects per band
                (for ODIN: 3-element array; for SUPRIME: 5-element array)

    Returns:
        mydict: a dictionary with the following structure:
            {
                band : {
            "TARGETID": np.zeros(nrow, dtype=int),
            "TERTIARY_TARGET": np.zeros(nrow, dtype="S30"),
            "PHOT_RA": np.zeros(nrow, dtype=np.dtype("f8")),
            "PHOT_DEC": np.zeros(nrow, dtype=np.dtype("f8")),
            "PHOT_SELECTION": np.zeros(nrow, dtype="<U100"),
            }
        }
    """
    assert img in allowed_imgs
    bands = get_img_bands(img)

    assert len(nrows) == len(bands)

    mydict = {
        band: {
            "TARGETID": np.zeros(nrow, dtype=int),
            "TERTIARY_TARGET": np.zeros(nrow, dtype="S30"),
            "PHOT_RA": np.zeros(nrow, dtype=np.dtype("f8")),
            "PHOT_DEC": np.zeros(nrow, dtype=np.dtype("f8")),
            "PHOT_SELECTION": np.zeros(nrow, dtype="<U100"),
        }
        for band, nrow in zip(bands, nrows)
    }

    return mydict


def get_img_infos(img, case, stdsky):
    """
    For a given {img, case}, returns the information from the photometric
        catalog used for target selection

    Args:
        img: element from allowed_imgs (str)
        case: round of DESI observation (str)

    Returns:
        mydict: a dictionary with a structure like get_init_infos()

    Notes:
        The per-band infos are: TARGETID, TERTIARY_TARGET, PHOT_RA, PHOT_DEC, PHOT_SELECTION
    """
    assert img in allowed_imgs
    assert case in get_img_cases(img)

    # if stdsky: we return an empty dict
    if stdsky:

        bands = get_img_bands(img)
        nrows = [0 for _ in bands]
        mydict = get_init_infos(img, nrows)

    # else: we grab laes
    else:

        if img == "odin":

            from desihiz.hizmerge_odin import (
                get_odin_cosmos_yr1_infos,
                get_odin_xmmlss_yr2_infos,
                get_odin_cosmos_yr2_infos,
            )

            if case == "cosmos_yr1":
                mydict = get_odin_cosmos_yr1_infos()
            if case == "xmmlss_yr2":
                mydict = get_odin_xmmlss_yr2_infos()
            if case == "cosmos_yr2":
                mydict = get_odin_cosmos_yr2_infos()

        if img == "suprime":

            from desihiz.hizmerge_suprime import (
                get_suprime_cosmos_yr2_infos,
                get_suprime_cosmos_yr3_infos,
            )

            if case == "cosmos_yr2":
                mydict = get_suprime_cosmos_yr2_infos()
            if case == "cosmos_yr3":
                mydict = get_suprime_cosmos_yr3_infos()

        if img == "clauds":

            from desihiz.hizmerge_clauds import (
                get_clauds_cosmos_yr1_infos,
                get_clauds_xmmlss_yr2_infos,
                get_clauds_cosmos_yr2_infos,
            )

            if case == "cosmos_yr1":
                mydict = get_clauds_cosmos_yr1_infos()
            if case == "xmmlss_yr2":
                mydict = get_clauds_xmmlss_yr2_infos()
            if case == "cosmos_yr2":
                mydict = get_clauds_cosmos_yr2_infos()

    bands = get_img_bands(img)
    for band in bands:
        log.info(
            "\t{}{}\t{}\tfound {} targets".format(
                img, case, band, len(mydict[band]["TARGETID"])
            )
        )

    return mydict


#
def read_targfn(targfn):
    """
    Read a targets file catalog, with few reformatting of the column
        names to ensure consistency across all {img, case}

    Args:
        targfn: full path to the catalog (str)

    Returns:
        p: astropy.table.Table() array
    """

    basename = os.path.basename(targfn)

    # CLAUDS official catalogs cannot be read with Table(fitsio.read())..
    if "11bands-SExtractor-Lephare" in basename:
        p = Table(fits.open(targfn)[1].data)
    else:
        p = Table(fitsio.read(targfn))

    # change colnames to upper cases
    for key in p.colnames:

        p[key].name = key.upper()

    # ODIN: fix/homogenize some column names

    if basename in [
        "LAE_Candidates_NB501_v1_targeting.fits.gz",
        "LAE_Candidates_NB673_v0_targeting.fits.gz",
    ]:

        old_roots = ["FORCED_MEAN_FLUX"]
        new_roots = ["FLUX"]

        for key in p.colnames:

            for old_root, new_root in zip(old_roots, new_roots):

                if old_root in key:

                    p[key].name = key.replace(old_root, new_root)
                    log.info(
                        "{}:\trename {} to {}".format(
                            basename,
                            key,
                            key.replace(old_root, new_root),
                        )
                    )

    if basename in [
        "ODIN_N419_tractor_DR10_forced_all.fits.gz",
        "ODIN_N419_tractor_HSC_forced_all.fits.gz",
        "tractor-xmm-N419-hsc-forced.fits",
    ]:

        old_roots = ["FORCED_FLUX", "FORCED_PSFDEPTH", "FORCED_GALDEPTH"]
        new_roots = ["FLUX", "PSFDEPTH", "GALDEPTH"]

        for key in p.colnames:

            for old_root, new_root in zip(old_roots, new_roots):

                if old_root in key:

                    p[key].name = key.replace(old_root, new_root)
                    log.info(
                        "{}:\trename {} to {}".format(
                            basename,
                            key,
                            key.replace(old_root, new_root),
                        )
                    )

    # SUPRIME: fix/homogenize some column names
    if basename in [
        "Subaru_tractor_forced_all.fits.gz",
        "Subaru_tractor_forced_all-redux-20231025.fits",
    ]:

        old_roots = ["I_A_L{}".format(band[1:]) for band in get_img_bands("suprime")]
        new_roots = [band for band in get_img_bands("suprime")]
        old_roots += ["FORCED_FLUX", "FORCED_PSFDEPTH", "FORCED_GALDEPTH"]
        new_roots += ["FLUX", "PSFDEPTH", "GALDEPTH"]

        for key in p.colnames:

            for old_root, new_root in zip(old_roots, new_roots):

                if old_root in key:

                    p[key].name = key.replace(old_root, new_root)
                    log.info(
                        "{}:\trename {} to {}".format(
                            basename,
                            key,
                            key.replace(old_root, new_root),
                        )
                    )

    # CLAUDS:
    # - homogenize some column names (EBV)
    # - convert ext.corr-mags and magerr to nanomaggies
    # - convert FLAG_FIELD_BINARY into a bit-coded int,
    #       as it s a pain to handle a 7-element array downstream..
    if basename in [
        "COSMOS_11bands-SExtractor-Lephare.fits",
        "COSMOS_11bands-SExtractor-Lephare-offset.fits",
        "XMMLSS_11bands-SExtractor-Lephare.fits",
        "XMMLSS_11bands-SExtractor-Lephare-offset.fits",
    ]:

        for key in p.colnames:

            p[key].name = key.upper()

        p["EB_V"].name = "EBV"
        log.info("{}: rename EB_V to EBV".format(basename))

        for band in ["U", "US", "G", "R", "I", "Z", "Y"]:

            # initializing with non-valid photometry values
            p["FLUX_{}".format(band)] = -99.0
            p["FLUX_IVAR_{}".format(band)] = 0.0
            # re-include gal. extinction
            mags = p[band] + get_ext_coeffs("clauds")["CLAUDS"][band] * p["EBV"]
            # valid values
            sel = (p[band] > 0) & (p["{}_ERR".format(band)] > 0)
            # flux and flux_ivar in nanomaggies
            p["FLUX_{}".format(band)][sel] = 10 ** (-0.4 * (mags[sel] - 22.5))
            p["FLUX_IVAR_{}".format(band)][sel] = (
                np.log(10)
                / 2.5
                * p["{}_ERR".format(band)][sel]
                * p["FLUX_{}".format(band)][sel]
            ) ** -2.0
            log.info(
                "{}: convert {} and {}_ERR to (reddened) FLUX_{} (nanomaggies) and FLUX_IVAR_{}".format(
                    basename, band, band, band, band
                )
            )

        p["FLAG_FIELD_BINARY_INT"] = 0

        for i in range(p["FLAG_FIELD_BINARY"].shape[1]):

            p["FLAG_FIELD_BINARY_INT"] += 2 ** p["FLAG_FIELD_BINARY"][:, i].astype(int)

        p.remove_column("FLAG_FIELD_BINARY")
        log.info(
            "{}: remove FLAG_FIELD_BINARY, add FLAG_FIELD_BINARY_INT".format(basename)
        )

    log.info("{} colnames: {}".format(basename, ", ".join(p.colnames)))

    return p


# https://desi.lbl.gov/svn/docs/technotes/targeting/target-truth/trunk/python/match_coord.py
# slightly edited (plot_q remove; u => units)
def match_coord(
    ra1,
    dec1,
    ra2,
    dec2,
    search_radius=1.0,
    nthneighbor=1,
    verbose=True,
    keep_all_pairs=False,
):
    """
    Match objects in (ra2, dec2) to (ra1, dec1).

    Inputs:
        RA and Dec of two catalogs;
        search_radius: in arcsec;
        (Optional) keep_all_pairs: if true, then all matched pairs are kept; otherwise, if more than
        one object in t2 is match to the same object in t1 (i.e. double match), only the closest pair
        is kept.

    Outputs:
        idx1, idx2: indices of matched objects in the two catalogs;
        d2d: distances (in arcsec);
        d_ra, d_dec: the differences (in arcsec) in RA and Dec; note that d_ra is the actual angular
        separation;
    """
    t1 = Table()
    t2 = Table()
    # protect the global variables from being changed by np.sort
    ra1, dec1, ra2, dec2 = map(np.copy, [ra1, dec1, ra2, dec2])
    t1["ra"] = ra1
    t2["ra"] = ra2
    t1["dec"] = dec1
    t2["dec"] = dec2
    t1["id"] = np.arange(len(t1))
    t2["id"] = np.arange(len(t2))
    # Matching catalogs
    sky1 = SkyCoord(ra1 * units.degree, dec1 * units.degree, frame="icrs")
    sky2 = SkyCoord(ra2 * units.degree, dec2 * units.degree, frame="icrs")
    idx, d2d, d3d = sky2.match_to_catalog_sky(sky1, nthneighbor=nthneighbor)
    # This finds a match for each object in t2. Not all objects in t1 catalog are included in the result.

    # convert distances to numpy array in arcsec
    d2d = np.array(d2d.to(units.arcsec))
    matchlist = d2d < search_radius
    if np.sum(matchlist) == 0:
        if verbose:
            log.info("0 matches")
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([]),
            np.array([]),
            np.array([]),
        )
    t2["idx"] = idx
    t2["d2d"] = d2d
    t2 = t2[matchlist]
    init_count = np.sum(matchlist)
    # --------------------------------removing doubly matched objects--------------------------------
    # if more than one object in t2 is matched to the same object in t1, keep only the closest match
    if not keep_all_pairs:
        t2.sort("idx")
        i = 0
        while i <= len(t2) - 2:
            if t2["idx"][i] >= 0 and t2["idx"][i] == t2["idx"][i + 1]:
                end = i + 1
                while end + 1 <= len(t2) - 1 and t2["idx"][i] == t2["idx"][end + 1]:
                    end = end + 1
                findmin = np.argmin(t2["d2d"][i : end + 1])
                for j in range(i, end + 1):
                    if j != i + findmin:
                        t2["idx"][j] = -99
                i = end + 1
            else:
                i = i + 1

        mask_match = t2["idx"] >= 0
        t2 = t2[mask_match]
        t2.sort("id")
        if verbose:
            log.info("Doubly matched objects = %d" % (init_count - len(t2)))
    # -----------------------------------------------------------------------------------------
    if verbose:
        log.info("Final matched objects = %d" % len(t2))
    # This rearranges t1 to match t2 by index.
    t1 = t1[t2["idx"]]
    d_ra = (t2["ra"] - t1["ra"]) * 3600.0  # in arcsec
    d_dec = (t2["dec"] - t1["dec"]) * 3600.0  # in arcsec
    ##### Convert d_ra to actual arcsecs #####
    mask = d_ra > 180 * 3600
    d_ra[mask] = d_ra[mask] - 360.0 * 3600
    mask = d_ra < -180 * 3600
    d_ra[mask] = d_ra[mask] + 360.0 * 3600
    d_ra = d_ra * np.cos(t1["dec"] / 180 * np.pi)
    ##########################################
    return (
        np.array(t1["id"]),
        np.array(t2["id"]),
        np.array(t2["d2d"]),
        np.array(d_ra),
        np.array(d_dec),
    )


def add_img_infos(img, d, mydict):
    """
    Propagates the values of the photometric catalog to the stacked FIBERMAP table

    Args:
        img: element from allowed_imgs (str)
        d: table from the stacked FIBERMAP
        mydict: dictionary (from get_img_infos())

    Returns:
        d: input d, with additional columns: TERTIARY_TARGET, PHOT_RA, PHOT_DEC, PHOT_SELECTION
    """

    assert img in allowed_imgs

    bands = get_img_bands(img)

    # add img filters used for targeting
    for band in bands:

        d[band] = False
        sel = np.in1d(d["TARGETID"], mydict[band]["TARGETID"])
        d[band][sel] = True

    # add img tertiary_target/ra/dec/selection information
    for key in ["TERTIARY_TARGET", "PHOT_RA", "PHOT_DEC", "PHOT_SELECTION"]:

        d[key] = np.zeros_like(mydict[bands[0]][key], shape=(len(d),))

    for band in bands:

        ## brute-force loop in case of possible duplicates in tids
        ##  but not larger numbers, so ok
        for tid, tertiary_target, ra, dec, sel in zip(
            mydict[band]["TARGETID"],
            mydict[band]["TERTIARY_TARGET"],
            mydict[band]["PHOT_RA"],
            mydict[band]["PHOT_DEC"],
            mydict[band]["PHOT_SELECTION"],
        ):
            ii = np.where(d["TARGETID"] == tid)[0]

            if ii.size > 0:

                old_tertiary_targets = d["TERTIARY_TARGET"][ii]
                old_tertiary_targets = np.unique(
                    [_ for _ in old_tertiary_targets if _.strip() != ""]
                )
                old_ras, old_decs = d["PHOT_RA"][ii], d["PHOT_DEC"][ii]
                old_ras = old_ras[old_ras != 0]
                old_decs = old_decs[old_decs != 0]
                old_sels = d["PHOT_SELECTION"][ii]
                old_sels = np.unique([_ for _ in old_sels if _.strip() != ""])

                if old_sels.size > 0:

                    assert (old_tertiary_targets.size == 1) & (
                        old_tertiary_targets[0] == tertiary_target
                    )
                    assert (old_ras.size == 1) & (old_ras[0] == ra)
                    assert (old_decs.size == 1) & (old_decs[0] == dec)
                    assert (old_sels.size == 1) & (old_sels[0] == sel)

                d["TERTIARY_TARGET"][ii] = tertiary_target
                d["PHOT_RA"][ii] = ra
                d["PHOT_DEC"][ii] = dec
                d["PHOT_SELECTION"][ii] = sel
    return d


# generate coadd from the cframe files
# tids: ! list !
def create_coadd_merge(cofn, tids, stdsky):
    """
    Get the Spectra() object with coadded-spectra for a list of TARGETIDs

    Args:
        cofn: coadd full path (str)
        tids: list of TARGETIDs (list of int)
        stdsky: are we dealing with standard stars (STD) + sky (SKY)? (boolean)

    Returns:
        s: a Spectra() object

    Notes:
        See https://desidatamodel.readthedocs.io/en/latest/DESI_SPECTRO_REDUX/SPECPROD/healpix/SURVEY/PROGRAM/PIXGROUP/PIXNUM/coadd-SURVEY-PROGRAM-PIXNUM.html
        We add a HEALPIX and COADDFN columns to s.fibermap
    """
    s = read_spectra(cofn)
    fm = s.fibermap

    if stdsky:

        # add std
        sel = (
            fm["DESI_TARGET"] & desi_mask["STD_FAINT"]
        ) > 0  # bright is subsample of faint
        tids = tids + fm["TARGETID"][sel].tolist()
        # add skies
        sel = fm["OBJTYPE"] == "SKY"
        tids = tids + fm["TARGETID"][sel].tolist()

    n = np.in1d(fm["TARGETID"], tids).sum()

    log.info(
        "found {}/{} requested TARGETIDs in {}".format(
            n, len(tids), os.path.basename(cofn)
        )
    )

    if n == 0:

        return None

    else:

        s = s.select(targets=tids)
        # coadd cameras
        s = coadd_cameras(s)
        # healpix + coaddfn
        hdr = fits.getheader(cofn, 0)
        if "HPXPIXEL" in hdr:
            s.fibermap["HEALPIX"] = hdr["HPXPIXEL"]
        else:
            msg = "no HPXPIXEL keyword in the header of {}".format(cofn)
            log.error(msg)
            raise ValueError(ms)
        s.fibermap["COADDFN"] = cofn

        return s


# FIBERMAP columns to remove
def get_fm_tractor_cols2rmv():
    """
    Get the list of tractor columns we remove downstream

    Args:
        None

    Returns:
        rmvcols: list of column names to remove (list of str)
    """
    rmvcols = [
        "RELEASE",
        "BRICKNAME",
        "BRICKID",
        "BRICK_OBJID",
        "MORPHTYPE",
        "FLUX_G",
        "FLUX_R",
        "FLUX_Z",
        "FLUX_W1",
        "FLUX_W2",
        "FLUX_IVAR_G",
        "FLUX_IVAR_R",
        "FLUX_IVAR_Z",
        "FLUX_IVAR_W1",
        "FLUX_IVAR_W2",
        "FIBERFLUX_G",
        "FIBERFLUX_R",
        "FIBERFLUX_Z",
        "FIBERTOTFLUX_G",
        "FIBERTOTFLUX_R",
        "FIBERTOTFLUX_Z",
        "MASKBITS",
        "SERSIC",
        "SHAPE_R",
        "SHAPE_E1",
        "SHAPE_E2",
        "REF_ID",
        "REF_CAT",
        "GAIA_PHOT_G_MEAN_MAG",
        "GAIA_PHOT_BP_MEAN_MAG",
        "GAIA_PHOT_RP_MEAN_MAG",
        "PARALLAX",
        "PHOTSYS",
    ]
    return rmvcols


# fix some columns in the FIBERMAP
#   due to some bugs in the desispec code
#   when the code was run
def fix_fibermap(fm, exp_fm):
    """
    Fix some columns in the FIBERMAP, which can be bugged because
        of bugs in the desispec code when it was run

    Args:
        fm: FIBERMAP table
        exp_fm: EXP_FIBERMAP table

    Returns:
        fm: FIBERMAP table with fixed columns (and additional ORIG_xxx
            columns if some fix has actually been done)

    Note:
        This has been fixed in the desispec code around Aug-Sep 2023;
            so spectro. data processed later than that should
            have no columns to fix; but we keep the function to be on
            the safe side
    """
    #
    # generate FIBERMAP, EXP_FIBERMAP
    new_fm, new_exp_fm = coadd_fibermap(exp_fm)
    #
    # first: verify EXP_FIBERMAP is not changed
    assert np.all(exp_fm.colnames == new_exp_fm.colnames)
    for k in exp_fm.colnames:
        assert np.all(exp_fm[k] == new_exp_fm[k])
    #
    # then: FIBERMAP
    ## columns which are not supposed to change
    ks = [
        "TARGETID",
        "SUBPRIORITY",
        "PLATE_RA",
        "PLATE_DEC",
        "COADD_NUMEXP",
        "COADD_EXPTIME",
        "COADD_NUMNIGHT",
        "COADD_NUMTILE",
    ]

    for k in ks:

        assert np.all(fm[k] == new_fm[k])

    ## columns which *can* change
    ks = [
        "COADD_FIBERSTATUS",
        "MEAN_DELTA_X",
        "RMS_DELTA_X",
        "MEAN_DELTA_Y",
        "RMS_DELTA_Y",
        "MEAN_PSF_TO_FIBER_SPECFLUX",
        "MEAN_FIBER_RA",
        "STD_FIBER_RA",
        "MEAN_FIBER_DEC",
        "STD_FIBER_DEC",
    ]

    for k in ks:

        ndiff = (new_fm[k] != fm[k]).sum()
        log.info("{}\t{}".format(ndiff, k))

        if ndiff > 0:

            fm["ORIG_{}".format(k)] = fm[k]
            fm[k] = new_fm[k]

    return fm


# https://github.com/desihub/desispec/blob/cfdbdab7444dbaaba6fd49ed813663ebbe401793/py/desispec/coaddition.py#L250-L257
def get_csv_expids(fm, exp_fm):
    """
    Get for each row the list of exposures that went in for a coadd
    Args:
        fm: FIBERMAP table
        exp_fm: EXP_FIBERMAP table

    Returns:
        csv_expids: comma-separated list of EXPIDs (array of str)
    """
    # - Only a subset of "good" FIBERSTATUS flags are included in the coadd
    fiberstatus_nonamp_bits = get_all_nonamp_fiberbitmask_val()
    fiberstatus_amp_bits = get_justamps_fiberbitmask()
    # plan 30 6-digits exposures (tertiary26 has 27 exposures max)
    csv_expids = np.zeros(len(fm), dtype="|U209")

    for i, (tid, conexp) in enumerate(zip(fm["TARGETID"], fm["COADD_NUMEXP"])):

        sel = exp_fm["TARGETID"] == tid
        fsts, expids = exp_fm["FIBERSTATUS"][sel], exp_fm["EXPID"][sel]
        assert np.unique(expids).size == len(expids)

        nonamp_fiberstatus_flagged = (fsts & fiberstatus_nonamp_bits) > 0
        allamps_flagged = (fsts & fiberstatus_amp_bits) == fiberstatus_amp_bits
        good_coadds = np.bitwise_not(nonamp_fiberstatus_flagged | allamps_flagged)
        assert good_coadds.sum() == conexp

        expids = expids[good_coadds]
        csv_expids[i] = ",".join(expids.astype(str))

    return csv_expids


def get_expids(img, case):
    """
    Get the exposures that were used to generate a coadd,
        along with their properties

    Args:
        img: element from allowed_imgs (str)
        case: round of DESI observation (str)

    Returns:
        d: the exposures-{specprod}.csv table cut on the relevant exposures (table)
    """
    assert img in allowed_imgs
    assert case in get_img_cases(img)

    specprod = get_specprod(case)

    fn = os.path.join(
        os.getenv("DESI_ROOT"),
        "spectro",
        "redux",
        specprod,
        "exposures-{}.csv".format(specprod),
    )

    d = Table.read(fn)
    specdirs = get_specdirs(img, case)
    sel = np.zeros(len(d), dtype=bool)
    for specdir in specdirs:
        fn = os.path.join(specdir, "exposures.fits")
        expids = Table.read(fn, "EXPOSURES")["EXPID"]
        sel2 = np.in1d(d["EXPID"], expids)
        assert sel2.sum() == expids.size
        sel |= sel2
    d = d[sel]

    return d


def get_phot_fns(img, case, band, photdir=None, v2=None):
    """
    Get the photometric tractor file name used for the target selection

    Args:
        img: element from allowed_imgs (str)
        case: round of DESI observation (str)
        photdir (optional, defaults to $DESI_ROOT/users/raichoor/laelbg/{img}/phot):
            folder where the files are
        v2 (optional, defaults to False): for img=suprime, clauds, if True, use custom catalogs
            - suprime: Dustin's rerun from 20231025
            - clauds: with per-HSC pointing photometric offset on the Desprez+23 catalogs,
            (see https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=7493)
            (bool)

    Returns:
        mydict: a dictionary with a structure like {case_band: fn}
    """
    assert img in allowed_imgs
    assert case in get_img_cases(img)

    if photdir is None:

        photdir = os.path.join(get_img_dir(img), "phot")

    # odin
    if img == "odin":

        mydict = {
            "cosmos_yr1_N501": [
                os.path.join(photdir, "LAE_Candidates_NB501_v1_targeting.fits.gz")
            ],
            "cosmos_yr1_N673": [
                os.path.join(photdir, "LAE_Candidates_NB673_v0_targeting.fits.gz")
            ],
            "xmmlss_yr2_N419": [
                os.path.join(photdir, "tractor-xmm-N419-hsc-forced.fits")
            ],
            # order matters!
            # - odin/hsc has been used first to select targets
            # - then targets have been completed by odin/dr10
            # see Arjun s email from 08/17/23 7:59AM pacific
            "cosmos_yr2_N419": [
                os.path.join(photdir, "ODIN_N419_tractor_HSC_forced_all.fits.gz"),
                os.path.join(photdir, "ODIN_N419_tractor_DR10_forced_all.fits.gz"),
            ],
            "cosmos_yr2_N501": [
                os.path.join(photdir, "LAE_Candidates_NB501_v1_targeting.fits.gz")
            ],
            "cosmos_yr2_N673": [
                os.path.join(photdir, "LAE_Candidates_NB673_v0_targeting.fits.gz")
            ],
        }

    # suprime
    if img == "suprime":

        if v2:

            basefn = "Subaru_tractor_forced_all-redux-20231025.fits"

        else:

            basefn = "Subaru_tractor_forced_all.fits.gz"
        mydict = {}
        for tmpband in get_img_bands("suprime"):
            mydict["cosmos_yr2_{}".format(tmpband)] = [os.path.join(photdir, basefn)]
            # no I427 selection for tertiary37
            if tmpband != "I427":
                mydict["cosmos_yr3_{}".format(tmpband)] = [
                    os.path.join(photdir, basefn)
                ]

    # clauds
    if img == "clauds":

        mydict = {
            "cosmos_yr1_UGR": [get_clauds_fn("cosmos_yr1", v2=v2, uband="u")],
            "cosmos_yr1_GRI": [get_clauds_fn("cosmos_yr1", v2=v2, uband="u")],
            "xmmlss_yr2_USGR": [get_clauds_fn("xmmlss_yr2", v2=v2, uband="uS")],
            "cosmos_yr2_UGR": [get_clauds_fn("cosmos_yr2", v2=v2, uband="u")],
            "cosmos_yr2_USGR": [get_clauds_fn("cosmos_yr2", v2=v2, uband="uS")],
        }

    if "{}_{}".format(case, band) in mydict:

        return mydict["{}_{}".format(case, band)]

    else:

        return None


def get_phot_init_table(img, n):
    """
    Get the initialized table with the photometric columns we will keep,
        so that we use the same datamodel throughout the code

    Args:
        img: element from allowed_imgs (str)
        n: length of the table (int)

    Returns:
        t: the photometric table with zeros_like() columns
    """

    assert img in allowed_imgs
    bands = get_img_bands(img)

    dtype = []

    # columns in common with SPECINFO
    dtype += [
        ("TARGETID", ">i8"),
        ("STD", "|b1"),
        ("SKY", "|b1"),
    ]

    for band in bands:

        dtype.append(
            (band, "|b1"),
        )

    # column to identify the broad-band photometry source
    dtype += [
        ("FILENAME", "S150"),
        ("BB_IMG", "S6"),  # HSC or LS or CLAUDS
    ]

    if img in ["odin", "suprime"]:

        # tractor columns
        # https://github.com/legacysurvey/legacypipe/blob/cbde86f7f78692091fca7b48d423450074aa0472/bin/generate-sweep-files.py#L360-L574
        dtype += [
            ("RELEASE", ">i2"),
            ("BRICKID", ">i4"),
            ("BRICKNAME", "S8"),
            ("OBJID", ">i4"),
            ("BRICK_PRIMARY", "|b1"),
            ("MASKBITS", ">i4"),
            ("FITBITS", ">i2"),
            ("TYPE", "S3"),
            ("RA", ">f8"),
            ("DEC", ">f8"),
            ("EBV", ">f4"),
        ]

        # img fluxs + depths
        for band in bands:

            dtype += [
                ("FLUX_{}".format(band), ">f4"),
                ("FLUX_IVAR_{}".format(band), ">f4"),
                ("FIBERFLUX_{}".format(band), ">f4"),
                ("PSFDEPTH_{}".format(band), ">f4"),
                ("GALDEPTH_{}".format(band), ">f4"),
            ]

        # ls-{dr9.1.1,dr10} or hsc fluxes
        for band in ["G", "R", "R2", "I", "I2", "Z"]:

            dtype += [
                ("FLUX_{}".format(band), ">f4"),
                ("FLUX_IVAR_{}".format(band), ">f4"),
                ("PSFDEPTH_{}".format(band), ">f4"),
                ("GALDEPTH_{}".format(band), ">f4"),
            ]

    if img in ["clauds"]:

        # sextractor columns
        # $DESI_ROOT/users/raichoor/laelbg/clauds/dr/COSMOS_11bands-SExtractor-Lephare.fits
        dtype += [
            ("ID", ">i8"),
            ("RA", ">f8"),
            ("DEC", ">f8"),
            ("MASK", ">i2"),
            ("FLAG_FIELD", ">i2"),
            ("A_WORLD", ">f4"),
            ("B_WORLD", ">f4"),
            ("KRON_RADIUS", ">f4"),
            ("THETA_WORLD", ">f4"),
            ("ELONGATION", ">f4"),
            ("ELLIPTICITY", ">f4"),
            ("EBV", ">f8"),
            ("FWHM_WORLD_HSC_I", ">f4"),
            ("MU_MAX_HSC_I", ">f4"),
            ("CLASS_STAR_HSC_I", ">f4"),
            ("FLUX_RADIUS_0.25_HSC_I", ">f4"),
            ("FLUX_RADIUS_0.5_HSC_I", ">f4"),
            ("FLUX_RADIUS_0.75_HSC_I", ">f4"),
            ("FLUX_U", ">f8"),
            ("FLUX_US", ">f8"),
            ("FLUX_G", ">f8"),
            ("FLUX_R", ">f8"),
            ("FLUX_I", ">f8"),
            ("FLUX_Z", ">f8"),
            ("FLUX_Y", ">f8"),
            ("FLUX_IVAR_U", ">f8"),
            ("FLUX_IVAR_US", ">f8"),
            ("FLUX_IVAR_G", ">f8"),
            ("FLUX_IVAR_R", ">f8"),
            ("FLUX_IVAR_I", ">f8"),
            ("FLUX_IVAR_Z", ">f8"),
            ("FLUX_IVAR_Y", ">f8"),
            ("ST_TRAIL", ">i8"),
            ("CLEAN", "bool"),
            # ("FLAG_FIELD_BINARY", "bool"),
            ("FLAG_FIELD_BINARY_INT", ">i4"),
        ]

    t = Table(np.zeros(n, dtype=dtype))

    return t


def get_phot_table(img, case, specinfo_table, photdir, v2=False):
    """
    Get the photometric information for a given {img, case}

    Args:
        img: element from allowed_imgs (str)
        case: round of DESI observation (str)
        specinfo_table: output of the get_spec_table() function
        photdir (optional, defaults to $DESI_ROOT/users/raichoor/laelbg/{img}/phot):
            folder where the files are
        v2 (optional, defaults to False): for img=suprime or clauds, if True,
            use custom catalogs
            - suprime: Dustin's rerun from 20231025
            - clauds: with per-HSC pointing photometric offset on the Desprez+23 catalogs,
                (see https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=7493)
            (bool)

    Returns:
        d: the phot table, row-matched to specinfo_table (array)

    Notes:
        We add the tractor photometry, and the COSMOS2020 and CLAUDS zphot
    """

    assert img in allowed_imgs
    bands = get_img_bands(img)

    if photdir is None:

        photdir = os.path.join(get_img_dir(img), "phot")

    # initializing
    d = get_phot_init_table(img, len(specinfo_table))

    # get phot infos
    if img == "odin":

        from desihiz.hizmerge_odin import get_odin_phot_infos

        d["BRICKNAME"], d["OBJID"], d["FILENAME"] = get_odin_phot_infos(
            case, specinfo_table, photdir=photdir
        )
        log.info("odin, targfns = {}".format(", ".join(np.unique(d["FILENAME"]))))

    if img == "suprime":

        from desihiz.hizmerge_suprime import get_suprime_phot_infos

        d["BRICKNAME"], d["OBJID"], d["FILENAME"] = get_suprime_phot_infos(
            case, specinfo_table, photdir, v2=v2
        )

    if img == "clauds":

        from desihiz.hizmerge_clauds import get_clauds_phot_infos

        d["ID"], d["FILENAME"] = get_clauds_phot_infos(
            case, specinfo_table, photdir, v2=v2
        )

    # propagating columns from specinfo_table
    keys = ["TARGETID", "STD", "SKY"] + bands + ["CASE"]

    for key in keys:

        d[key] = specinfo_table[key]

    # unique identifier (from a tractor catalog point of view)
    #   note that for odin:
    #       cosmos_yr1, N673 actually have 373 duplicated d_unqids...
    if img in ["odin", "suprime"]:
        d_unqids = np.array(
            ["{}-{}".format(b, o) for b, o in zip(d["BRICKNAME"], d["OBJID"])]
        )
    if img in ["clauds"]:
        d_unqids = d["ID"].copy()

    targfns = np.unique(d["FILENAME"])
    log.info("targfns = {}".format(", ".join(targfns)))
    targfns = [targfn for targfn in targfns if targfn != ""]

    for targfn in targfns:

        # cut d
        iid = np.where(d["FILENAME"] == targfn)[0]
        dcut, dcut_unqids = d[iid], d_unqids[iid]

        ignore_keys = (
            [
                "TARGETID",
                "STD",
                "SKY",
                "FILENAME",
            ]
            + bands
            + ["CASE"]
        )
        if img in ["odin", "suprime"]:
            ignore_keys += ["BRICKNAME", "OBJID"]
        if img in ["clauds"]:
            ignore_keys += ["ID"]

        for key in d.colnames:

            if key not in ignore_keys:

                assert np.all(
                    np.unique(dcut[key]) == np.zeros_like(dcut[key], shape=(1,))
                )

        log.info(
            "{}:\t{}/{} {} target rows".format(
                os.path.basename(targfn),
                iid.size,
                ((~d["SKY"]) & (~d["STD"])).sum(),
                img,
            )
        )

        # read input photometry (with some column names manipulation)
        p = read_targfn(os.path.join(photdir, os.path.basename(targfn)))

        if img in ["odin", "suprime"]:
            p_unqids = np.array(
                ["{}-{}".format(b, o) for b, o in zip(p["BRICKNAME"], p["OBJID"])]
            )
        if img in ["clauds"]:
            p_unqids = p["ID"].copy()

        _, ii = np.unique(p_unqids, return_index=True)

        if ii.size != len(p):

            log.info("{}:\tremove {} duplicates".format(targfn, len(p) - ii.size))
            p, p_unqids = p[ii], p_unqids[ii]

        # cut on targetids present in specinfo_table
        #   to speed up things
        # note: as cosmos_yr1, N673 has some duplicates
        sel = np.in1d(p_unqids, dcut_unqids)
        p, p_unqids = p[sel], p_unqids[sel]
        assert np.all(np.in1d(dcut_unqids, p_unqids))

        # now p is rather small so we can just loop
        #   to row-match it to dcut
        ii = [np.where(p_unqids == dcut_unqid)[0][0] for dcut_unqid in dcut_unqids]
        p, p_unqids = p[ii], p_unqids[ii]
        assert np.all(p_unqids == dcut_unqids)
        if img in ["odin", "suprime"]:
            assert np.all(p["BRICKNAME"] == dcut["BRICKNAME"])
            assert np.all(p["OBJID"] == dcut["OBJID"])
        if img in ["clauds"]:
            assert np.all(p["ID"] == dcut["ID"])

        if img in ["odin", "suprime"]:
            # easy columns..
            for key in [
                "RELEASE",
                "BRICKID",
                "BRICK_PRIMARY",
                "MASKBITS",
                "FITBITS",
                "TYPE",
                "RA",
                "DEC",
                "EBV",
            ]:

                dcut[key] = p[key]

            for band in bands:

                for key in [
                    "FLUX_{}".format(band),
                    "FLUX_IVAR_{}".format(band),
                    "FIBERFLUX_{}".format(band),
                    "PSFDEPTH_{}".format(band),
                    "GALDEPTH_{}".format(band),
                ]:

                    if key in p.colnames:

                        dcut[key] = p[key]

                    else:

                        log.warning(
                            "{} not present in {}".format(key, os.path.basename(targfn))
                        )

            # dr or hsc?
            dcut["BB_IMG"] = get_bb_img(targfn)

            # now bb photometry
            for band in ["G", "R", "R2", "I", "I2", "Z"]:

                for key in [
                    "FLUX_{}".format(band),
                    "FLUX_IVAR_{}".format(band),
                    "PSFDEPTH_{}".format(band),
                    "GALDEPTH_{}".format(band),
                ]:

                    if key in p.colnames:

                        dcut[key] = p[key]

                    else:

                        log.warning(
                            "{} not present in {}".format(key, os.path.basename(targfn))
                        )

        if img in ["clauds"]:

            dcut["BB_IMG"] = get_bb_img(targfn)

            for key in p.colnames:

                dcut[key] = p[key]

        # update d
        d[iid] = dcut

    search_radius = 1.0

    if img in ["odin", "suprime"]:

        sel = np.zeros(len(d), dtype=bool)

        for band in bands:

            sel |= d[band]

    if img in ["clauds"]:

        sel = np.ones(len(d), dtype=bool)

    iibands = np.where(sel)[0]

    # add zphot cosmos2020
    fn = get_cosmos2020_fn(case)
    log.info("cosmos2020_fn = {}".format(fn))
    d["COSMOS2020"] = np.zeros(len(d), dtype=bool)
    d["COSMOS2020_ID"] = np.zeros(len(d), dtype=">i8")
    d["COSMOS2020_ZPHOT"] = np.zeros(len(d))

    if fn is not None:

        z = fitsio.read(fn, columns=["ID", "ALPHA_J2000", "DELTA_J2000", "lp_zBEST"])
        iid, iiz, _, _, _ = match_coord(
            d["RA"][iibands],
            d["DEC"][iibands],
            z["ALPHA_J2000"],
            z["DELTA_J2000"],
            search_radius=search_radius,
        )
        d["COSMOS2020"] = np.zeros(len(d), dtype=bool)
        d["COSMOS2020_ID"] = np.zeros_like(z["ID"], shape=(len(d),))
        d["COSMOS2020_ZPHOT"] = np.zeros(len(d))
        d["COSMOS2020"][iibands[iid]] = True
        d["COSMOS2020_ID"][iibands[iid]] = z["ID"][iiz]
        d["COSMOS2020_ZPHOT"][iibands[iid]] = z["lp_zBEST"][iiz]

    # add zphot clauds
    # remark: as the "official" and offset catalogs are row-matched
    #           there is no need to distinguish for the zphots
    fn = get_clauds_fn(case)
    log.info("clauds_fn = {}".format(fn))
    d["CLAUDS"] = np.zeros(len(d), dtype=bool)
    d["CLAUDS_ID"] = np.zeros(len(d), dtype=">i8")
    d["CLAUDS_ZPHOT"] = np.zeros(len(d))

    if fn is not None:

        z = Table.read(fn)  # cannot be read with fitsio...
        iid, iiz, _, _, _ = match_coord(
            d["RA"][iibands],
            d["DEC"][iibands],
            z["RA"],
            z["DEC"],
            search_radius=search_radius,
        )
        d["CLAUDS"] = np.zeros(len(d), dtype=bool)
        d["CLAUDS_ID"] = np.zeros_like(z["ID"], shape=(len(d),))
        d["CLAUDS_ZPHOT"] = np.zeros(len(d))
        d["CLAUDS"][iibands[iid]] = True
        d["CLAUDS_ID"][iibands[iid]] = z["ID"][iiz]
        d["CLAUDS_ZPHOT"][iibands[iid]] = z["ZPHOT"][iiz]

    # clauds cosmos_yr1: at least DESILBG_TMG_FINAL and DESILBG_BXU_FINAL
    #   have objects in common, but those have different TARGETIDs
    #   such duplicates are excluded with the approach above
    #   we here handle them
    if (img == "clauds") & (case == "cosmos_yr1"):

        claudsids = d["ID"]
        unq_claudsids, counts = np.unique(claudsids, return_counts=True)
        unq_claudsids = unq_claudsids[counts > 1]  # cut on repeats
        unq_claudsids = unq_claudsids[
            np.in1d(unq_claudsids, claudsids[iid])
        ]  # cut on matched rows for this band
        log.info(
            "{}\t{}\thandle {} CLAUDS ID which have repeats".format(
                img, case, unq_claudsids.size
            )
        )
        fill_counts = 0
        fill_keys = [
            "COSMOS2020",
            "COSMOS2020_ID",
            "COSMOS2020_ZPHOT",
            "CLAUDS",
            "CLAUDS_ID",
            "CLAUDS_ZPHOT",
        ]
        log.info(
            "{}\t{}\twill propagate values for: {}".format(
                img, case, ", ".join(fill_keys)
            )
        )

        for unqclaudsid in unq_claudsids:

            ii = np.where(claudsids == unqclaudsid)[0]
            assert np.all(np.in1d(ii, iibands))
            tmpdict = {key: d[key][ii] for key in fill_keys}
            assert ii.size > 1
            tmp_fill_keys = [
                key for key in fill_keys if np.unique(tmpdict[key]).size > 1
            ]
            log.info(
                "CLAUDS ID={}\t: fill {}".format(unqclaudsid, ", ".join(tmp_fill_keys))
            )
            tmpvals = {
                key: [
                    _ for _ in tmpdict[key] if _ != np.zeros_like(d[key], shape=(1,))[0]
                ][0]
                for key in tmp_fill_keys
            }
            # overwrite the already recorded (matched) value, but ok, simpler code-wise
            for key in tmp_fill_keys:
                d[key][ii] = tmpvals[key]
            fill_counts += ii.size - 1  # because one value was already filled

        log.info(
            "{}\t{}\tfill an additional {}/{} rows with handling duplicates".format(
                img, case, fill_counts, iibands.size
            )
        )

    return d


def get_spec_table(img, case, stack_s, mydict):
    """
    Get the "enhanced" FIBERMAP table

    Args:
        img: element from allowed_imgs (str)
        case: round of DESI observation (str)
        stack_s: Spectra() object, resulting from create_coadd_merge() and spectra_stack() (Spectra object)
        mydict: dictionary with targeting info, output from get_img_infos() (dictionary)

    Returns:
        d: Table array

    Notes:
        In addition to the FIBERMAP infos, we propagate:
        - info about the targeting (see get_img_infos())
        - EXPIDS: list of exposures used for each spectra
        - TSNR2_{LYA,LRG,ELG,QSO} and EFFTIME_SPEC
        - STD, SKY: booleans indicating if the row is a standard star or a sky fiber
        - Visual Inspection (VI) information
    """
    assert img in allowed_imgs
    assert case in allowed_img_cases[img]

    ## fibermap
    log.info("")
    log.info("start from FIBERMAP")
    log.info("")
    d = Table(stack_s.fibermap)

    # add case
    d["CASE"] = case

    # remove dr9-like columns, which are dummy for laes
    rmvcols = get_fm_tractor_cols2rmv()
    log.info("remove from FIBERMAP: {}".format(rmvcols))
    d.remove_columns(rmvcols)

    ## fix some columns (due to buggy desispec code
    ##  when it was run)
    log.info("")
    log.info("fix some columns (due to buggy desispec code when it was run)")
    log.info("")
    d = fix_fibermap(d, Table(stack_s.exp_fibermap))

    ## add csv-list of expids
    log.info("")
    log.info("add csv-list of expids")
    log.info("")
    csv_expids = get_csv_expids(d, Table(stack_s.exp_fibermap))
    d["EXPIDS"] = csv_expids

    ## add tsn2_lrg + efftime_spec
    log.info("")
    log.info("add tsn2_lrg + efftime_spec")
    log.info("")
    sc = Table(stack_s.scores)
    assert np.all(sc["TARGETID"] == d["TARGETID"])

    # for key in ["TSNR2_LYA", "TSNR2_LRG", "TSNR2_ELG", "TSNR2_QSO"]:
    for key in ["TSNR2_LRG"]:

        d[key] = sc[key]

    ens = get_ensemble()["lrg"]
    d["EFFTIME_SPEC"] = ens.meta["SNR2TIME"] * d["TSNR2_LRG"]

    ## add img filter used for targeting + few infos
    log.info("")
    log.info("add img={} filter used for targeting + few infos".format(img))
    log.info("")
    d = add_img_infos(img, d, mydict)

    ## add std/sky info
    log.info("")
    log.info("add std/sky info")
    log.info("")
    d["STD"] = (d["DESI_TARGET"] & desi_mask["STD_FAINT"]) > 0
    d["SKY"] = d["OBJTYPE"] == "SKY"

    # add vi
    d["VI"] = np.zeros(len(d), dtype=bool)
    d["VI_Z"] = np.nan + np.zeros(len(d), dtype=">f4")
    d["VI_QUALITY"] = np.nan + np.zeros(len(d), dtype=">f4")
    d["VI_SPECTYPE"] = np.zeros(len(d), dtype="<U10")
    d["VI_COMMENTS"] = np.zeros(len(d), dtype="<U450")
    only_stdsky = ((d["STD"]) | (d["SKY"])).sum() == len(d)

    if only_stdsky:

        log.info("only STD or SKY fibers => no VI")

    else:

        bands = get_img_bands(img)
        sel = np.zeros(len(d), dtype=bool)

        for band in bands:

            sel |= d[band]

        iibands = np.where(sel)[0]
        #
        fns = get_vi_fns(img)
        log.info("vi_fns = {}".format(", ".join(fns)))

        if fns is not None:

            for fn in fns:

                vi = read_vi_fn(fn)

                # we beforehand removed possible duplicates in the vi catalog
                # TODO
                # well, actually maybe not true for suprime/cosmos_yr3...
                for iiband in iibands:

                    iivi = np.where(vi["TARGETID"] == d["TARGETID"][iiband])[0]

                    if iivi.size == 1:

                        d["VI"][iiband] = True
                        for key in ["VI_Z", "VI_QUALITY", "VI_SPECTYPE", "VI_COMMENTS"]:
                            d[key][iiband] = vi[key][iivi[0]]

                    if iivi.size > 1:

                        msg = "TARGETID={} appears {} times in {}".format(
                            d["TARGETID"][iiband], ii.size, os.path.basename(fn)
                        )
                        log.error(msg)
                        raise ValueError(msg)

                log.info("{} has {} rows".format(os.path.basename(fn), len(vi)))
                log.info(
                    "VI added for {}/{} targets".format(
                        (np.isfinite(d["VI_QUALITY"])).sum(), iibands.size
                    )
                )

    return d


def merge_cases(img, stack_ss, spec_ds, phot_ds, phot_v2_ds, exps_ds):
    """
    Stacks results from the different cases

    Args:
        img: element from allowed_imgs (str)
        stack_ss: list of NCASE Spectra() object, resulting from create_coadd_merge() and spectra_stack() (list of Spectra objects)
        spec_ds: list of NCASE specinfo tables from get_spec_table() (list of arrays)
        phot_ds: list of NCASE photinfo tables from get_phot_table()
            is relevant for LAE/LBG; if dealing with --stdsky, then set to None (list of arrays or None)
        phot_v2_ds: same as phot_ds, but for phot. with offsets;
            is relevant for clauds only; otherwise just set it to None (list of arrays or None)
        exp_ds: list of NCASE exps tables from get_expids() (list of arrays)

    Returns:
        stack_s: stack of stack_ss
        spec_d: stack of spec_ds
        phot_d: stack of phot_ds
        phot_v2_d: stack of phot_offset_ds for clauds; None otherwise
        exps_d: stack exps_ds
    """
    # stack_s
    cases = list(stack_ss.keys())
    stack_s = stack_ss[cases[0]]

    for case in cases[1:]:

        stack_s.update(stack_ss[case])

    # spec_ds
    assert np.all(cases == list(spec_ds.keys()))
    spec_d = vstack([spec_ds[case] for case in cases])

    # phot_ds
    if phot_ds is None:

        phot_d = None

    else:

        assert np.all(cases == list(phot_ds.keys()))
        phot_d = vstack([phot_ds[case] for case in cases])

    # phot_offset_ds
    if phot_v2_ds is None:

        phot_v2_d = None

    else:

        assert np.all(cases == list(phot_v2_ds.keys()))

        if np.sum([phot_v2_ds[case] is None for case in cases]) == len(phot_v2_ds):

            phot_v2_d = None

        else:

            phot_v2_d = vstack([phot_v2_ds[case] for case in cases])

    # expids_ds
    assert np.all(cases == list(exps_ds.keys()))
    exps_d = vstack([exps_ds[case] for case in cases])

    # TODO: handle possible duplicates?

    return stack_s, spec_d, phot_d, phot_v2_d, exps_d


def build_hs(
    img,
    cases,
    stack_s,
    spec_d,
    phot_d,
    phot_v2_d,
    exps_d,
):
    """
    Build the HDU list to then create a multi-extension fits file

    Args:
        img: element from allowed_imgs (str)
        cases: list of round of DESI observations (list of str)
        stack_s: Spectra() object, resulting from create_coadd_merge() and spectra_stack() (list of Spectra objects)
        spec_d: specinfo table from get_spec_table() (list of arrays)
        phot_d: photinfo table from get_phot_table()
            is relevant for LAE/LBG; if dealing with --stdsky, then set to None (list of arrays or None)
        phot_v2_d: as phot_d, but for phot. with offset; relevant only for clauds; otherwise
            is relevant for clauds only; otherwise just set it to None (list of arrays or None)
        exp_d: exps table from get_expids() (list of arrays)

    Returns:
        hs: HDU list

    Notes:
        cases should be expected to be allowed_img_cases[img]..
    """
    assert img in allowed_imgs
    for case in cases:
        assert case in allowed_img_cases[img]

    hs = fits.HDUList()

    # header
    h = fits.PrimaryHDU()
    # get the date (from the machine) and dump that
    date_str = (
        subprocess.check_output("date +%Y-%m-%dT%H:%M:%S%:z", shell=True)
        .decode("utf-8")
        .split("\n")[0]
    )
    yyyymmdd = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S%z").strftime("%Y%m%d")
    version = "v{}".format(yyyymmdd)
    h.header["CREADATE"] = date_str
    h.header["VERSION"] = version
    log.info("Storing in the PRIMARY header:")
    for key in ["CREADATE", "VERSION"]:
        log.info("\t{} = {}".format(key, h.header[key]))
    hs.append(h)

    # images
    for ext, bunit in zip(
        ["wave", "flux", "ivar", "mask"],
        [
            "Angstrom",
            "10**-17 erg/(s cm2 Angstrom)",
            "10**+34 (s2 cm4 Angstrom2) / erg2",
            None,
        ],
    ):

        h = fits.ImageHDU(name="BRZ_{}".format(ext.upper()))

        if bunit is not None:

            h.header["BUNIT"] = bunit

        d = eval("stack_s.{}['brz']".format(ext))
        h.data = d
        hs.append(h)

    # specinfo, photinfo, photoffinfo, expids
    ds, extnames = [spec_d], ["SPECINFO"]

    if phot_d is not None:

        ds.append(phot_d)
        extnames.append("PHOTINFO")

    if phot_v2_d is not None:

        ds.append(phot_v2_d)
        extnames.append("PHOTV2INFO")

    ds.append(exps_d)
    extnames.append("EXPIDS")

    for d, extname in zip(ds, extnames):

        h = fits.convenience.table_to_hdu(d)
        h.header["EXTNAME"] = extname

        # SPECINFO: cases, specdirs, vi fns
        if extname == "SPECINFO":

            h.header.append(
                fits.Card(
                    "LONGSTRN",
                    "OGIP 1.0",
                    "The OGIP Long String Convention may be used.",
                )
            )
            h.header["CASES"] = ",".join(cases)
            h.header["SPECDIRS"] = ",".join(
                np.hstack([get_specdirs(img, case) for case in cases])
            )
            fns = get_vi_fns(img)
            h.header["VIFNS"] = ",".join(fns)

        # PHOTINFO, PHOTV2INFO: cases, ext. coeffs (a bit hacky...), zphot fns
        if extname in ["PHOTINFO", "PHOTV2INFO"]:

            exts = get_ext_coeffs(img)
            h.header.append(
                fits.Card(
                    "LONGSTRN",
                    "OGIP 1.0",
                    "The OGIP Long String Convention may be used.",
                )
            )
            # zphot fn: use str() to protect against None
            h.header["ZC20FNS"] = ",".join(
                [str(get_cosmos2020_fn(case)) for case in cases]
            )
            h.header["ZCLAFNS"] = ",".join([str(get_clauds_fn(case)) for case in cases])
            h.header["EXTCOEFF"] = repr(exts).replace("'", '"')

        hs.append(h)

    return hs


def print_infos(img, fn):
    """
    Print few infos

    Args:
        img: element from allowed_imgs (str)
        fn: full path the merged file (str)

    Returns:
        Nothing
    """
    assert img in allowed_imgs

    bands = get_img_bands(img)

    d = Table.read(fn, "SPECINFO")
    std = (d["DESI_TARGET"] & desi_mask["STD_FAINT"]) > 0
    sky = d["OBJTYPE"] == "SKY"

    log.info("")
    log.info("# FN ALL STD SKY {}".format(" ".join(bands)))
    txt = "{}\t{}\t{}\t{}".format(
        os.path.basename(fn),
        len(d),
        std.sum(),
        sky.sum(),
    )

    for band in bands:

        txt += "\t{}".format(d[band].sum())

    log.info(txt)
