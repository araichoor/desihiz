#!/usr/bin/env python

import os
import tempfile
from scipy.ndimage import gaussian_filter1d
import numpy as np
from astropy.table import Table
from astropy import units
from speclite import filters as speclite_filters


def get_opt_waves():

    return np.arange(3000, 10001, dtype=float)


def get_filt_fns():

    mydir = os.path.join(os.getenv("DESI_ROOT"), "users", "raichoor", "laelbg")

    mydict = {}

    # clauds: cfht/u and uS
    mydict["MEGACAM_U"] = os.path.join(mydir, "clauds", "filt", "CFHT_MegaCam.u.dat")
    mydict["MEGACAM_US"] = os.path.join(mydir, "clauds", "filt", "CFHT_MegaCam.u_1.dat")

    # hsc
    for band in ["G", "R", "I", "Z"]:
        mydict["HSC_{}".format(band)] = os.path.join(
            mydir, "suprime", "filt", "Subaru_HSC.{}.dat".format(band.lower())
        )

    # suprime
    for band in ["I427", "I464", "I484", "I505", "I527"]:
        mydict["SUPRIME_{}".format(band)] = os.path.join(
            mydir,
            "suprime",
            "filt",
            "Subaru_Suprime.{}.dat".format(band.replace("I", "IB")),
        )

    # odin
    for band in ["N419", "N501", "N673"]:
        mydict["DECAM_{}".format(band)] = os.path.join(
            mydir,
            "odin",
            "filt",
            "{}_simulated_total_transmission_f3.6.txt".format(band),
        )

    return mydict


def get_filts(norm=True):

    mydict = get_filt_fns()

    filts = {}
    waves = get_opt_waves()

    for instband in mydict:
        fn = mydict[instband]
        d = Table.read(fn, format="ascii.commented_header")

        # odin passbands are in nm...
        if instband.split("_")[1] in ["N419", "N501", "N673"]:

            d["WAVE"] *= 10.

        filts[instband] = {}
        filts[instband]["WAVE"] = waves
        filts[instband]["RESP"] = np.interp(
            waves, d["WAVE"], d["TRANSMISSION"], left=0, right=0
        )
        if norm:
            filts[instband]["RESP"] /= np.nanmax(filts[instband]["RESP"])

    return filts


# ws : wavelengths in A
# ts : transmission
def get_speclite_filt(ws, ts):

    tmp_speclite_dir = tempfile.mkdtemp()

    tmp_filt = speclite_filters.FilterResponse(
        wavelength=ws * units.Angstrom,
        response=ts,
        meta=dict(group_name="grp", band_name="band"),
    )
    tmp_name = tmp_filt.save(tmp_speclite_dir)

    # now read them in
    speclite_filt = speclite_filters.load_filter("grp-band")

    return speclite_filt


# instands : list of {instrument_band}, e.g. "MEGACAM_U"
def get_speclite_filts(instbands):

    filts = get_filts()
    speclite_filts = {}

    for instband in instbands:

        ws, ts = filts[instband]["WAVE"], filts[instband]["RESP"]
        speclite_filts[instband] = get_speclite_filt(ws, ts)

    return speclite_filts


# handle nan s for smoothing..
# https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
def get_smooth(fs, ivs, gauss_smooth):
    tmp0fs = fs.copy()
    tmp0fs[ivs == 0] = 0
    tmp1fs = 1 + 0 * fs.copy()
    tmp1fs[ivs == 0] = 0
    tmp0smfs = gaussian_filter1d(tmp0fs, gauss_smooth, mode="constant", cval=0)
    tmp1smfs = gaussian_filter1d(tmp1fs, gauss_smooth, mode="constant", cval=0)
    return tmp0smfs / tmp1smfs
