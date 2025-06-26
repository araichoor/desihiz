#!/usr/bin/env python

import os
import tempfile
from scipy.ndimage import gaussian_filter1d, correlate1d
import numpy as np
from astropy.table import Table
from astropy import units
from speclite import filters as speclite_filters


def get_opt_waves():

    return np.arange(3500, 10001, dtype=float)


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

    # ibis
    for band in ["M411", "M438", "M464", "M490", "M517"]:
        mydict["DECAM_{}".format(band)] = os.path.join(
            mydir,
            "ibis",
            "filt",
            "ibis-20241112-{}.ecsv".format(band),
        )

    # merian
    for band in ["N540"]:
        mydict["DECAM_{}".format(band)] = os.path.join(
            mydir,
            "ibis",
            "filt",
            "ibis-20240608-{}.ecsv".format(band),
        )


    return mydict


def get_filts(norm=True):

    mydict = get_filt_fns()

    filts = {}
    waves = get_opt_waves()

    for instband in mydict:
        fn = mydict[instband]
        d = Table.read(fn, format="ascii.commented_header")
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


# instands : list of {instrument_band}, e.g. "CFHT_U"
# bands : corresponding list of band names for the output dictionary
def get_speclite_filts(instbands, bands):

    filts = get_filts()
    specilte_filts = {}
    for instband, band in zip(instbands, bands):
        ws, ts = filts["WAVE"], filts["RESP"]
        speclite_filts[band] = get_speclite_filt(ws, ts)

    return speclite_filts

# extracting part of ndimage code, to get the kernel used in the gaussian_filter1d() call
# so that we can propagate the smoothing to the ivar
# https://github.com/scipy/scipy/blob/705b490b094f96c6f63dbed5e9ef849dc1be47ce/scipy/ndimage/_filters.py#L225-L253
def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


# handle nan s for smoothing..
# https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
def get_smooth(fs, ivs, gauss_smooth):

    # default order, truncate, axis, cval, output in gaussian_filter1d
    order, truncate, axis, cval, output = 0, 4, -1, 0., None
    mode="constant"

    tmp0fs = fs.copy()
    tmp0fs[ivs == 0] = 0
    tmp1fs = 1 + 0 * fs.copy()
    tmp1fs[ivs == 0] = 0

    tmp0smfs = gaussian_filter1d(tmp0fs, gauss_smooth, mode=mode, cval=cval, order=order, truncate=truncate, axis=axis, output=output)
    tmp1smfs = gaussian_filter1d(tmp1fs, gauss_smooth, mode=mode, cval=cval, order=order, truncate=truncate, axis=axis, output=output)

    # we convolve the variance with the square of the kernel (https://iopscience.iop.org/article/10.3847/2515-5172/abe8df)
    lw = int(truncate * gauss_smooth + 0.5)
    weights = _gaussian_kernel1d(gauss_smooth, order, lw)[::-1]
    smivs = 1. / correlate1d(1. / ivs, weights ** 2, axis=axis, output=output, mode=mode, cval=cval, origin=0)

    return tmp0smfs / tmp1smfs, smivs
