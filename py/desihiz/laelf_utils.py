#!/usr/bin/env python

import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import gridspec
from desiutil.log import get_logger


log = get_logger()

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


allowed_lfsrcs = ["anand_variable"]


def get_filt_lminmax(survey):
    """
    Returns the filter "edges" in Angstroms.

    Args:
        survey: "odin", "suprime", "desi2", "specs5" (str)

    Returns:
        filts: dictionary with one key per filter, and
            for each filter the lmin and lmax in A (dict)
    """

    assert survey in ["odin", "suprime", "desi2", "specs5"]

    # https://noirlab.edu/science/programs/ctio/filters/Dark-Energy-Camera
    if survey == "odin":

        filts = {
            "N419": {"lmin": 4195.0 - 75.0 / 2.0, "lmax": 4195.0 + 75.0 / 2.0},
            "N501": {"lmin": 5010.0 - 75.0 / 2.0, "lmax": 5010.0 + 75.0 / 2.0},
            "N673": {"lmin": 6730.0 - 100.0 / 2.0, "lmax": 6730.0 + 100.0 / 2.0},
        }

    # https://data.desi.lbl.gov/desi/survey/fiberassign/special/tertiary/0026/inputcats/COSMOS_LAE_Candidates_2023apr04v2_tertiary26.README.txt
    if survey == "suprime":

        filts = {
            "I427": {"lmin": 4263.0 - 196.0 / 2.0, "lmax": 4263.0 + 196.0 / 2.0},
            "I464": {"lmin": 4635.0 - 214.0 / 2.0, "lmax": 4635.0 + 214.0 / 2.0},
            "I484": {"lmin": 4849.0 - 222.0 / 2.0, "lmax": 4849.0 + 222.0 / 2.0},
            "I505": {"lmin": 5063.0 - 226.0 / 2.0, "lmax": 5063.0 + 226.0 / 2.0},
            "I527": {"lmin": 5261.0 - 246.0 / 2.0, "lmax": 5261.0 + 246.0 / 2.0},
        }

    # https://www.overleaf.com/project/6394b196852f0e1affab67e9
    if survey in ["desi2", "specs5"]:

        filts = {
            "MB0": {"lmin": 3750.0, "lmax": 4006.0},
            "MB1": {"lmin": 4006.0, "lmax": 4265.0},
            "MB2": {"lmin": 4265.0, "lmax": 4519.0},
            "MB3": {"lmin": 4519.0, "lmax": 4779.0},
            "MB4": {"lmin": 4779.0, "lmax": 5036.0},
            "MB5": {"lmin": 5036.0, "lmax": 5297.0},
            "N540": {"lmin": 5297.0, "lmax": 5507.0},
        }

    return filts


# Lstar in erg / s, phistar in Mpc ** -3
def get_schechter_params(lfsrc, z=None, outpng=None):
    """
    Retrieve the Schechter parameters for a LAE luminosity function.

    Args:
        lfsrc: "anand_variable" (str)
        z (optional, defaults to None): redshift at which to estimate the LF (float)
        outpng (optional, default to None): if set, plotting the Schechter parmeters (str)

    Returns
        phistar: Phi_star in Mpc ** -3 (float)
        Lstar: L_star in erg / s (float)
        alpha: alpha slope (float)

    Notes:
        If lfsrc == "anand_variable", then z has to be set
        "anand_variable": simple interpolation of the parameters, from:
            - z=2.2 (Konno+16)
            - z=3.1, 3.7, 5.7 (Ouchi+08)
    """
    assert lfsrc in allowed_lfsrcs

    # variable - simple interpolation,
    #   - z=2.2: Konno+16
    #   - z=3.1, 3.7, 5.7: Ouchi+08
    #   values from Table 5 of Konno+16
    if lfsrc == "anand_variable":

        assert z is not None

        zs = np.array([2.2, 3.1, 3.7, 5.7])
        srcs = np.array(["Konno+16", "Ouchi+08", "Ouchi+08", "Ouchi+08"])
        phistars = np.array([6.32e-4, 3.90e-4, 3.31e-4, 4.44e-4])
        Lstars = np.array([5.29e42, 8.49e42, 9.16e42, 9.09e42])
        alphas = np.array([-1.75, -1.8, -1.8, -1.8])

        # simple interpolation
        if z < zs[0]:

            phistar, Lstar, alpha = phistars[0], Lstars[0], alphas[0]
            log.warning(
                "z = {} is lower than the lowest z={} LF value; taking values for z={}".format(
                    z, zs[0], zs[0]
                )
            )

        elif z >= zs[-1]:

            phistar, Lstar, alpha = phistars[-1], Lstars[-1], alphas[-1]
            log.warning(
                "z = {} is higher than the highest z={} LF value; taking values for z={}".format(
                    z, zs[-1], zs[-1]
                )
            )

        else:

            i = np.where(zs <= z)[0][-1]  # first index just below z
            assert i != len(zs) - 1
            phistar = phistars[i] + (z - zs[i]) / (zs[i + 1] - zs[i]) * (
                phistars[i + 1] - phistars[i]
            )
            Lstar = Lstars[i] + (z - zs[i]) / (zs[i + 1] - zs[i]) * (
                Lstars[i + 1] - Lstars[i]
            )
            alpha = alphas[i] + (z - zs[i]) / (zs[i + 1] - zs[i]) * (
                alphas[i + 1] - alphas[i]
            )

        # plot?
        if outpng is not None:

            fig = plt.figure(figsize=(15, 5))
            gs = gridspec.GridSpec(1, 3, wspace=0.3)

            for ip, (ys, ylab, ylim) in enumerate(
                zip(
                    [1e4 * phistars, Lstars * 1e-42, alphas],
                    [
                        r"$\Phi \star$ [1e-4 Mpc$^{-3}$]",
                        r"$L \star$ [1e42 erg / s]",
                        r"$\alpha$",
                    ],
                    [(3, 7), (5, 10), (-2.0, -1.5)],
                )
            ):

                ax = fig.add_subplot(gs[ip])
                ax.plot(zs, ys, "-o")
                ax.set_xlabel("Redshift")
                ax.set_ylabel(ylab)
                ax.set_xlim(2, 6)
                ax.set_ylim(ylim)
                ax.grid()

                if ip == 0:

                    x, y, dy = 0.5, 0.9, -0.05

                    for z, src in zip(zs, srcs):

                        ax.text(
                            x, y, "z={} : {}".format(z, src), transform=ax.transAxes
                        )
                        y += dy
            plt.savefig(outpng, bbox_inches="tight")
            plt.close()
            print("")
            print("# Z PHI_STAR L_STAR ALPHA SRC")

            for z, phistar, Lstar, alpha, src in zip(
                zs, phistars, Lstars, alphas, srcs
            ):

                print(
                    "{:.2f}\t{:.2e}\t{:.2e}\t{:.2f}\t{}".format(
                        z,
                        phistar,
                        Lstar,
                        alpha,
                        src,
                    )
                )
            print("")

    return phistar, Lstar, alpha


# LAE Schechter LF parameters
# Lstar in erg / s, phistar in Mpc ** -3
def get_nlae_per_mpc3(Llim, lfsrc, z=None):
    """
    Compute the LAE density (per Mpc3) for a given Luminosity Function,
        brighter than Llim (in erg / s), with integrating the Schechter function.

    Args:
        Llim: luminosity threshold in erg / s (float)
        lfsrc: "anand_variable" (str)
        z (optional, defaults to None): redshift at which to estimate the LF (float)
        outpng (optional, default to None): if set, plotting the Schechter parmeters (str)

    Returns
        nlae_per_mpc3: LAE density per Mpc3 (float)

    Notes:
        The maximum range for integration is 0.0001 * Lstar to 10 * Lstar.
        If lfsrc == "anand_variable", then z has to be set
        "anand_variable": simple interpolation of the parameters, from:
            - z=2.2 (Konno+16)
            - z=3.1, 3.7, 5.7 (Ouchi+08)
    """

    assert lfsrc in allowed_lfsrcs

    phistar, Lstar, alpha = get_schechter_params(lfsrc, z=z)

    # L/L* grid to integrate the LF
    ls = np.linspace(0.0001, 10, 100000)
    llim = Llim / Lstar
    sel = ls > llim

    nlae_per_mpc3 = phistar * np.trapz(ls[sel] ** alpha * np.exp(-ls[sel]), x=ls[sel])

    return nlae_per_mpc3


def get_skyarea_sqdeg():
    """
    Compute the full sky area in deg2 (4 * pi * (180/pi) ** 2).

    Args:
        None

    Returns:
        skyarea_sqdeg (float)
    """

    return 4 * np.pi * (180 / np.pi) ** 2


# comoving volume in Mpc3, observed in fov_sqdeg deg2
def get_obs_covols(fov_sqdeg, zmins, zmaxs):
    """
    Compute the comoving volume subtended by a given field-of-view.

    Args:
        fov_sqdeg: field-of-view in deg2 (float)
        zmins: lower-redshift boundary of the shell(s) (float or array of floats)
        zmaxs: upper-redshift boundary of the shell(s) (float or array of floats)

    Returns:
        obs_covols: comoving volume in (zmins, zmaxs) subtended by fov_sqdeg;
            units: Mpc3 / deg2 (float or array of floats)
    """

    covols = cosmo.comoving_volume(zmaxs).value - cosmo.comoving_volume(zmins).value
    obs_covols = covols * (fov_sqdeg / get_skyarea_sqdeg())

    return obs_covols


def get_nlae(lfluxlim, lfsrc, zmin, zmax, dz=0.01):
    """
    Compute the density of LAE brighter than a flux limit,
        in (zmin, zmax) for a given Luminosity Function.

    Args:
        lfluxlim: flux limit in erg / s / cm2 / A (float)
        lfsrc: "anand_variable" (str)
        zmin: lower-redshift boundary of the shell (float)
        zmax: upper-redshift boundary of the shell (float)
        dz (optional, defaults to 0.01): we slice the (zmin, zmax) shell
            in slices of dz with (float)

    Returns
        n_per_mpc3: LAE density per Mpc3 (float)
        n_per_deg2: LAE density per deg2 (float)

    Notes:
        If lfsrc == "anand_variable", then z has to be set
        "anand_variable": simple interpolation of the parameters, from:
            - z=2.2 (Konno+16)
            - z=3.1, 3.7, 5.7 (Ouchi+08)
    """
    # define the area of the frame
    fov_sqdeg = 1.0  # field of view in sq. deg
    #
    zs = [zmin]

    while zs[-1] + dz < zmax:

        zs.append(np.round(zs[-1] + dz, 4))

    zs.append(zmax)
    zs = np.array(zs)

    # comoving volume in Mpc3, observed in 1 deg2
    obs_covols = get_obs_covols(fov_sqdeg, zs[:-1], zs[1:])

    #
    n_per_mpc3, n_per_deg2 = 0, 0

    for i in range(len(zs) - 1):

        zcen = 0.5 * (zs[i] + zs[i + 1])
        dl = cosmo.luminosity_distance(zcen).to("cm").value
        Llim = lfluxlim * 4.0 * np.pi * dl**2
        # nb of LAEs per Mpc3
        n_per_mpc3_i = get_nlae_per_mpc3(Llim, lfsrc, z=zcen)
        # nb of LAEs per deg2
        n_per_deg2_i = n_per_mpc3_i * obs_covols[i]
        #
        n_per_mpc3 += n_per_mpc3_i
        n_per_deg2 += n_per_deg2_i

    return n_per_mpc3, n_per_deg2


def get_nlaes(lfluxlims, lfsrc, zmin, zmax, dz=0.01):
    """
    Compute the density of LAE brighter than a flux limit,
        in (zmin, zmax) for a given Luminosity Function.

    Args:
        lfluxlims: array of flux limits in erg / s / cm2 / A (array of floats)
        lfsrc: "anand_variable" (str)
        zmin: lower-redshift boundary of the shell (float)
        zmax: upper-redshift boundary of the shell (float)
        dz (optional, defaults to 0.01): we slice the (zmin, zmax) shell
            in slices of dz with (float)

    Returns
        n_per_mpc3: LAE density per Mpc3 (array of floats)
        n_per_deg2: LAE density per deg2 (array of floats)

    Notes:
        If lfsrc == "anand_variable", then z has to be set
        "anand_variable": simple interpolation of the parameters, from:
            - z=2.2 (Konno+16)
            - z=3.1, 3.7, 5.7 (Ouchi+08)
    """
    #
    nflux = len(lfluxlims)
    n_per_mpc3, n_per_deg2 = np.zeros(nflux), np.zeros(nflux)

    for i in range(nflux):

        n_per_mpc3[i], n_per_deg2[i] = get_nlae(lfluxlims[i], lfsrc, zmin, zmax, dz=dz)

    return n_per_mpc3, n_per_deg2


# https://data.desi.lbl.gov/desi/survey/fiberassign/special/tertiary/0026/inputcats/COSMOS_LAE_Candidates_2023apr04v2_tertiary26.README.txt
def get_filtmag(flam, filt_cen, filt_wid):
    """
    Convert a flux limit into a magnitude limit for a filter.

    Args:
        flam: flux flimit in erg / s / cm2 / A (float)
        filt_cen: filter central wavelength in A (float)
        filt_wid: filter width in A (float)

    Returns:
        maglim: the corresponding magnitude limit in AB (float)

    Notes:
        Credit to A. Dey
    """

    filt_flam = flam * u.erg / u.cm**2 / u.s / (filt_wid * u.angstrom)

    filt_fnu = filt_flam.to(
        u.erg / u.cm**2 / u.s / u.Hz,
        equivalencies=u.spectral_density(filt_cen * u.angstrom),
    )

    mag = -48.60 - 2.5 * np.log10(filt_fnu.value)

    return mag


def get_filtflam(mag, filt_cen, filt_wid):
    """
    Convert a AB magnitude to a flux for a filter.

    Args:
        mag: AB magnitude (float)
        filt_cen: filter central wavelength in A (float)
        filt_wid: filter width in A (float)

    Returns:
        flam: flux flimit in erg / s / cm2 / A (float)

    Notes:
        Credit to A. Dey.
        This is the inverse of get_filtmag().
    """

    filt_fnu = 10. ** (-0.4 * (mag + 48.60))

    filt_fnu *= u.erg / u.cm**2 / u.s / u.Hz

    filt_lam = filt_fnu.to(
        u.erg / u.cm**2 / u.s / u.angstrom,
        equivalencies=u.spectral_density(filt_cen * u.angstrom),
    )

    flam = filt_lam.value * filt_wid

    return flam
