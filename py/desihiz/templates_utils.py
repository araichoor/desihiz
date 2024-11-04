#!/usr/bin/env python

import os
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy import units
from desihiz.specphot_utils import get_opt_waves, get_speclite_filts
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator


def get_template_infos():

    mydir = os.path.join(
        os.getenv("DESI_ROOT"), "users", "raichoor", "laelbg", "templates"
    )
    mydict = {
        "LBG_ABS": {
            "FN": os.path.join(mydir, "lbgsmooth", "rrtemplate-lbgsmooth.ecsv"),
            "WAVE": "WAVELENGTH",
            "FLUX": "CY_LBG_ABS_FLUX",
        },
        "LBG_EM": {
            "FN": os.path.join(mydir, "lbgsmooth", "rrtemplate-lbgsmooth.ecsv"),
            "WAVE": "WAVELENGTH",
            "FLUX": "CY_LBG_LAE_2_FLUX",
        },
        "LAE": {
            "FN": os.path.join(mydir, "lae-orig", "rrtemplate-lae.ecsv"),
            "WAVE": "WAVELENGTH",
            "FLUX": "LAE_ODIN_FLUX",
        },
    }

    return mydict


def read_templates(names):

    mydict = get_template_infos()
    templates = {}

    for name in names:
        fn, wkey, fkey = mydict[name]["FN"], mydict[name]["WAVE"], mydict[name]["FLUX"]
        d = Table.read(fn)
        templates[name] = {
            "WAVE": d[wkey],
            "FLUX": d[fkey],
        }

    return templates


def plot_templates(outpng, templates):

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 1, hspace=0.1)

    for ip, (templname, templcol) in enumerate(zip(templnames, templcols)):

        ax = fig.add_subplot(gs[ip])
        ax.plot(
            templates[templname]["WAVE"],
            templates[templname]["FLUX"],
            color=templcol,
            lw=0.5,
            label=templname,
        )
        if ip == 2:
            ax.set_xlabel("Rest-frame wavelength [A]")
        else:
            ax.set_xticklabels([])
        ax.set_ylabel("Flux [A.U.]")
        ax.set_xlim(800, 2000)
        ax.set_ylim(1e-2, 1e0)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.grid()
        ax.legend(loc=2)

    plt.savefig(outpng, bbox_inches="tight")
    plt.close()


# template flux in: units.erg / (units.cm**2 * units.s * units.Angstrom)
def compute_mags(templates, zs, speclite_filts):

    names = list(templates.keys())

    instbands = list(speclite_filts.keys())
    assert np.unique(instbands).size == len(instbands)

    hs = fits.HDUList()
    h = fits.PrimaryHDU()
    hs.append(h)

    for name in names:

        d = Table()
        d["Z"] = zs

        for instband in instbands:
            rf_ws, fs = templates[name]["WAVE"], templates[name]["FLUX"]
            d["MAG_{}".format(instband)] = 0.0
            for i in range(len(zs)):
                ws = rf_ws * (1 + zs[i])
                d["MAG_{}".format(instband)][i] = speclite_filts[
                    instband
                ].get_ab_magnitude(
                    fs * units.erg / (units.cm**2 * units.s * units.Angstrom),
                    ws * units.Angstrom,
                )

        h = fits.convenience.table_to_hdu(d)
        h.header["EXTNAME"] = name
        hs.append(h)

    return hs

