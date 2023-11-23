# desihiz

This code generates "merged" catalogs for the DESI hiz pilot observations.
It gathers:
- the spectra (from custom healpix reductions in $DESI_ROOT/users/raichoor/laelbg)
- FIBERMAP information
- tractor photometry used for the target selection
- COSMOS2020_zphot, CLAUDS_zphot, if available
- Visual Inspection result, if available
- informations about the exposures used for the coadds

## Commands to run to generate the files
First define $YOUR_OUTPUT_DIR, your output folder.
Grab a logging node at NERSC.

### ODIN
desi_hiz_merge --outfn $YOUR_OUTPUT_DIR/desi-odin.fits --img odin --numproc 32

desi_hiz_merge --outfn $YOUR_OUTPUT_DIR/desi-odin-stdsky.fits --img odin --numproc 32

### SUPRIME
desi_hiz_merge --outfn $YOUR_OUTPUT_DIR/desi-suprime.fits --img suprime --numproc 32

desi_hiz_merge --outfn $YOUR_OUTPUT_DIR/desi-suprime-stdsky.fits --img suprime --numproc 32
