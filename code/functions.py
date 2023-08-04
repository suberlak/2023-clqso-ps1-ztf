# common qso analysis functions

# Imports
import os
from astropy.table import Table
from astropy import units as u
from astropy.io import fits
from astropy.convolution import convolve, Box1DKernel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import binned_statistic
from scipy import integrate

rcParams["ytick.labelsize"] = 15
rcParams["xtick.labelsize"] = 15
rcParams["axes.labelsize"] = 20
rcParams["axes.linewidth"] = 2
rcParams["font.size"] = 15
rcParams["axes.titlesize"] = 18


def flux2absigma(flux, fluxsigma):
    """Compute AB mag sigma given flux and flux sigma

    Here units of flux,  fluxsigma  don't matter
    as long as they are consistent, since we are dividing
    one by the other, so all the units cancel out.
    """
    FIVE_OVER_2LOG10 = 1.085736204758129569
    return FIVE_OVER_2LOG10 * fluxsigma / flux


def flux2ab(flux, unit="Jy"):
    """Compute AB mag given flux.

    Accept two unit types :
    *  'cgs', meaning flux is in  ergs / s / Hz / cm2
    *  'Jy', meaning flux is in Jy.  1 Jy = 1E-23 * ergs/s/Hz/cm2
    """
    if unit == "Jy":
        return -2.5 * np.log10(flux) + 8.90
    elif unit == "cgs":
        return -2.5 * np.log10(flux) - 48.6


class spectrum1d(object):
    def __init__(self, wave, flux, error=None):
        # Could be np.asarray, but change fit_spectral_lines.py
        self.wave = np.float_(wave)
        self.flux = np.float_(flux)
        self.error = np.float_(error)


def scale_spectra(spec1, spec2, wlmin=None, wlmax=None, scale_factor=False):
    """
    Scale spec1 to spec2 by integrating from wlmin to wlmax and using
    the area as a scaling factor
    Inputs:
        spec1, spec2: two spectrum1d objects with attributes wave and flux
        wlmin: shortest wavelength to be used in scaling
        wlmax: longest wavelength to be used in scaling
    Outputs:
        spec1: spec1d object with scaled flux
        scale_factor (optional): if scale_factor = True, then the multiplicative factor is returned
    """
    if wlmin is None:
        wlmin = max(spec1.wave.min(), spec2.wave.min())
    if wlmax is None:
        wlmax = min(spec1.wave.max(), spec2.wave.max())
    windx_1 = (spec1.wave >= wlmin) & (spec1.wave <= wlmax)
    windx_2 = (spec2.wave >= wlmin) & (spec2.wave <= wlmax)
    area1 = integrate.trapezoid(spec1.flux[windx_1], x=spec1.wave[windx_1])
    area2 = integrate.trapezoid(spec2.flux[windx_2], x=spec2.wave[windx_2])
    if spec1.error is not None:
        spec1 = spectrum1d(
            spec1.wave, spec1.flux / area1 * area2, spec1.error / area1 * area2
        )
    else:
        spec1 = spectrum1d(spec1.wave, spec1.flux / area1 * area2)
    if scale_factor is True:
        return spec1, area2 / area1
    else:
        return spec1


def combine_spectra(spectrum1, spectrum2):
    # Reverse sort wavelengths into ascending order
    wl_spectrum1 = spectrum1.spectral_axis.value[::-1]
    wl_spectrum2 = spectrum2.spectral_axis.value[::-1]

    # investigate dispersion
    dwl_spectrum1 = np.abs(np.median(np.diff(wl_spectrum1)))
    dwl_spectrum2 = np.abs(np.median(np.diff(wl_spectrum2)))
    print(dwl_spectrum1, dwl_spectrum2)

    wl_out = np.arange(
        min(wl_spectrum1.min(), wl_spectrum2.min()),
        max(wl_spectrum1.max(), wl_spectrum2.max()),
        dwl_spectrum2,
    )

    # interpolate to full wavelength range
    sort_indx_spectrum1 = np.argsort(spectrum1.spectral_axis.value)
    flux_spectrum1 = np.interp(
        wl_out,
        spectrum1.spectral_axis.value[sort_indx_spectrum1],
        spectrum1.flux.value[sort_indx_spectrum1],
        left=np.nan,
        right=np.nan,
    )
    err_spectrum1 = np.interp(
        wl_out,
        spectrum1.spectral_axis.value[sort_indx_spectrum1],
        spectrum1.uncertainty.array[sort_indx_spectrum1],
        left=np.nan,
        right=np.nan,
    )
    sort_indx_spectrum2 = np.argsort(spectrum2.spectral_axis.value)
    flux_spectrum2 = np.interp(
        wl_out,
        spectrum2.spectral_axis.value[sort_indx_spectrum2],
        spectrum2.flux.value[sort_indx_spectrum2],
        left=np.nan,
        right=np.nan,
    )
    err_spectrum2 = np.interp(
        wl_out,
        spectrum2.spectral_axis.value[sort_indx_spectrum2],
        spectrum2.uncertainty.array[sort_indx_spectrum2],
        left=np.nan,
        right=np.nan,
    )

    flux_combine = np.nanmean([flux_spectrum1, flux_spectrum2], axis=0)
    error_combine = 0.5 * np.nansum([err_spectrum1, err_spectrum2], axis=0)
    error_combine[np.isnan(err_spectrum2)] = err_spectrum1[np.isnan(err_spectrum2)]
    error_combine[np.isnan(err_spectrum1)] = err_spectrum2[np.isnan(err_spectrum1)]

    plt.figure()
    plt.plot(spectrum1.spectral_axis.value, spectrum1.flux.value)
    plt.plot(spectrum2.spectral_axis.value, spectrum2.flux.value)
    plt.plot(wl_out, flux_combine)
    plt.plot(wl_out, convolve(flux_combine, Box1DKernel(9)))

    tbdata = Table(
        [wl_out, flux_combine, error_combine], names=["wavelength", "flux", "error"]
    )

    return tbdata


def write_final_spectrum(header, tbdata, fname):
    # Overlap region is pretty small, probably most accurate to not change the exposure time
    # for kwd in ['exptime', 'xposure', 'telapse', 'ttime']:
    #    if kwd in hdr1 and kwd in hdr2:
    #        hdr[kwd] += hdr2[kwd]
    tbdata.meta = header
    tbdata.write("{name}_combine.fits".format(name=fname), overwrite=True)
    tbdata.write(
        "{name}_combine.ascii".format(name=fname),
        format="ascii.commented_header",
        overwrite=True,
    )
    tbdata.write(
        "{name}_combine.dat".format(name=fname),
        format="ascii.no_header",
        overwrite=True,
    )

    array_snex = np.vstack([tbdata["flux"], tbdata["error"]])
    dispersion = tbdata["wavelength"][1] - tbdata["wavelength"][0]
    header["CTYPE1"] = "LINEAR"
    header["CRVAL1"] = tbdata["wavelength"][0]
    header["CRPIX1"] = 0
    header["CD1_1"] = dispersion

    hdu = fits.PrimaryHDU(array_snex, header=header)
    hdu.writeto("{name}_snex.fits".format(name=fname), overwrite=True)


def median_n_exposures(spectra, Nbins):
    """
    A convenience function to take N exposures
    (assumed to be the result of the same wavelength
    calibration, so that the spectral axis for each is the same )

    Parameters:
    -----------
    spectra: a dictionary with keys correponding to the exposure number,
          each of which has "flux" and "wavelength" in identical units,
          eg.  spectra[0]["flux"],  spectra[0]["wavelength"],
          spectra[1]["flux"], spectra[1]["wavelength"].
    Nbins : number of bins into which divide the wavelegnth axis
    """

    xs = {}
    ys = {}
    edges = {}
    binn = {}

    for i in range(len(spectra)):
        x = spectra[i]["wavelength"]
        y = spectra[i]["flux"]
        stat_med, bin_edges, binnumber = binned_statistic(
            x, y, statistic="median", bins=Nbins
        )
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        xs[i] = x
        ys[i] = y
        edges[i] = bin_edges
        binn[i] = binnumber
        # ax.scatter(x[binnumber==1], y[binnumber==1])
    # Now take all points from both exposures that fall
    # in a given bin, and calculate the median ..

    y_med = []
    y_std = []
    all_xs = []
    all_ys = []

    # Below will only work for the case of at least two exposures to combine
    # NB if we had more than 1 spectrum it wouldn't work, b/c below we take
    # ys[0] and ys[1]
    # that's future implementation ,
    if len(spectra) == 1:
        print("Returning only the median per bin as there is only one exposure")
        stat_std, bin_edges, binnumber = binned_statistic(
            x, y, statistic="std", bins=Nbins
        )
        y_med = stat_med
        y_std = stat_std
        all_xs = xs
        all_ys = ys

    elif len(spectra) == 2:
        # iterate over wavelength bins
        for b in range(1, Nbins + 1):
            # combine fluxes from both exposures that fall in the wavelength bin
            ys_in_bin = np.append(ys[0][binnumber == b], ys[1][binnumber == b])
            xs_in_bin = np.append(xs[0][binnumber == b], xs[1][binnumber == b])
            # print(len(xs_in_bin))
            all_xs = np.append(all_xs, xs_in_bin)
            all_ys = np.append(all_ys, ys_in_bin)
            # print(len(all_xs))
            # print(len(all_ys))
            y_med.append(np.median(ys_in_bin))
            y_std.append(np.std(ys_in_bin))

    elif len(spectra) > 2:
        print("Not yet implemented for more than 2 spectra")
        return {}

    result = {
        "flux_all": all_ys,
        "wavelength_all": all_xs,
        "flux_med_per_bin": y_med,
        "flux_std_per_bin": y_std,
        "bins": bin_centers,
    }

    return result


def apply_wavelength_solution(
    obj_spectrum,
    xpts=None,
    wpts=None,
    sky_spectrum=None,
    fit_wavelength_args={"mode": "interp", "deg": 1},
):
    """This is needed to apply a wavelength solution (xpts, wpts) mapping,
    i.e. what wavelength  each x-pixel position along the spectral axis
    corresponds to. These points are used to fit a simple eg. linear
    function that maps x-position to wavelength.

    obj_spectrum: specutils.spectra.spectrum1d.Spectrum1D,
        a 1D spectrum (i.e. after extraction of the spectrum
        along  a trace from an original FITS image, which
        is a 2D spectrum).
        If the  obj_spectrum corresponds to a sky observation,
        it will contain sky lines contamination (eg. science object,
        standard star). In that case, sky_spectrum should also be
        provided to create a sky-subtracted spectrum.
        However, uf obj_spectrum corresponds to a calibration,
        eg. external or internal calibration arc lamps,
        then there is no need to subtract sky background emission
        (since there isn't any).
    xpts:  numpy.ndarray  of x-positions along spectral axis
    wpts:  numpy.ndarray of wavelength (in Angstroms) along spectral axis
    sky_spectrum: specutils.spectra.spectrum1d.Spectrum1D (optional)
       a 1D spectrum (as above), usually from running
       obj_spectrum, sky_spectrum = kosmos.BoxcarExtract(obj_2D_spectrum,
                                       obj_trace, **kwargs)
    mode: str, mode used by kosmos.fit_wavelength() ('interp' or 'gp')
    deg : int,  degree of polynomial (default is 1) options
         passed to   kosmos.fit_wavelength()

    """
    # xpts = xpxl_int_arc_red
    # wpts = wpts_int_arc_red
    # obj_spectrum = red_neon_1d
    # sky_spectrum = red_neon_sky

    # Create sky subtracted spectrum
    if sky_spectrum is not None:
        obj_flux = obj_spectrum.flux - sky_spectrum.flux
    else:
        obj_flux = obj_spectrum.flux
    sky_sub_obj_spectrum = specutils.Spectrum1D(
        flux=obj_flux,
        spectral_axis=obj_spectrum.spectral_axis,
        uncertainty=obj_spectrum.uncertainty,
    )

    # Apply the wavelength solution to the sky subtracted spectrum
    obj_wave = kosmos.fit_wavelength(
        sky_sub_obj_spectrum, xpts, wpts, **fit_wavelength_args
    )
    # mode='interp', deg=deg)
    return obj_wave


def get_ZTF_DR_lc(name, ra, dec, band, dr_path, keep_all_columns=False):
    """
    Author: Paula Sánchez Sáez, PhD

    Function to dowload an individual ZTF light curve using the ZTF API.
    For more details about ZTF service, go to their documentation:
    https://irsa.ipac.caltech.edu/data/ZTF/docs/releases/dr08/ztf_release_notes_dr08.pdf
    See section "iii. Querying Lightcurves using the API."

    Parameters:
    -----------
    name: str
        Object name (e.g. the object "SDSS J005132.94+180120.5" should be
        refered here only by "J005132.94+180120.5")
    ra: float
        Right ascension of the object (in degrees).
    dec: float
        Declination of the object (in degrees).
    band: {'g', 'r', 'i'}
        ZTF photometric band.
    dr_path: str
        Path to the download directory.
    keep_all_columns: bool, default=False
        If False, only the most important light curve parameters will be retrieved.
        Otherwise, keep all the columns.

    Returns:
    --------
    df: pd.DataFrame
        DataFrame containing the measurments of the retrieved light curve.
    """

    # make up a convenient filename
    fname = f"{name}_ZTF_{band}.csv"

    # don't download if the file already exists
    if os.path.exists(fname):
        print(f"File {fname} already exists")

    else:
        file_path = os.path.join(dr_path, fname)
        irsa_path = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
        url = f"{irsa_path}?POS=CIRCLE {ra} {dec} 0.000277778&BANDNAME={band}&FORMAT=csv&NOBS_MIN=3"
        cmd = f"wget -O {file_path} " + f'"{url}"'
        print(cmd)
        os.system(cmd)


def flux2absigma(flux, fluxsigma):
    """Compute AB mag sigma given flux and flux sigma

    Here units of flux,  fluxsigma  don't matter
    as long as they are consistent, since we are dividing
    one by the other, so all the units cancel out.
    """
    FIVE_OVER_2LOG10 = 1.085736204758129569
    return FIVE_OVER_2LOG10 * fluxsigma / flux


def flux2ab(flux, unit="Jy"):
    """Compute AB mag given flux.

    Accept two unit types :
    *  'cgs', meaning flux is in  ergs / s / Hz / cm2
    *  'Jy', meaning flux is in Jy.  1 Jy = 1E-23 * ergs/s/Hz/cm2
    """
    if unit == "Jy":
        return -2.5 * np.log10(flux) + 8.90
    elif unit == "cgs":
        return -2.5 * np.log10(flux) - 48.6


def get_ztf_file_list(dr_path, n_min=2):
    """
    Get a list of files in the directory,
    and only return those that are longer
    than the n_min argument.

    Parameters:
    -----------
    dr_path: str, full path to the directory
        to test
    n_min: int (optional), minimum lenght of
        file to accept

    Returns:
    --------
    file_array: list of accepted file names
    """
    # platform-agnostic solution
    # get the file list
    filelist = os.listdir(dr_path)
    filelength = []
    for file in filelist:
        filepath = os.path.join(dr_path, file)
        # open the file and count the lines
        with open(filepath, "r") as f:
            flength = len(f.readlines())
            # append that length to the file length array
            filelength.append(flength)
            # print(filepath,flength)

    # select only those that have more than  4 points...
    file_array = np.array(filelist)
    file_length_array = np.array(filelength)
    mask = file_length_array > n_min

    # make a new array selecting only files with more than N lines
    file_array_long = file_array[mask]
    return file_array_long


def count_ztf_r_band(dr_path):
    # get a list of all ZTF light curves
    ztf_all_bands = get_ztf_file_list(dr_path)

    # select only r-band objects
    ztf_r_band = [file for file in ztf_all_bands if file.__contains__("_r.")]
    print(f"There are {len(ztf_r_band)} light curves with ZTF r-band data")

    dr_path = os.path.join(os.getcwd(), "ZTFDR17")


# N=0


def get_data_to_dict_NEW(
    ztf_dr_path, ztf_n=-99, sdss_name_pattern="J204303", ztf_all_bands=[]
):
    """Quick function to get a downloaded ZTF  S82 light curve,
    and get the SDSS and PS1 data for it.

    It is the same as get_data_to_dict() below, except
    that it uses an extended list of S82 CLQSO (114 cans),
    which includes Celerite data, as well as Shen2011-2008,
    and DR9 DBQSO, which lends the SDSS lcname.

    So instead of loading DBQSO  to get dbID based on SDSSJID,
    like 000246.47-000615.3  ,
    it simply reads it directly from the clqso table ...

    ztf_dr_path : path to ZTF data directory
    N: select n-th CLQSO that has ZTF data (useful if we just want to plot all)

    """
    path_to_repo = "/Users/chris/GradResearch/Paper2_SDSS_PTF_PS1/code3/"
    file_name = "CLQSO_S82_Shen2011_Celerite_114_cans.txt"
    path_to_file = os.path.join(path_to_repo, file_name)
    clqso = Table.read(path_to_file, format="ascii")
    sdss_jid_col = "SDSS_NAME"  # column containing eg. 000246.47-000615.3
    sdss_dbId_col = "dbID"
    # def get_data_to_dict_2023(dr_path, N, clqso):
    # only a few light curves have ZTF r-band data...

    # make dictionary to store data
    data_r_band = {}

    # get a list of all ZTF light curves
    # if it has not been provided
    if len(ztf_all_bands) == 0:
        ztf_all_bands = get_ztf_file_list(ztf_dr_path)

    # select only r-band objects
    ztf_r_band = [file for file in ztf_all_bands if file.__contains__("_r.")]

    # print(ztf_r_band)
    # print(ztf_n)
    if ztf_n > 0:
        # Select the N-th object
        print(f"Selecting {ztf_n}-th ZTF light curve")
        ztf_fname = ztf_r_band[ztf_n]

    else:  # Select by matching to SDSS name pattern
        matched_name = [name for name in ztf_r_band if sdss_name_pattern in name]
        if len(matched_name) < 1:
            print(
                f"Could not find a ZTF light curve in {ztf_dr_path} that\ contains{sdss_name_pattern}"
            )
        else:
            ztf_fname = matched_name[0]
            print(f"Selected ZTF light curve that starts with {sdss_name_pattern}:")
            print(f"{ztf_fname}")

    # store the name of the file for the ZTF
    data_r_band["ZTF_filename"] = ztf_fname

    # read the ZTF data
    ztf_lc = Table.read(os.path.join(ztf_dr_path, ztf_fname), format="csv")

    # select SDSS data  - the files from SDSS are named
    # after dbID, need clean dbID
    sdss_jid = ztf_fname.split("_")[0][1:]

    # store that JID
    data_r_band["SDSS_JID"] = sdss_jid

    # I don't even need here dbId as the extended CLQSO list already has
    # the Shen2011 information, which has quasar median
    # g-i ...
    mask = clqso[sdss_jid_col] == sdss_jid
    clqso_row = clqso[mask]
    print(clqso_row)
    sdss_dbid = clqso_row[sdss_dbId_col][0]

    # store that identifier
    data_r_band["SDSS_dbID"] = sdss_dbid

    # translate ZTF r to synthetic SDSS r
    # use g,i already in clqso table

    gi = clqso_row["g"][0] - clqso_row["i"][0]

    # store the original ZTF r-band data just in case,
    # before they get overwritten
    data_r_band["ztf_original"] = ztf_lc[["mjd", "mag", "magerr"]]
    # r_SDSS_synth = r_ZTF  + 0.01  + 0.04  * gi
    # just overwrite the old value of magnitude
    # with the new value

    ztf_lc["mag"] = ztf_lc["mag"] + 0.01 + 0.04 * gi
    data_r_band["ztf_synthetic"] = ztf_lc[["mjd", "mag", "magerr"]]

    # read in the SDSS light curve
    # sdss_dbid
    sdss_lc = Table.read(os.path.join("QSO_S82/", str(sdss_dbid)), format="ascii")

    # given the file structure explained in
    # https://faculty.washington.edu/ivezic/macleod/qso_dr7/Southern_format_LC.html
    # rename the relevant columns
    # and do not keep the filter name
    # in the column name,
    # i.e. mag rather than mag_r ,
    # given that for all surveys we are only
    # selecting (for now) data in r-band
    old_names = ["col7", "col8", "col9"]  # r-band mjd, mag, magerr are cols 7,8,9
    new_names = ["mjd", "mag", "magerr"]
    sdss_lc.rename_columns(old_names, new_names)
    sdss_lc_select = sdss_lc[new_names]

    # -99 means missing data - ignore those
    mask = sdss_lc_select["mag"] > 0
    # store in a dictionary
    data_r_band["sdss"] = sdss_lc_select[mask]

    # add PS1 data , like in Fig_14_22-25_clqso.ipynb
    botDircleaning = "/Users/chris/GradResearch/Paper2_SDSS_PTF_PS1/dp2/real_sdss_ps1r_dr2_cleaning_NEW/"
    # has all the intermediate data products :
    lcname = clqso_row["lcname"][0]
    # mjd, mag,  magerr, Nobs,  avgmag,  medmag,  avgerr
    lc = Table.read(botDircleaning + lcname, format="ascii")

    # here mag, magerr, mjd has original points,
    # and then after day-averaging, we have stored the
    # mjdint, and according to that we have the same
    # medmag, avgmag, avgerr for each placve where

    mask_survey = lc["survey"] == "ps1"
    data_r_band["ps1"] = lc[mask_survey]

    return data_r_band


def get_data_to_dict(dr_path, N=None, sdss_jid=None):
    """
    N: select n-th CLQSO that has ZTF data (useful if we just want to plot all)
    sdss_jid : SDSS jid object name, if we want to plot a curve
        for an object with a specific name ...
    """
    # only a few light curves have ZTF r-band data...
    # make dictionary to store data
    data_r_band = {}

    # get a list of all ZTF light curves
    ztf_all_bands = get_ztf_file_list(dr_path)

    # select only r-band objects
    # list of names like 'J012114.19-010310.8_ZTF_r.csv'
    ztf_r_band = [file for file in ztf_all_bands if file.__contains__("_r.")]

    if len(ztf_r_band) > 0:
        # read the ZTF data
        # select N-th light curve
        ztf_fname = ""
        if sdss_jid is not None:
            # check if that jid is in the ztf files
            for name in ztf_r_band:
                if sdss_jid in name:
                    ztf_fname = name
                    print(f"Based on given {sdss_jid}, found data for {ztf_fname}")

        elif N is not None:
            ztf_fname = ztf_r_band[N]
            print(f"Asking for N={N} ZTF light curve - using {ztf_fname}")
        else:
            print("must either provide N or sdss_jid")

        # only read if there is a valid ZTF name ...
        if len(ztf_fname) > 1:
            ztf_lc = Table.read(os.path.join(dr_path, ztf_fname), format="csv")

            # store the name of the file for the ZTF
            data_r_band["ZTF_filename"] = ztf_fname  # ztf_r_band[N]

    # select SDSS data  - the files from SDSS are named
    # after dbID, need clean dbID
    if (sdss_jid is None) and (N is not None):
        sdss_jid = ztf_r_band[N].split("_")[0][1:]
    else:
        print(f"Using provided {sdss_jid}")

    # store that JID
    data_r_band["SDSS_JID"] = sdss_jid

    # list the clqso candidates
    colnames = [
        "dbID",
        "SDSSJID",
        "ra",
        "dec",
        "Redshift",
        "log10_Lbol",
        "log10_MBH",
        "f_Edd",
        "Delta(mag)",
        "Delta(sigma_G)",
        "MedianPS1",
    ]
    clqso = Table.read("CLQSO_candidates.txt", format="ascii", names=colnames)

    mask = clqso["SDSSJID"] == sdss_jid
    clqso_row = clqso[mask]
    sdss_dbid = clqso_row["dbID"][0]
    sdss_dbid_clean = sdss_dbid.split("^")[0]  # remove the "^b" part from dbID

    # store that identifier
    data_r_band["SDSS_dbID"] = sdss_dbid_clean

    # translate ZTF r to synthetic SDSS r
    path = os.getcwd()
    file_path = os.path.join(path, "DB_QSO_S82.dat.gz")

    # we know the column meaning from
    # https://faculty.washington.edu/ivezic/macleod/qso_dr7/Southern_format_DB.html
    colnames = [
        "dbID",
        "ra",
        "dec",
        "SDR5ID",
        "M_i",
        "M_i_corr",
        "redshift",
        "mass_BH",
        "Lbol",
        "u",
        "g",
        "r",
        "i",
        "z",
        "Au",
    ]
    SDSS_DB_QSO = Table.read(file_path, format="ascii", names=colnames)

    mask = SDSS_DB_QSO["dbID"] == int(sdss_dbid_clean)
    gi = SDSS_DB_QSO[mask]["g"][0] - SDSS_DB_QSO[mask]["i"][0]

    if len(ztf_fname) > 0:
        # store the original ZTF r-band data just in case,
        # before they get overwritten
        data_r_band["ztf_original"] = ztf_lc[["mjd", "mag", "magerr"]]
        # r_SDSS_synth = r_ZTF  + 0.01  + 0.04  * gi
        # just overwrite the old value of magnitude
        # with the new value

        ztf_lc["mag"] = ztf_lc["mag"] + 0.01 + 0.04 * gi
        data_r_band["ztf_synthetic"] = ztf_lc[["mjd", "mag", "magerr"]]

    # read in the SDSS light curve
    sdss_lc = Table.read(os.path.join("QSO_S82/", sdss_dbid_clean), format="ascii")

    # given the file structure explained in
    # https://faculty.washington.edu/ivezic/macleod/qso_dr7/Southern_format_LC.html
    # rename the relevant columns
    # and do not keep the filter name
    # in the column name,
    # i.e. mag rather than mag_r ,
    # given that for all surveys we are only
    # selecting (for now) data in r-band
    old_names = ["col7", "col8", "col9"]  # r-band mjd, mag, magerr are cols 7,8,9
    new_names = ["mjd", "mag", "magerr"]
    sdss_lc.rename_columns(old_names, new_names)
    sdss_lc_select = sdss_lc[new_names]

    # -99 means missing data - ignore those
    mask = sdss_lc_select["mag"] > 0
    # store in a dictionary
    data_r_band["sdss"] = sdss_lc_select[mask]

    # add PS1 data
    # list all PS1 light curves
    ps1 = Table.read("CLQSO_candidates_PS1_DR2.csv", format="csv")

    # select only  rows for that object
    mask_object = ps1["_ra_"] == clqso_row["ra"][0]

    # plot the light curve in r-band
    mask_filter = ps1["filterID"] == 2

    # combine the masks
    mask_ps1 = mask_object * mask_filter
    ps1_select = ps1[mask_ps1][
        "_ra_", "_dec_", "obsTime", "nr", "psfFlux", "psfFluxErr"
    ]

    # calculate AB magnitudes
    ps1_select["mag"] = flux2ab(ps1_select["psfFlux"])
    ps1_select["magerr"] = flux2absigma(ps1_select["psfFlux"], ps1_select["psfFluxErr"])

    # rename obsTime to mjd
    ps1_select.rename_column("obsTime", "mjd")

    # add to the dictionary
    data_r_band["ps1"] = ps1_select

    for key in data_r_band.keys():
        print(f"Got {key}")
    # return the dic and the  object name
    return data_r_band


def plot_combined_data(
    data,
    surveys=["sdss", "ps1", "ztf_synthetic"],
    labels=["SDSS", "PS1", "ZTF r-synth"],
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    for survey, label in zip(surveys, labels):
        lc = data[survey]
        ax.errorbar(
            lc["mjd"],
            lc["mag"],
            yerr=lc["magerr"],
            fmt="o",
            markersize=2,
            alpha=0.6,
            capsize=3,
            label=label,
        )

    # ax.set_ylim(14,22)
    ax.set_xlabel("Time (MJD)", fontsize=18, labelpad=12)
    ax.set_ylabel("Magnitude", fontsize=18, labelpad=12)
    ax.tick_params(direction="in", pad=5, labelsize=13)
    ax.legend(
        fontsize=16,
        markerscale=3,
    )
    name = data["SDSS_JID"]
    ax.set_title(f"Light curve {name} r-band")

    # invert y-axis because smaller value of magnitude means brighter object
    ax.invert_yaxis()


def average_data(data, surveys_to_average=["ps1", "ztf_synthetic"], Nsigma=5):
    """A quick function to day-average the data.
    Add to the original data table columns with
    number of points per day, error-weighted average magnitude,
    average mjd, weighter error, and boolean flag columns indicating
    that a given point has anomalously large (>5sigmaG) departure
    in magnitude space or in the photometric uncertainty space
    (using the unaveraged quantities). Methodology of adding an
    uncertainty floor of 0.02 mag follows Suberlak+2017
    https://faculty.washington.edu/ivezic/Publications/Suberlak2017.pdf

    Parameters:
    -----------
    data : a dictionary containing light-curve data for
    each survey, with each data[survey] including
    mjd, mag, magerr
    surveys_to_average : names of surveys to average,
    assumed to be the keys of data dictionary
    Nsigma: float, multiplier of sigmaG that sets the
        threshold for large departure in magnitude or
        error distribution

    """
    for survey in surveys_to_average:
        # select all data points for that light curve
        mjds = data[survey]["mjd"].data
        mag = data[survey]["mag"].data
        err = data[survey]["magerr"].data

        # find unique days
        mjd_int = [int(mjd) for mjd in mjds]

        # count unique days
        mjd_unique = np.unique(mjd_int)
        n_unique = len(mjd_unique)
        n_total = len(mjd_int)
        print(f"\n{survey}")
        print(f"Unique days:{n_unique} total number of observations: {n_total}")

        # prepare storage arrays
        n_obs_day = np.zeros_like(mjds)
        avgmjd_ = np.zeros_like(mjds)
        avgmag_ = np.zeros_like(mjds)
        avgerr_ = np.zeros_like(mjds)
        large_error = np.zeros_like(mjds, dtype=bool)
        large_departure = np.zeros_like(mjds, dtype=bool)

        # flag points that depart more
        # than 5 sigmaG from mag distribution...
        # Nsig = 5
        x = mag
        sigmaGmag = 0.7413 * (np.percentile(x, 75) - np.percentile(x, 25))
        msig = np.abs(np.ma.median(x) - x) > Nsigma * sigmaGmag
        large_departure = msig

        # flag points that have error more than
        # 5*sigmaG from the median of  error distribution...
        x = err
        sigmaGerr = 0.7413 * (np.percentile(x, 75) - np.percentile(x, 25))
        merr = np.abs(np.ma.median(x) - x) > Nsigma * sigmaGerr
        large_error = merr

        # perform day-averaging
        # iterate over all days
        for i in range(n_unique):
            # select points from a given day
            mjd_day = mjd_unique[i]
            mask = mjd_int == mjd_day

            # count points and store
            n_points = np.sum(mask)
            n_obs_day[mask] = n_points

            # select points and plot
            mjd_points = np.array(mjds)[mask]
            mag_points = mag[mask]

            # average the selected points...
            # but only if there is more than one point per day
            if n_points > 1:
                err_points = err[mask]
                w = 1 / (err_points * err_points)
                avgmag = np.average(mag_points, weights=w)
                avgerr = 1.0 / np.sqrt(np.sum(w))
                avgmjd = np.mean(mjd_points)
                # increase error if too small
                if avgerr < 0.02:
                    avgerr = np.sqrt(avgerr**2.0 + 0.01**2.0)

                # store the averaged mjd, mag, magerr
                avgmag_[mask] = avgmag
                avgmjd_[mask] = avgmjd
                avgerr_[mask] = avgerr

        # store that information
        data[survey]["Nday"] = n_obs_day
        data[survey]["avgmjd"] = avgmjd_
        data[survey]["avgmag"] = avgmag_
        data[survey]["avgerr"] = avgerr_
        data[survey]["large_error"] = large_error
        data[survey]["large_departure"] = large_departure

        # store the value of N sigma that was used to flag the points
        data["Nsigma"] = Nsigma
    # return the updated dictionary
    return data


def plot_averaged_data(
    data,
    surveys=["sdss", "ps1", "ztf_synthetic"],
    averaged=["ps1", "ztf_synthetic"],
    labels={"sdss": "SDSS", "ps1": "PS1", "ztf_synthetic": "ZTF r-synth"},
    colors={"sdss": "#1f77b4", "ps1": "#2ca02c", "ztf_synthetic": "#9467bd"},
    plot_flagged=True,
    suffix="",
    mjds=[],
    ylims=None,
):
    """A function to plot the day-averaged data

    Parameters:
    ----------
    data: a dictionary with keys corresponding to the surveys to plot
    surveys: a list of surveys to plot
    labels: dict of labels to use for each survey
    averaged: which surveys were day-averaged
    colors: dict of colors to use for plotting
    plot_flagged: boolean - whether  or not to plot flagged points
    """

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=150, facecolor="w")

    # plot SDSS, which was not day-averaged
    survey = "sdss"
    lc = data[survey]
    points = ax.errorbar(
        lc["mjd"],
        lc["mag"],
        yerr=lc["magerr"],
        fmt="o",
        markersize=2,
        alpha=0.6,
        capsize=3,
        label=labels[survey],
        color=colors[survey],
    )

    for survey in averaged:
        # only plot original if there are no averaged for PS1, ZTF

        # select which points had only one observation per day
        mask = data[survey]["Nday"] == 1

        # do not plot flagged points if not needed
        if not plot_flagged:
            m1 = data[survey]["large_error"]  # True if large error
            m2 = data[survey]["large_departure"]  # True if large departure
            m12 = m1 | m2  # logical OR - True if either large error OR large departure
            mask_flagged = ~m12  # True only for not flagged points
            mask = mask & mask_flagged  # logical AND

        lc = data[survey]
        points = ax.errorbar(
            lc["mjd"][mask],
            lc["mag"][mask],
            yerr=lc["magerr"][mask],
            fmt="o",
            markersize=2,
            alpha=0.6,
            capsize=3,
            label=labels[survey],
            color=colors[survey],
        )

        print(f"{survey} single point color is {points[0].get_color()}")

        # select which points  had more than one point per day
        mask = data[survey]["Nday"] > 1
        # do not plot flagged points if not needed
        if not plot_flagged:
            m1 = data[survey]["large_error"]  # True if large error
            m2 = data[survey]["large_departure"]  # True if large departure
            m12 = m1 | m2  # logical OR - True if either large error OR large departure
            mask_flagged = ~m12  # True only for not flagged points
            mask = mask & mask_flagged  # logical AND

        points = ax.errorbar(
            data[survey]["avgmjd"][mask],
            data[survey]["avgmag"][mask],
            data[survey]["avgerr"][mask],
            fmt=".",
            markersize=10,
            mfc="white",
            mew=2,
            label="",
            color=colors[survey],
        )
        # print(f'{survey} averaged color is {points[0].get_color()}')

        # circle points that have 5sigma departure
        if plot_flagged:
            # read the Nsigma parameter used for flagging
            Nsig = data["Nsigma"]
            for flag, color, label in zip(
                ["large_error", "large_departure"],
                ["magenta", "orange"],
                [str(Nsig) + r"$\sigma$ err", str(Nsig) + r"$\sigma$ mag"],
            ):
                mask = data[survey][flag]
                ax.scatter(
                    data[survey]["mjd"][mask],
                    data[survey]["mag"][mask],
                    s=80,
                    facecolors="none",
                    edgecolors=colors[survey],
                    label="",
                )
    if len(mjds) > 0:
        for j in range(len(mjds)):
            ax.axvline(mjds[j], ls="--", lw=2)

    # add title, ticks, labels, etc.
    ax.set_xlabel("Time (MJD)", fontsize=18, labelpad=12)
    ax.set_ylabel("Magnitude", fontsize=18, labelpad=12)
    ax.tick_params(direction="in", pad=5, labelsize=13)
    ax.legend(
        fontsize=16,
        markerscale=3,
    )
    name = data["SDSS_JID"]
    ax.set_title(f"Light curve {name} r-band")

    # invert y-axis because smaller value of magnitude means brighter object
    ax.invert_yaxis()
    if ylims is not None:
        ax.set_ylim(ylims)
    name = data["SDSS_JID"][:5]

    if plot_flagged:
        suffix = "with_flagged"
    fname = f"sdss_ztf_ps1_{name}_combined_{suffix}.png"
    plt.savefig(
        fname,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )
    print(f"Saved as {fname} in {os.getcwd()}")
