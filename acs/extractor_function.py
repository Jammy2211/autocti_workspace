"""
This script illustrates extractions of CCD sections from ACS .fits imaging. It does not perform any unit conversions
or bias subtractions, it only shows how rotations are handled.
"""

from astropy.io import fits
import numpy as np

import autoarray as aa
import autoarray.plot as aplt
from autoarray import exc
from os import path


def extract_quadrant(file_path, quadrant_letter, bias_path=None):

    """
    Map quadrant to its letter.
    """
    if quadrant_letter == "D" or quadrant_letter == "C":
        hdu = 1
    elif quadrant_letter == "B" or quadrant_letter == "A":
        hdu = 4
    else:
        raise exc.ArrayException("Quadrant letter for FrameACS must be A, B, C or D.")

    """
    Load relevent fits info from image file.
    """
    hdulist = fits.open(file_path)

    sci_header = hdulist[0].header

    if sci_header["TELESCOP"] != "HST":
        raise exc.ArrayException(
            f"The file {file_path} does not point to a valid HST ACS dataset."
        )

    if sci_header["INSTRUME"] != "ACS":
        raise exc.ArrayException(
            f"The file {file_path} does not point to a valid HST ACS dataset."
        )

    exposure_time = sci_header["EXPTIME"]
    date_of_observation = sci_header["DATE-OBS"]
    time_of_observation = sci_header["TIME-OBS"]

    ext_header = hdulist[hdu].header

    units = ext_header["BUNIT"]
    bscale = ext_header["BSCALE"]
    bzero = ext_header["BZERO"]

    hdu_list = fits.open(file_path, do_not_scale_image_data=True)

    """
    Load image data and convert using bscale.
    """
    array = np.array(hdu_list[hdu].data)

    if units in "COUNTS":
        array_electrons = (array * bscale) + bzero
    elif units in "CPS":
        array_electrons = (array * exposure_time * bscale) + bzero
    else:
        array_electrons = array

    """
    Use GAIN to convert to electrons.
    """
    gain = sci_header["CCDGAIN"]

    if round(gain) == 1:
        calibrated_gain = [0.99989998, 0.97210002, 1.01070000, 1.01800000]
    elif round(gain) == 2:
        calibrated_gain = [0.99989998, 0.97210002, 1.01070000, 1.01800000]
    elif round(gain) == 4:
        calibrated_gain = [4.011, 3.902, 4.074, 3.996]
    else:
        raise exc.ArrayException(
            "Calibrated gain of ACS file does not round to 1, 2 or 4."
        )

    """
    NOT 100% SURE THAT A = 0, B = 1. C = 2, D = 3
    """
    if quadrant_letter == "A":
        calibrated_gain = calibrated_gain[0]
    elif quadrant_letter == "B":
        calibrated_gain = calibrated_gain[1]
    elif quadrant_letter == "C":
        calibrated_gain = calibrated_gain[2]
    elif quadrant_letter == "D":
        calibrated_gain = calibrated_gain[3]

    array_electrons = array_electrons * calibrated_gain

    """
    Flip up-down for A
    """
    if quadrant_letter == "B" or quadrant_letter == "A":
        array_electrons = np.flipud(array_electrons)

    """
    Repeat for Bias.
    """
    bias_hdu_list = fits.open(bias_path, do_not_scale_image_data=True)

    bias = np.array(bias_hdu_list[hdu].data)

    bias_ext_header = bias_hdu_list[hdu].header

    bias_units = bias_ext_header["BUNIT"]

    """
    There are no BSCALE / BZERO params in the bias frame, and when I use them I get a dodgy result. So don't allow for
    a bias frame that isn't in COUNTS for now.
    """
    if bias_units != "COUNTS":
        raise exc.ArrayException("Cannot use bias frame not in counts.")

    bias_electrons = bias * calibrated_gain

    """
    Flip up down for quadrant A or B
    """
    if quadrant_letter == "B" or quadrant_letter == "A":
        bias_electrons = np.flipud(bias_electrons)

    """
    Extract region for quadrant:
    """
    if quadrant_letter == "A" or quadrant_letter == "C":
        array_electrons = array_electrons[0:2068, 0:2072]
        bias_electrons = bias_electrons[0:2068, 0:2072]
    elif quadrant_letter == "B" or quadrant_letter == "D":
        array_electrons = array_electrons[0:2068, 2072:4144]
        bias_electrons = bias_electrons[0:2068, 2072:4144]
    else:
        raise exc.ArrayException("Quadrant letter for FrameACS must be A, B, C or D.")

    """
    Rotate to orient for CTI clicking in arctic (no rotation for quadrant A or C)
    """
    if quadrant_letter == "B" or quadrant_letter == "D":
        array_electrons = array_electrons[:, ::-1].copy()
        bias_electrons = bias_electrons[:, ::-1].copy()

    """
    Bias subtractions.
    """
    array_electrons -= prescan_fitted_bias_column(array_electrons[:, 18:24])
    array_electrons -= bias_electrons

    return aa.Array2D.manual_native(array=array_electrons, pixel_scales=0.05)


def prescan_fitted_bias_column(prescan, n_rows=2048, n_rows_ov=20):
    """
    Generate a bias column to be subtracted from the main image by doing a
    least squares fit to the serial prescan region.

    e.g. image -= prescan_fitted_bias_column(image[18:24])

    See Anton & Rorres (2013), S9.3, p460.

    Parameters
    ----------
    prescan : [[float]]
        The serial prescan part of the image. Should usually cover the full
        number of rows but may skip the first few columns of the prescan to
        avoid trails.

    n_rows : int
        The number of rows in the image, exculding overscan.

    n_rows_ov : int, int
        The number of overscan rows in the image.

    Returns
    -------
    bias_column : [float]
        The fitted bias to be subtracted from all image columns.
    """
    n_columns_fit = prescan.shape[1]

    # Flatten the multiple fitting columns to a long 1D array
    # y = [y_1_1, y_2_1, ..., y_nrow_1, y_1_2, y_2_2, ..., y_nrow_ncolfit]
    y = prescan[:-n_rows_ov].T.flatten()
    # x = [1, 2, ..., nrow, 1, ..., nrow, 1, ..., nrow, ...]
    x = np.tile(np.arange(n_rows), n_columns_fit)

    # M = [[1, 1, ..., 1], [x_1, x_2, ..., x_n]].T
    M = np.array([np.ones(n_rows * n_columns_fit), x]).T

    # Best-fit values for y = M v
    v = np.dot(np.linalg.inv(np.dot(M.T, M)), np.dot(M.T, y))

    # Map to full image size for easy subtraction
    bias_column = v[0] + v[1] * np.arange(n_rows + n_rows_ov)

    print("# fitted bias v =", v)
    # plt.figure()
    # pixels = np.arange(n_rows + n_rows_ov)
    # for i in range(n_columns_fit):
    #     plt.scatter(pixels, prescan[:, i])
    # plt.plot(pixels, bias_column)
    # plt.show()

    return np.transpose([bias_column])


"""
The dataset path to the ACS data.
"""
dataset_path = path.join("acs", "dataset")

"""
Load and plot the already complete ACS data reduction of this example image, for extraction.
"""
file_path = path.join(dataset_path, "j9epf6kjq_raw.fits")
bias_path = path.join(dataset_path, "q4a1532mj_bia.fits")

image = extract_quadrant(file_path=file_path, quadrant_letter="D", bias_path=bias_path)

mat_plot_2d = aplt.MatPlot2D(cmap=aplt.Cmap(vmin=0.0, vmax=10.0))
array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

print("\n\nVia Explicit function RESULTS:\n")
print(
    f"Corner Values = {image.native[0,0]}, {image.native[0, -1]}, {image.native[-1, 0]}, {image.native[-1, -1]}"
)
print(f"Min / Max Values = {np.min(image)}, {np.max(image)}")
print(f"Shape of Bias Extraction (via autoarray) = {image.shape_native}")
