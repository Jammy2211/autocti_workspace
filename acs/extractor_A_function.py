"""
This script illustrates extractions of CCD sections from ACS .fits imaging. It does not perform any unit conversions
or bias subtractions, it only shows how rotations are handled.
"""

from astropy.io import fits
import numpy as np

import autoarray as aa
import autoarray.plot as aplt
from os import path


def extract_quadrant_a(file_path, bias_path=None):

    """
    Quadrant A is in HDU 4
    """
    hdu = 4

    """
    Load relevent fits info from image file.
    """
    hdulist = fits.open(file_path)

    sci_header = hdulist[0].header

    exposure_time = sci_header["EXPTIME"]
    date_of_observation = sci_header["DATE-OBS"]
    time_of_observation = sci_header["TIME-OBS"]

    ext_header = hdulist[hdu].header

    units = ext_header["BUNIT"]
    bscale = ext_header["BSCALE"]
    bzero = ext_header["BZERO"]

    hdu_list = fits.open(file_path, do_not_scale_image_data=True)

    """
    Load image data and convert to electrons.
    """
    array = np.array(hdu_list[hdu].data)

    print(array)


    if units in "COUNTS":
        array_electrons = (array * bscale) + bzero
    elif units in "CPS":
        array_electrons = (array * exposure_time * bscale) + bzero
    else:
        array_electrons = array

    """
    Flip up-down for A
    """
    array_electrons = np.flipud(array_electrons)

    """
    Repeat for Bias.
    """
    hdu_list = fits.open(bias_path, do_not_scale_image_data=True)

    bias = np.array(hdu_list[hdu].data)


    """
    Dont convert?
    """
    # if units in "COUNTS":
    #     bias_electrons = (bias * bscale) + bzero
    # elif units in "CPS":
    #     bias_electrons = (bias * exposure_time * bscale) + bzero
    # else:
    bias_electrons = bias

    """
    Flip up down for quadrant A
    """
    bias_electrons = np.flipud(bias_electrons)

    """
    Extract region for quadrant A:
    """
    parallel_size = 2068
    serial_size = 2072

    array_electrons = array_electrons[0:parallel_size, 0:serial_size]
    bias_electrons = bias_electrons[0:parallel_size, 0:serial_size]

    """
    Rotate to orient for CTI clicking in arctic (no rotation for quadrant A)
    """
    array_electrons = array_electrons
    bias_electrons = bias_electrons

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
Load and plot the already complete ACS data reduction of this example image, for extraction in half A.
"""
file_path = path.join(dataset_path, "j9epf6kjq_raw.fits")
bias_path = path.join(dataset_path, "q4a1532mj_bia.fits")

image = extract_quadrant_a(file_path=file_path, bias_path=bias_path)

mat_plot_2d = aplt.MatPlot2D(cmap=aplt.Cmap(vmin=0.0, vmax=10.0))
array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

print("\n\nA Via Explicit function RESULTS:\n")
print(
    f"Corner Values = {image.native[0,0]}, {image.native[0, -1]}, {image.native[-1, 0]}, {image.native[-1, -1]}"
)
print(f"Min / Max Values = {np.max(image)}, {np.min(image)}")
print(f"Shape of Bias Extraction A (via autoarray) = {image.shape_native}")