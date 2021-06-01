"""
This script illustrates extractions of CCD sections from ACS .fits imaging. It does not perform any unit conversions
or bias subtractions, it only shows how rotations are handled.
"""

import numpy as np
import os
from autoconf import conf
import matplotlib.pyplot as plt

import pickle
import autoarray as aa
import autoarray.plot as aplt
from os import path

"""
The dataset path to the ACS data.
"""
dataset_path = path.join("acs", "dataset")

"""
Load and plot the already complete ACS data reduction of this example image, for extraction in half A.
"""
file_path = path.join(dataset_path, "j9epf6kjq_rawA.fits")
image = aa.Array2D.from_fits(file_path=file_path, hdu=0, pixel_scales=0.05)


mat_plot_2d = aplt.MatPlot2D(cmap=aplt.Cmap(vmin=0.0, vmax=10.0))
array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

print("\nRAW A FILE RESULTS:\n")
print(
    f"Corner Values = {image.native[0,0]}, {image.native[0, -1]}, {image.native[-1, 0]}, {image.native[-1, -1]}"
)
print(f"Min / Max Values = {np.max(image)}, {np.min(image)}")
print(f"Shape of Extraction A (HST reduction) = {image.shape_native}")

"""
Extraction A: Lets now use PyAutoArray's `ImageACS` class method to load quadrant A from this data, which does the 
following:

 - Loads the data from HDU 1.
 - Loads the right hand half of data corresponding to the numpy indexes array[0:2068, 2072:4144].

We will compare this to the already processed quadrant A dataset (which is in different units).
"""
file_path = path.join(dataset_path, "j9epf6kjq_raw.fits")
bias_path = path.join(dataset_path, "q4a1532mj_bia.fits")

image_aa = aa.acs.ImageACS.from_fits(
    file_path=file_path, quadrant_letter="A", bias_path=bias_path
)


mat_plot_2d = aplt.MatPlot2D(cmap=aplt.Cmap(vmin=0.0, vmax=10.0))
array_plotter = aplt.Array2DPlotter(array=image_aa, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

print("\n\nA Via AutoArray RESULTS:\n")
print(
    f"Corner Values = {image_aa.native[0,0]}, {image_aa.native[0, -1]}, {image_aa.native[-1, 0]}, {image_aa.native[-1, -1]}"
)
print(f"Min / Max Values = {np.max(image_aa)}, {np.min(image_aa)}")
print(f"Shape of Bias Extraction A (via autoarray) = {image_aa.shape_native}")

residuals = image - image_aa

array_plotter = aplt.Array2DPlotter(array=residuals)
array_plotter.figure_2d()

print("\n\nResiduals of Two images:\n")
print(
    f"Corner Values = {residuals.native[0,0]}, {residuals.native[0, -1]}, {residuals.native[-1, 0]}, {residuals.native[-1, -1]}"
)
print(f"Min / Max Values = {np.max(residuals)}, {np.min(residuals)}")
print(f"Shape of Bias Extraction A (via autoarray) = {residuals.shape_native}")
