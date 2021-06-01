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
Load and plot the already complete ACS data reduction of this example image, for extraction in half B.
"""

file_path = path.join(dataset_path, "j9epf6kjq_rawB.fits")
image = aa.Array2D.from_fits(file_path=file_path, hdu=0, pixel_scales=0.05)

print(image.native[0, 0])
print(image.native[0, -1])
print(image.native[-1, 0])
print(image.native[-1, -1])

mat_plot_2d = aplt.MatPlot2D(cmap=aplt.Cmap(vmin=0.0, vmax=10.0))
array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

mat_plot_2d = aplt.MatPlot2D(cmap=aplt.Cmap(vmin=100.0, vmax=300.0))
array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

print(f"Shape of Extraction B (HST reduction) = {image.shape_native}")

cmap = aplt.Cmap(vmin=2000.0, vmax=3000.0)
mat_plot_2d = aplt.MatPlot2D(cmap=cmap)

"""
Extraction B: Lets now use PyAutoArray's `ImageACS` class method to load quadrant B from this data, which does the 
following:

 - Loads the data from HDU 1.
 - Loads the right hand half of data corresponding to the numpy indexes array[0:2068, 2072:4144].

We will compare this to the already processed quadrant A dataset (which is in different units).
"""
file_path = path.join(dataset_path, "j9epf6kjq_raw.fits")

image_aa = aa.acs.ImageACS.from_fits(file_path=file_path, quadrant_letter="B")

print(image_aa.native[0, 0])
print(image_aa.native[0, -1])
print(image_aa.native[-1, 0])
print(image_aa.native[-1, -1])

array_plotter = aplt.Array2DPlotter(array=image_aa, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

print(f"Shape of original Extraction B (via autoarray) = {image.shape_native}")
