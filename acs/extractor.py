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
The images have a high dyanmic range, so the cmap below makes them plot in a clearer way.
"""
cmap = aplt.Cmap(vmin=2000.0, vmax=3000.0)
mat_plot_2d = aplt.MatPlot2D(cmap=cmap)

"""
Load & plot the original HST ACS data and print its shape before extraction.

This loads both images in the .fits file, which correspond to HDU 1 and HDU 4.
"""
dataset_path = path.join("acs", "dataset")
file_path = path.join(dataset_path, "j9epf6kjq_raw.fits")

array_hdu1 = aa.Array2D.from_fits(file_path=file_path, hdu=1, pixel_scales=0.05)

array_plotter = aplt.Array2DPlotter(array=array_hdu1.native, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

print(f"Shape of original CCD (HDU 1) = {array_hdu1.shape_native}")

array_hdu4 = aa.Array2D.from_fits(file_path=file_path, hdu=4, pixel_scales=0.05)

array_plotter = aplt.Array2DPlotter(array=array_hdu4.native, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

print(f"Shape of original CCD (HDU 4) = {array_hdu4.shape_native}")

"""
Extraction A: Lets now use PyAutoArray's `ImageACS` class method to load quadrant A from this data, which does the 
following:

 - Loads the data from HDU 1.
 - Loads the right hand half of data corresponding to the numpy indexes array[0:2068, 2072:4144].

We will compare this to the already processed quadrant A dataset (which is in different units).
"""
cmap = aplt.Cmap(vmin=100.0, vmax=300.0)
mat_plot_2d = aplt.MatPlot2D(cmap=cmap)

file_path_a = path.join(dataset_path, "j9epf6kjq_rawC.fits")
image = aa.Array2D.from_fits(file_path=file_path_a, hdu=0, pixel_scales=0.05)

array_plotter = aplt.Array2DPlotter(array=image, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

print(f"Shape of Extraction A (HST reduction) = {image.shape_native}")


cmap = aplt.Cmap(vmin=2000.0, vmax=3000.0)
mat_plot_2d = aplt.MatPlot2D(cmap=cmap)

image_a_via_autoarray = aa.acs.ImageACS.from_fits(
    file_path=file_path, quadrant_letter="A"
)

array_plotter = aplt.Array2DPlotter(
    array=image_a_via_autoarray, mat_plot_2d=mat_plot_2d
)
array_plotter.figure_2d()

print(f"Shape of original Extraction A (via autoarray) = {image.shape_native}")


cmap = aplt.Cmap(vmin=2000.0, vmax=3000.0)
mat_plot_2d = aplt.MatPlot2D(cmap=cmap)

image_a_via_autoarray = aa.acs.ImageACS.from_fits(
    file_path=file_path, quadrant_letter="B"
)

array_plotter = aplt.Array2DPlotter(
    array=image_a_via_autoarray, mat_plot_2d=mat_plot_2d
)
array_plotter.figure_2d()

print(f"Shape of original Extraction A (via autoarray) = {image.shape_native}")


cmap = aplt.Cmap(vmin=2000.0, vmax=3000.0)
mat_plot_2d = aplt.MatPlot2D(cmap=cmap)

image_a_via_autoarray = aa.acs.ImageACS.from_fits(
    file_path=file_path, quadrant_letter="C"
)

array_plotter = aplt.Array2DPlotter(
    array=image_a_via_autoarray, mat_plot_2d=mat_plot_2d
)
array_plotter.figure_2d()

print(f"Shape of original Extraction A (via autoarray) = {image.shape_native}")


cmap = aplt.Cmap(vmin=2000.0, vmax=3000.0)
mat_plot_2d = aplt.MatPlot2D(cmap=cmap)

image_a_via_autoarray = aa.acs.ImageACS.from_fits(
    file_path=file_path, quadrant_letter="D"
)

array_plotter = aplt.Array2DPlotter(
    array=image_a_via_autoarray, mat_plot_2d=mat_plot_2d
)
array_plotter.figure_2d()

print(f"Shape of original Extraction A (via autoarray) = {image.shape_native}")
