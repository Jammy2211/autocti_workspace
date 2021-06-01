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
file_path = path.join(dataset_path, "q4a1532mj_bia.fits")
bias = aa.Array2D.from_fits(file_path=file_path, hdu=1, pixel_scales=0.05)

array_plotter = aplt.Array2DPlotter(array=bias)
array_plotter.figure_2d()

print(f"Shape of Bias Image = {bias.shape_native}")

"""
Different units of bias file that are possible:
"""

print(
    np.max(
        aa.util.array_2d.numpy_array_2d_from_fits(
            file_path=file_path, hdu=1, do_not_scale_image_data=False
        )
    )
)

print(
    np.max(
        aa.util.array_2d.numpy_array_2d_from_fits(
            file_path=file_path, hdu=1, do_not_scale_image_data=True
        )
    )
)

print(np.max(bias))
