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
Extraction A: Lets now use PyAutoArray's `ImageACS` class method to load quadrant A from this data, which does the 
following:

 - Loads the data from HDU 1.
 - Loads the right hand half of data corresponding to the numpy indexes array[0:2068, 2072:4144].

We will compare this to the already processed quadrant A dataset (which is in different units).
"""
file_path = path.join(dataset_path, "j9epf6kjq_raw.fits")
bias_path = path.join(dataset_path, "q4a1532mj_bia.fits")

image_a = aa.acs.ImageACS.from_fits(
    file_path=file_path,
    quadrant_letter="A",
    bias_path=bias_path,
    bias_subtract_via_prescan=True,
)

image_b = aa.acs.ImageACS.from_fits(
    file_path=file_path,
    quadrant_letter="B",
    bias_path=bias_path,
    bias_subtract_via_prescan=True,
)

image_c = aa.acs.ImageACS.from_fits(
    file_path=file_path,
    quadrant_letter="C",
    bias_path=bias_path,
    bias_subtract_via_prescan=True,
)

image_d = aa.acs.ImageACS.from_fits(
    file_path=file_path,
    quadrant_letter="D",
    bias_path=bias_path,
    bias_subtract_via_prescan=True,
)

"""
Ouptut these images back to the original ACS data .fits format.
"""
file_path = path.join(dataset_path, "acs_output.fits")

aa.acs.output_quadrants_to_fits(
    file_path=file_path,
    quadrant_a=image_a,
    quadrant_b=image_b,
    quadrant_c=image_c,
    quadrant_d=image_d,
    overwrite=True,
)
