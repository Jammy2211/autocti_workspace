"""
Charge Injection Imaging: Data Preparation
==========================================

When a charge injection imaging dataset is analysed, it must conform to certain standards in order
for the analysis to be performed correctly. This tutorial describes these standards and links to more detailed scripts
which will help you prepare your dataset to adhere to them if it does not already.

__Pixel Scale__

The "pixel_scale" of the image (and the data in general) is pixel-units to arcsecond-units conversion factor of
your telescope. You should look up now if you are unsure of the value.

The pixel scale of some common telescopes is as follows:

 - Hubble Space telescope 0.04" - 0.1" (depends on the instrument and wavelength).
 - James Webb Space telescope 0.06" - 0.1" (depends on the instrument and wavelength).
 - Euclid 0.1" (Optical VIS instrument) and 0.2" (NIR NISP instrument).

It is absolutely vital you use the correct pixel scale, so double check this value!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autocti as ac
import autocti.plot as aplt


"""
__Dataset + Layout__

The paths pointing to the dataset we will use for CTI modeling.
"""
dataset_name = "non_uniform"
dataset_path = path.join("dataset", "imaging_ci", dataset_name)

"""
__Shape__

The 2D shape of the images.
"""
shape_native = (2000, 100)

"""
__Regions__

We next define the locations of the prescan and overscan on the 2D data. 

2D regions are defined as a tuple of the form (y0, y1, x0, x1) = (top-row, bottom-row, left-column, right-column), 
where the integer values of the tuple are used to perform NumPy array indexing of the 2D data.

For example, if the serial overscan of 2D data is 100 columns from the read-out electronics and spans a total of
150 rows, its region is `region=(0, 150, 0, 100)`.

These are used to visualize these regions of the 2D CTI dataset during the model-fit and customize aspects of the 
model-fit.
"""
parallel_overscan = ac.Region2D((1980, 2000, 5, 95))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, 95, 100))

"""
Specify the charge regions on the 2D CTI Dataset, corresponding to where a signal is contained that has its electrons 
captured and trailed by CTI (e.g. the FPR).

This dataset has five charge regions, which are spaced in on / off blocks of 200 pixels.

Note that the charge injections do not extend to inside of the serial prescan or serial overscan regions.
"""
region_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

"""
Specify the normalization of the charge in the dataset we use to illustrate data preparation.

This is not used internally by **PyAutoCTI**, and only required for loading the dataset because the dataset file
names use the normalizations.
"""
norm = 5000

"""
__Layout__

We now create a `Layout2D` object for every 1D dataset fitted in this script.

This object contains all functionality associated with the layout of the data (e.g. where the FPR is, where the
EPERs are, where the overscans are, etc.). 

This is used for performing tasks like extracting a small region of the data for visualization.
"""
layout = ac.Layout2DCI(
    shape_2d=shape_native,
    region_list=region_list,
    parallel_overscan=parallel_overscan,
    serial_prescan=serial_prescan,
    serial_overscan=serial_overscan,
)

"""
__Dataset__

We now use a `ImagingCI` object to load every 2D CTI charge injection dataset, including a noise-map and pre-cti data 
containing the data before read-out and therefore without CTI. 

The `pixel_scales` define the arc-second to pixel conversion factor of the image, which for the dataset we are using 
is 0.1" / pixel.
"""
dataset = ac.ImagingCI.from_fits(
    data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
    noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
    pre_cti_data_path=path.join(dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"),
    layout=layout,
    pixel_scales=0.1,
)

"""
Use a `ImagingCIPlotter` to the plot the data.
"""
array_2d_plotter = aplt.Array2DPlotter(array=dataset.data)
array_2d_plotter.figure_2d()

"""
__Image__

This image conforms to **PyAutoCTI** standards for the following reasons.

 - Units: The image flux is in units of electrons (as opposed to electrons, counts, ADU`s etc.). 
   Internal **PyAutoCTI** functions which perform CTI clocking assume the image is in electrons.
   
 - Bias: Although not clear from the visual itself, the image has been bias subtracted, which **PyAutoCTI**
   assumes has always been performed for data it processes (in addition to other effects like non-linearity).
   
If your image conforms to all of the above standards, you are good to use it for an analysis (but must also check
you noise-map and PSF conform to standards first!).

If it does not, checkout the `examples/bias_subtraction.ipynb` notebooks for tools to process the data so it does (or 
use your own data reduction tools to do so).

This workspace does not currently have an example of how to convert your data from another data unit to electrons,
because this is often an instrument specific process which a general example cannot cover. 

__Noise Map__

The noise-map defines the uncertainty in every pixel of your strong lens image, where values are defined as the 
RMS standard deviation in every pixel (not the variances, HST WHT-map values, etc.). 

Lets inspect a noise-map which conforms to **PyAutoCTI** standards:
"""
array_plotter = aplt.Array2DPlotter(array=dataset.noise_map)
array_plotter.figure_2d()

"""
This noise-map conforms to **PyAutoCTI** standards for the following reasons:

 - Units: Like its corresponding image, it is in units of electrons (as opposed to electrons per second, counts, 
   ADU`s etc.). Internal **PyAutoCTI** functions for computing quantities like a galaxy magnitude assume the data and 
   model light profiles are in electrons per second.

 - Values: The noise-map values themselves are the RMS standard deviations of the noise in every pixel. When a model 
   is fitted to data in **PyAutoCTI** and a likelihood is evaluated, this calculation assumes that this is the
   corresponding definition of the noise-map. The noise map therefore should not be the variance of the noise, or 
   another definition of noise.

If you are not certain what the definition of the noise-map you have available to you is, or do not know how to
compute a noise-map at all, you should refer to the instrument handbook of the telescope your data is from. It is
absolutely vital that the noise-map is correct, as it is the only way to quantify the goodness-of-fit.

A sanity check for a reliable noise map is that there is a a near constant set of values corresponding to the 
read-out noise of the instrument. There are not many other sources of noise in charge injection data, but
there may be charge injection noise in the FPR.
   
If your noise-map conforms to all of the above standards, you are good to use it for an analysis (but must also check
you image conform to standards first!).

If it does not, checkout the `examples/noise_map.ipynb` notebook for tools to process the data so it does (or use your 
own data reduction tools to do so).

__Pre CTI Data__

To perform CTI calibration, we need to know what the data looked like before read-out and therefore before CTI. 
This is because CTI calibration first adds CTI to this image, before subtracting it from the observed image to
quantify the likelihood and CTI effect.

The pre-CTI data can be estimated from the observed image using its first pixel response (FPR). This is because no
electrons are captured in the central region os the FPR (because all traps on the CCD are already full). 
There the median of the inner region of the FPR can be used to estimate the pre-CTI data.

Other aspects of charge injection data, such as column-to-column non-uniformity and charge injection noise can also
be estimate using the FPR.

Lets inspect pre-cti data which conforms to **PyAutoCTI** standards:
"""
array_plotter = aplt.Array2DPlotter(array=dataset.pre_cti_data)
array_plotter.figure_2d()

"""
This conforms to **PyAutoCTI** standards for the following reasons.

 - Units: Like its corresponding image, it is in units of electrons (as opposed to electrons per second, counts, 
   ADU`s etc.). 
   
 - Accuracy: The pre-CTI data is estimated from the FPR and therefore provides an accurate estimate of the
   signal truly injected into the CCD before read-out. 

If your pre-CTI data conforms to all of the above standards, you are good to use it for an analysis (but must also check
you noise-map and image conform to standards first!).

If it does not, checkout the `examples/pre_cti.ipynb` notebook for tools to process the data so it does (or use your 
own data reduction tools to do so).

__Cosmic Ray Flagging__

Charge injection data taken in space is affected by cosmic rays. These are high energy particles which appear as
delta functions (or small extended spikes) in the image. 

Because one has no knowledge of when a cosmic ray hit the CCD, it is impossible to know when it occured and thus
no useful information can be extracted from it about CTI. 

Therefore, cosmic rays must be flagged and masked so they do not affect the analysis. This masking must also include 
their CTI EPER trails.

Due to the regular and predictable nature of charge injection data, it is possible to flag cosmic rays to a very high
completeness using simple thresholding techniques. The process therefore does not use more complicated techniques
which are often applied to science data, which by default would flag the charge injection regions as cosmic rays.

Lets inspect a cosmic ray map which conforms to **PyAutoCTI** standards:
"""
dataset_path = path.join("dataset", "imaging_ci", "cosmic_rays")

cosmic_ray_map = ac.Array2D.from_fits(
    file_path=path.join(dataset_path, f"norm_{int(norm)}", "cosmic_ray_map.fits"),
    pixel_scales=0.1,
)

array_plotter = aplt.Array2DPlotter(array=cosmic_ray_map)
array_plotter.figure_2d()

"""
This conforms to **PyAutoCTI** standards for the following reasons.

 - Accuracy: The cosmic ray map has successfully identified the vast majority of cosmic rays in the image.

__Data Processing Complete__

If your image, noise-map and PSF conform the standards above, you are ready to analyse your dataset!

Below, we provide an overview of optional data preparation steos which prepare other aspects of the analysis. 

New users are recommended to skim-read the optional steps below so they are aware of them, but to not perform them 
and instead analyse their dataset now. You can come back to the data preparation scripts below if it becomes necessary.
"""
