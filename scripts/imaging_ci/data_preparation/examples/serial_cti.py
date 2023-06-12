"""
Data Preparation: Serial CTI
----------------------------

The `preprocess_1_serial_cti.py` script shows how to estimate the pre-CTI charge injection images from an observed
charge injection data which includes parallel CTI.

This was centred around using the inner regions of the FPR of the charge injection data, which should not have had
any electrons captured due to CTI.

This script builds on this example, showing how this process can account for serial CTI, which moves electrons between
the charge injection line FPR regions. This assumes that a known serial CTI model is available, for example
from the previous day's CTI calibration observations.

I recommend you have completed all previous preprocess scripts before this one.
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

We begin by loading a charge injection image which has parallel and serial CTI added, which we will use to illustrate 
pre-CTI estimation methods.

We also set up its corresponding `Layout2DCI` object, which is used to estimate the charge injection normalization
in the FPR / charge injection regions.

You should be familiar with the **PyAutoCTI** API below, if not check out other scripts throughout the workspace.
"""
dataset_name = "serial_cti"
dataset_path = path.join("dataset", "imaging_ci", dataset_name)

shape_native = (2000, 100)

"""
The overscans input here are not used to perform pre-CTI data estimation.
"""
parallel_overscan = ac.Region2D((1980, 2000, 5, 95))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, 95, 100))

"""
These charge injection regions are used to estimate the charge injection normalization in each column.
"""
region_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

norm = 5000

"""
The layout object contains all the charge injection normalization functionality used in this example.
"""
layout = ac.Layout2DCI(
    shape_2d=shape_native,
    region_list=region_list,
    parallel_overscan=parallel_overscan,
    serial_prescan=serial_prescan,
    serial_overscan=serial_overscan,
)

"""
We will demonstrate injection estimation on a charge injection image which we load via a .fits file below.
"""
dataset = ac.ImagingCI.from_fits(
    data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
    noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
    pre_cti_data_path=path.join(dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"),
    layout=layout,
    pixel_scales=0.1,
)

"""
A plot of the data shows it has non-uniform charge injection lines and cosmic rays.
"""
array_2d_plotter = aplt.Array2DPlotter(array=dataset.data)
array_2d_plotter.figure_2d()

"""
__Normalization List__

The `proprocess_1_pre_cti.py` example explains the code below, which estimates the charge injection normalizations 
from the inner regions of each FPR.
"""
injection_norm_list = layout.extract.parallel_fpr.median_list_from(
    array=dataset.data, settings=ac.SettingsExtract(pixels=(150, 200))
)

print(injection_norm_list)

"""
__Pre CTI Image Estimate__

From this `normalization_list` we can create a pre-CTI charge injection image, where each column corresponds to
the normalization estimated above.

Note that due to serial CTI mixing, we are expecting this image to be less accurate of an estimate than we found
in the first data preparation example.
"""
pre_cti_data = layout.pre_cti_data_non_uniform_from(
    injection_norm_list=injection_norm_list, pixel_scales=dataset.data.pixel_scales
)

"""
If we plot the original data and this pre-CTI estimate we can see they are similar by eye.
"""
array_2d_plotter = aplt.Array2DPlotter(array=dataset.data)
array_2d_plotter.figure_2d()

array_2d_plotter = aplt.Array2DPlotter(array=pre_cti_data)
array_2d_plotter.figure_2d()

"""
However, if we subtract the two images, we find that there are residuals contained in the parallel and serial 
FPR and EPERs.

These are because our pre-CTI estimate image does not account for the CTI contained in the original data.
"""
residual_map = dataset.data.native - pre_cti_data.native

array_2d_plotter = aplt.Array2DPlotter(array=residual_map)
array_2d_plotter.figure_2d()

"""
The aim of this example is to account for the impact of serial CTI mixing in the charge injection FPR, which the
residual map above shows but does not show clearly.

We can get an image the residuals only due to this effect by adding the true parallel and serial CTI models to the
estimated image above. Because the CTI model is the same one used to simulate the data, the results residuals can only
be attributed to a misestimation of the pre-CTI data.
"""
clocker = ac.Clocker2D(
    parallel_express=5,
    parallel_roe=ac.ROEChargeInjection(),
    parallel_fast_mode=True,
    serial_express=5,
    iterations=5,
)

parallel_trap_0 = ac.TrapInstantCapture(density=0.13, release_timescale=1.25)
parallel_trap_1 = ac.TrapInstantCapture(density=0.25, release_timescale=4.4)

parallel_trap_list = [parallel_trap_0, parallel_trap_1]

parallel_ccd = ac.CCDPhase(
    well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
)


serial_trap_0 = ac.TrapInstantCapture(density=0.0442, release_timescale=0.8)
serial_trap_1 = ac.TrapInstantCapture(density=0.1326, release_timescale=4.0)
serial_trap_2 = ac.TrapInstantCapture(density=3.9782, release_timescale=20.0)

serial_trap_list = [serial_trap_0, serial_trap_1, serial_trap_2]

serial_ccd = ac.CCDPhase(
    well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
)

cti = ac.CTI2D(
    parallel_trap_list=parallel_trap_list,
    parallel_ccd=parallel_ccd,
    serial_trap_list=serial_trap_list,
    serial_ccd=serial_ccd,
)

post_cti_data = clocker.add_cti(data=pre_cti_data, cti=cti)

"""
We now use this post-CTI data to create the residual map.

It shows a distinct rippling effect, whereby certain charge injections leave large residuals because their charge
estimate has been most impacted by serial CTI. This is the effect we need to remove to get an accurate charge
injection image estimate.
"""
residual_map = dataset.data.native - post_cti_data

array_2d_plotter = aplt.Array2DPlotter(array=residual_map)
array_2d_plotter.figure_2d()

"""
__CTI Correction__

To mitigate this effect, we use a previous estimate of the parallel and serial CTI models to correct the charge 
injection data.

This should relocate the majority of electrons back to their original charge injection lines, such that our
estimate of the pre-CTI charge injection data is now accurate.
"""
data_corrected = clocker.remove_cti(data=dataset.data, cti=cti)

injection_norm_list = layout.extract.parallel_fpr.median_list_from(
    array=data_corrected, settings=ac.SettingsExtract(pixels=(0, 200))
)

pre_cti_data = layout.pre_cti_data_non_uniform_from(
    injection_norm_list=injection_norm_list, pixel_scales=dataset.data.pixel_scales
)

"""
As we did above, we want to determine if the residuals due to the misestimation of charge injection normalizations
has been accounted for. 

Thus we again add the true CTI model to this estimated pre-CTI data before computing the residuals.

Upon plotting the residuals, we find that the rippling effect has been removed and we are left with a clean
pre-CTI dataset.
"""
post_cti_data = clocker.add_cti(data=pre_cti_data, cti=cti)

residual_map = dataset.data.native - post_cti_data

array_2d_plotter = aplt.Array2DPlotter(array=residual_map)
array_2d_plotter.figure_2d()

"""
__Parallel CTI Correction__

In the previous preprocesing tutorial, parallel CTI was in the data but a model was not used for correction. This works
because the charge injection is uniform in the parallel direction, meaning that electrons are only captured and
release after the FPR (all traps are filled when clocking the inner regions of the FPR).

Nevertheless, it is likely one would have a parallel CTI model available to them and this could be used to correct
CTI from the data before estimating the pre-CTI image. This would mean that the full FPR could be used (instead of
the inner regions), but has the downside that if the CTI model is not perfect uncertainty will be introduced.

__What if I dont have a serial CTI model?__

If a serial CTI model is not available, it is probably still possible to get an accurate estimate of the pre-CTI 
data. 

For example, one could imagine altering the **PyAutoCTI** likelihood function to work as follows:

 1) Assume a serial CTI model and use this to correct serial CTI from the charge injection image.
 2) Use this serial CTI corrected data to estimate the pre-CTI data using the functionality illustrated above.
 3) Use the same serial CTI model to add CTI to this pre-CTI data estimate.
 4) Compute the likelihood using this data as per usual.

We have not found a use-case to write such an algorithm, but if it sounds like you could use one please contact us!

__Wrap Up__

This example shows how serial CTI degrades the use of the inner FPR regions to estimate the pre-CTI data, and
that this can be mitigated by using an already known serial CTI model to correct the data first. 

We next consider how cosmic rays can be flagged and removed from charge injection imaging.
"""
