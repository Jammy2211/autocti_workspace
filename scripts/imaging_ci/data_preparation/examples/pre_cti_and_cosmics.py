"""
Data Preparation: Pre-CTI and Cosmics (Optional)
------------------------------------------------

We have seen how to estimate the pre-CTI image from charge injection data (including accounting for serial CTI)
and how to flag cosmic rays in that data.

However, we have not done both at the same time; a process which is somewhat degenerate. In order to estimate
the pre-CTI image, we take the median of the inner regions of the FPR, a process which will be biased by cosmic
rays if they are not masked and flagged beforehand. However, to flag cosmic rays we first subtracted a pre-CTI
image, to ensure the cosmic ray flagging algorithm does not flag the non-uniform charge.

This tutorial demonstrates an iterative approach which alternates between these two steps, to accurately estimate
the pre-CTI data whilst fully flagging cosmic rays.

__Foreword__

I wrote this tutorial script expecting the degeneracy between cosmic rays and pre CTI estimation to be
a big issue that required an iterative approach. However, after writing the script, this does not seem neceesary
for realistic levels of cosmic rays (e.g. in Euclid data).

My conclusion was that the that ratio of pixels impacted by cosmic rays to the number of pixels available to
estimate each FPR was to low to require the iterative approach. Nevertheless, it is conceivable that there
are datasets where this approach is necessary, in which case this tutorial may prove useful.
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

We begin by loading a charge injection image which has cosmic rays and parallel and serial CTI added, which we will 
use to illustrate pre-CTI estimation methods.

We also set up its corresponding `Layout2DCI` object, which is used to estimate the charge injection normalization
in the FPR / charge injection regions.

You should be familiar with the **PyAutoCTI** API below, if not check out other scripts throughout the workspace.
"""
dataset_name = "cosmic_rays"
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

norm = 100

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
__CTI Correction__

Following the same strategy as the previous example, we correct the charge injection data for parallel and serial 
CTI before flagging cosmic rays.

Due to the interplay between the cosmic ray flagging algorithm and pre-CTI estimation routine, this simplifies the
simplifies the iterative approach necessary to perform both.
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

data_corrected = clocker.remove_cti(data=dataset.data, cti=cti)

"""
__Normalization List__

We first estimate the charge injection normalizations from the inner regions of each FPR.

Note that we have performed any cosmic ray flagging yet, thus a small fraction of estimates are likely to be very 
inaccurate because a cosmic ray overlaps the inner FPR.
"""
injection_norm_list = layout.extract.parallel_fpr.median_list_from(
    array=data_corrected, settings=ac.SettingsExtract(pixels=(150, 200))
)

print(injection_norm_list)

"""
__Pre CTI Image Estimate__

From this `normalization_list` we can create a pre-CTI charge injection image, where each column corresponds to
the normalization estimated above.

This again uses the charge injection data's layout attribute, which knows where the charge injections are contained
on the data.
"""
pre_cti_data = layout.pre_cti_data_non_uniform_from(
    injection_norm_list=injection_norm_list, pixel_scales=dataset.data.pixel_scales
)

"""
If we plot the original data and this pre-CTI estimate we can see they are similar, but that the `pre_cti_data`
does not include cosmic rays.
"""
array_2d_plotter = aplt.Array2DPlotter(array=data_corrected)
array_2d_plotter.figure_2d()

array_2d_plotter = aplt.Array2DPlotter(array=pre_cti_data)
array_2d_plotter.figure_2d()

"""
If we subtract the two images, we find that the large residuals are left due to cosmic rays.
"""
residual_map = data_corrected - pre_cti_data.native

array_2d_plotter = aplt.Array2DPlotter(array=residual_map)
array_2d_plotter.figure_2d()

"""
We are specifically interested in whether the presence of cosmic rays lead to some charge injection estimates
being inaccurate.

To plot this, we create the residual map again but first subtract all of the cosmic rays. We will use the true
cosmic ray map, which is output from the simulation script.

We add CTI to these cosmic rays before subtracting them, so that their trails are also subtracted.
"""
cosmic_ray_map = ac.Array2D.from_fits(
    file_path=path.join(dataset_path, f"norm_{int(norm)}", "cosmic_ray_map.fits"),
    pixel_scales=0.1,
)

mask = ac.Mask2D.all_false(
    shape_native=data_corrected.shape_native,
    pixel_scales=data_corrected.pixel_scales,
)

mask = ac.Mask2D.from_cosmic_ray_map_buffed(
    cosmic_ray_map=cosmic_ray_map,
    settings=ac.SettingsMask2D(
        cosmic_ray_parallel_buffer=80,
        cosmic_ray_serial_buffer=80,
        cosmic_ray_diagonal_buffer=5,
    ),
)

residual_map = residual_map.apply_mask(mask=mask)

array_2d_plotter = aplt.Array2DPlotter(array=residual_map)
array_2d_plotter.figure_2d()

"""
Had the interplay between pre-CTI estimation and cosmic rays been important I would expect a 

There is a residual signal over the data, which is due to inaccurate pre-CTI estimation in certain columns.

__Cosmic Ray Flagging__

Nevertheless, the image should be good enough to subtract the charge injections and flag the majority of cosmic rays.

[Note that the routine below does not use the true cosmic ray map loaded above, we only used that for illustration
but from here on do not assume any knowledge of the truth to proprocess the data].
"""
data_charge_subtracted = data_corrected.native - pre_cti_data.native

cr_threshold = 4.0

cosmic_ray_flag_mask = data_charge_subtracted > cr_threshold * dataset.noise_map.native

"""
We now create and plot a cosmic ray map.
"""
cosmic_ray_map = data_charge_subtracted * cosmic_ray_flag_mask

array_2d_plotter = aplt.Array2DPlotter(array=cosmic_ray_map)
array_2d_plotter.figure_2d()

"""
We subtract this from the original data to visualize how many cosmic rays were flagged.
"""
image_cosmic_ray_cleaned = data_corrected - cosmic_ray_map

array_2d_plotter = aplt.Array2DPlotter(array=image_cosmic_ray_cleaned)
array_2d_plotter.figure_2d()

"""
We now reperform pre-CTI image estimation and create the same visuals as above, to see if the residuals seen
above are reduced.
"""
injection_norm_list_after_cr = layout.extract.parallel_fpr.median_list_from(
    array=image_cosmic_ray_cleaned, settings=ac.SettingsExtract(pixels=(150, 200))
)

"""
We can compare the injection estimate including cosmic rays to the one with cosmic rays flagged.

The residuals below are all < 1.0, which is confirmation that the presence of cosmic rays in this dataset
is not having a meaningful impact on the charge injection normalization estimation.
"""
injection_residuals = [
    injection - injection_after_cr
    for injection, injection_after_cr in zip(
        injection_norm_list, injection_norm_list_after_cr
    )
]

pre_cti_data = layout.pre_cti_data_non_uniform_from(
    injection_norm_list=injection_norm_list_after_cr,
    pixel_scales=dataset.data.pixel_scales,
)

"""
Residual map plots also show no visual difference with those plotted above, again confirming the cosmic rays
are not impacting pre-CTI estimation.
"""
residual_map = data_corrected - pre_cti_data.native

array_2d_plotter = aplt.Array2DPlotter(array=residual_map)
array_2d_plotter.figure_2d()


residual_map = residual_map.apply_mask(mask=mask)

array_2d_plotter = aplt.Array2DPlotter(array=residual_map)
array_2d_plotter.figure_2d()

"""
__Iterative Approach__

We end using an iterative approach that alternatives between cosmic ray flagging and pre-CTI estimation.

As discussed throughout the tutorial, this is not necessary for this data where cosmic rays are subdominant,
but it may be important for certain datasets where cosmic rays dominate.

The routine below starts from the original data loaded from .fits, as opposed to the already corrected / flagged
data above, so it can easily be copy, pasted and used.
"""
iterations = 3

data = data_corrected

for i in range(iterations):
    injection_norm_list = layout.extract.parallel_fpr.median_list_from(
        array=data, settings=ac.SettingsExtract(pixels=(150, 200))
    )

    pre_cti_data = layout.pre_cti_data_non_uniform_from(
        injection_norm_list=injection_norm_list,
        pixel_scales=dataset.data.pixel_scales,
    )

    data_charge_subtracted = data_corrected.native - pre_cti_data.native

    cosmic_ray_flag_mask = (
        data_charge_subtracted > cr_threshold * dataset.noise_map.native
    )

    cosmic_ray_map = data_charge_subtracted * cosmic_ray_flag_mask

    data = data_corrected - cosmic_ray_map

"""
We plot the final residual map, which has had the pre-CTI estimate and all cosmic rays subtracted.
"""
residual_map = data - pre_cti_data.native

array_2d_plotter = aplt.Array2DPlotter(array=residual_map)
array_2d_plotter.figure_2d()

"""
__Wrap Up__

This example uses the inner regions of a charge injection image's FPR to estimate the original charge injection
image's appearance before clocking and therefore before electrons are captured and trailed due to CTI. 

We then showed that using this estimated image, we could fit a CTI model to the original data. This gave an accurate
model of CTI for the data, which we used to add CTI to the estimated pre-CTI data. This gave a cleaned subtract
from the original data with minimal residuals. 

The next example, titled `complex.py` uses the same tools, but extends the problem to include cosmic rays in
the charge injection image (which must be accounted for when estimate the charge levels) and serial CTI (which
moves electrons between the FPR regions we use to estimate the charge injection)
"""
