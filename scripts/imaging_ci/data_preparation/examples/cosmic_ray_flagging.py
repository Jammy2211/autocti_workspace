"""
Data Preparation: Cosmic Rays
=============================

We have seen how to estimate the pre-CTI image from charge injection data, including accounting for serial CTI.

We will now demonstrate how cosmic rays can be flagged in a pre-CTI data.

Many cosmic rays hit the charge injection FPR and therefore dilute the signal used to estimate the pre-CTI data.
This creates a degeneracy between cosmic ray flagging and injection normalizaiton estimation, which requires special
care to mitigate, which the next tutorial explains.

Therefore, to provide a simple explanation of cosmic ray flagging, this tutorial uses the true pre-CTI data output
from the simulation procedure.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
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
dataset_name = "parallel_x2__serial_x3"
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
__CTI Correction__

Following the same strategy as the previous example, we correct the charge injection data for CTI before flagging
cosmic rays.

We also correct for parallel CTI, assuming that a model from a previous calibration run would be available. 

This is not absolutely necessary for cosmic ray flagging, but we will see there is an interplay between the cosmic
ray flagging algorithm and pre-CTI estimation routine, and correcting CTI first simplifies this problem.
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
__Cosmic Ray Flagging__

To flag cosmic rays, we use a simple threshold cut whereby any pixel with a signal to noise value above a threshold is 
flagged as a cosmic ray.

We use a threshold value of 4.0, which is such a high value that it is hard to associate any other signal in the data
with such high signal to noise values other than a cosmic ray. 
"""
cr_threshold = 4.0

cosmic_ray_flag_mask = data_corrected.native > cr_threshold * dataset.noise_map.native
cosmic_ray_flag_mask = ac.Array2D.no_mask(
    values=cosmic_ray_flag_mask, pixel_scales=dataset.pixel_scales
)

"""
We now plot the cosmic ray flag mask, which is a boolean array that contained `True` for any pixel 
flagged as containing a cosmic ray and `False` for all other pixels.

Inspection of the plotted image reveals two insights:

- Cosmic rays have been successfully flagged, with small streaks of flagged pixels being shown which look like
cosmic rays.

- However, the majority of flagged data is the non-uniform charge injection region.

So, whats happening? Well, we forgot an obvious fact above, that the charge injection region may also have a signal 
to noise well above our threshold value of 4.0!
"""
array_2d_plotter = aplt.Array2DPlotter(array=cosmic_ray_flag_mask)
array_2d_plotter.figure_2d()

"""
To mitigate this effect, we can simply subtract off the charge injection pattern, such that we are left with an 
image only containing cosmic rays. 
"""
pre_cti_data = ac.Array2D.from_fits(
    file_path=path.join(dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"),
    pixel_scales=0.1,
)

data_charge_subtracted = data_corrected.native - pre_cti_data.native

array_2d_plotter = aplt.Array2DPlotter(array=data_charge_subtracted)
array_2d_plotter.figure_2d()

"""
To mitigate this effect, we can simply subtract off the charge injection pattern, such that we are left with an 
image only containing cosmic rays which we can flag. 
"""
cosmic_ray_flag_mask = data_charge_subtracted > cr_threshold * dataset.noise_map.native
cosmic_ray_flag_mask = ac.Array2D.no_mask(
    values=cosmic_ray_flag_mask, pixel_scales=dataset.pixel_scales
)

array_2d_plotter = aplt.Array2DPlotter(array=cosmic_ray_flag_mask)
array_2d_plotter.figure_2d()

"""
We can also use the cosmic ray mask to create a `cosmic_ray_map`, which only contains the flagged cosmic rays.
"""
cosmic_ray_map = data_charge_subtracted * cosmic_ray_flag_mask.native
cosmic_ray_map = ac.Array2D.no_mask(
    values=cosmic_ray_map, pixel_scales=dataset.pixel_scales
)

array_2d_plotter = aplt.Array2DPlotter(array=cosmic_ray_map)
array_2d_plotter.figure_2d()

"""
__What about Cosmic Ray CTI Trails?__

Every cosmic ray in our charge injection dataset has CTI trails due to parallel and serial CTI. In this script, we 
corrected the data for CTI before performing cosmic ray flagging, therefore our cosmic ray map does not also flag
for cosmic ray trails.

For CTI modeling, these trails must also be masked to ensure CTI calibration is not biased. In modeling script
which analyse data with cosmic rays, you'll see that a special type of mask is created from the cosmic ray flag
mask, which extends the mask in the parallel and serial directions by an input number of pixels, so that the
cosmic ray trails are masked. 

In fact, this also masks the diagonal around each cosmic ray containing charge trailed first by parallel CTI, and then 
serial CTI.

If we had flagged cosmic rays using the uncorrected image, it would likely have struggled to flag the CTI trails of 
cosmic rays.. This is because these trails have a low signal and therefore low signal to noise!

__Wrap Up__

This example shows how we can flag cosmic rays in charge injection imaging data. 

The algorithm does a great job, provided we have a clean subjection of the charge injeciton reigon to ensure it does 
flag the charge injection pattern.

The next example shows how we can combine this method with the pre-CTI data estimator to flag cosmic rays in a dataset
where we do not have knowledge of the charge injection imaging beforehand. 
"""
