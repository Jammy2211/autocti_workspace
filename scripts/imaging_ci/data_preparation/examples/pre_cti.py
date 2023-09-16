"""
Data Preparation: Pre CTI
-------------------------

Throughout the `autocti_workspace/imaging_ci` package, the charge injection imaging `pre_cti_data` is output to .fits
format when data is simulated and loaded from .fits for modeling scripts.

This perfect knowledge of the pre-CTI data is only possible for a real telescope if:

 - The charge injection is temporally stable, such that the pre-CTI data is (close to) identical every time data is
 acquired.

 - The charge injection's appearance has been quantified pre-launch, before there are significant levels of CTI on the
 CCD.

This script demonstrates **PyAutoCTI** methods which estimate the pre-CTI data from charge injection imaging data,
even when that charge injection imaging is subject to CTI during read out.

This uses the fact that the inner regions of each First Pixel Response (FPR) of each charge injection region should
not have had any electrons captured, because all traps are filled by the front pixels in the FPR. By taking the median
of the inner regions of the FPR one can therefore estimate how much charge was injected.

This script demonstrates a simple example, where only parallel CTI is present in the CCD and there are no cosmic
rays in the charge injection imaging. The script `advanced.py` shows how this can be done with serial CTI and cosmic
rays also included.
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

We begin by loading a charge injection image which has parallel CTI added, which we will use to illustrate pre-CTI 
estimation methods.

We also set up its corresponding `Layout2DCI` object, which is used to estimate the charge injection normalization
in the FPR / charge injection regions.

You should be familiar with the **PyAutoCTI** API below, if not check out other scripts throughout the workspace.
"""
dataset_name = "non_uniform"
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
A plot of the data shows it has non-uniform charge injection lines.
"""
array_2d_plotter = aplt.Array2DPlotter(array=dataset.data)
array_2d_plotter.figure_2d()

"""
__Normalization List__

The 2D region of every set of parallel FPRs on the charge injection data is contained in the layout's extract
object.

To estimate the normalization of each FPR's inner region, we use the `median_list_from` of the
charge injection `Layout2DCI` object with the input `pixels=(180, 200)`, which:

 - Extracts the parallel FPR of every charge injection region between the inner 180th to 200th pixels (noting that
 the `region_list` above defining where the charge injections are has each FPR spanning 200 pixels.

 - Stacks all of the extracted parallel FPR's to remove read-noise.

 - Takes the median of these stacks to estimate the normalization value.

The normalizations are returned as a list:
"""
injection_norm_list = layout.extract.parallel_fpr.median_list_from(
    array=dataset.data, settings=ac.SettingsExtract(pixels=(150, 200))
)

"""
The number of entries in the list corresponds to the number of columns of charge injection:
"""
print(len(injection_norm_list))
print(region_list[0][3] - region_list[0][2])

"""
__Pre CTI Image Estimate__

From this `normalization_list` we can create a pre-CTI charge injeciton image, where each column corresponds to
the normalization estimated above.

This again uses the charge injection data's layout attribute, which knows where the charge injections are contained
on the data.
"""
pre_cti_data = layout.pre_cti_data_non_uniform_from(
    injection_norm_list=injection_norm_list, pixel_scales=dataset.data.pixel_scales
)

"""
If we plot the original data and this pre-CTI estimate we can see they are similar.
"""
array_2d_plotter = aplt.Array2DPlotter(array=dataset.data)
array_2d_plotter.figure_2d()

array_2d_plotter = aplt.Array2DPlotter(array=pre_cti_data)
array_2d_plotter.figure_2d()

"""
If we subtract the two images, we find that the only residuals left are contained in the parallel FPR and EPERs.

These are because our pre-CTI estimate image does not account for the CTI contained in the original data.
"""
residual_map = dataset.data.native - pre_cti_data.native

array_2d_plotter = aplt.Array2DPlotter(array=residual_map)
array_2d_plotter.figure_2d()

"""
__CTI Modeling__

We now demonstrate that this pre-CTI data can be used to estimate an accurate CTI model, with a quick model-fit.

The `ImagingCI` data loaded above contained the true `pre_cti_data`, which was loaded via a .fits file. We create a 
new instance of the `ImagingCI` data which uses the pre-CTI image we estimated above.
"""
dataset = ac.ImagingCI(
    data=dataset.data,
    noise_map=dataset.noise_map,
    pre_cti_data=pre_cti_data,
    layout=dataset.layout,
)

"""
If you are not familiar with the CTI modeling API, checkout the scripts contained in 
the `autocti_workspace/*/imaging_ci/modeling` package.
"""
clocker = ac.Clocker2D(
    parallel_express=5, parallel_roe=ac.ROEChargeInjection(), parallel_fast_mode=True
)

parallel_trap_0 = af.Model(ac.TrapInstantCapture)
parallel_trap_list = [parallel_trap_0]

parallel_ccd = af.Model(ac.CCDPhase)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 200000.0

model = af.Collection(
    cti=af.Model(
        ac.CTI2D, parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd
    )
)

search = af.Nautilus(
    path_prefix=path.join("imaging_ci", "pre_cti_estimate_simple"),
    name="parallel[x1]",
    n_live=100,
)

analysis = ac.AnalysisImagingCI(dataset=dataset, clocker=clocker)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

We now use the maximum likelihood inferred CTI model to add CTI to the pre-CTI data estimated above and used in this
model-fit.
"""
instance = result.max_log_likelihood_instance

post_cti_data = clocker.add_cti(data=pre_cti_data, cti=instance.cti)

residual_map = dataset.data.native - post_cti_data

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
