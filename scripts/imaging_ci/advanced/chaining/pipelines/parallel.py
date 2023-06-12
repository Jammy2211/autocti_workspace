"""
Pipelines: Parallel CTI
========================

By chaining together three searches this script fits A CTI model using `ImagingCI`, where in the final model:

 - The CTI model consists of an input number of parallel trap species.
 - The `CCD` volume filling is an input.
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
__Dataset__ 

The paths pointing to the dataset we will use for CTI modeling.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "imaging_ci", dataset_name)

"""
__Layout__

Set up the 2D layout of the charge injection data and load it using this layout.
"""
shape_native = (2000, 100)

parallel_overscan = ac.Region2D((1980, 2000, 5, 95))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, 95, 100))

region_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

norm_list = [100, 5000, 25000, 200000]

total_datasets = len(norm_list)

layout_list = [
    ac.Layout2DCI(
        shape_2d=shape_native,
        region_list=region_list,
        parallel_overscan=parallel_overscan,
        serial_prescan=serial_prescan,
        serial_overscan=serial_overscan,
    )
    for i in range(total_datasets)
]

dataset_list = [
    ac.ImagingCI.from_fits(
        data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
        noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
        pre_cti_data_path=path.join(
            dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
        ),
        layout=layout,
        pixel_scales=0.1,
    )
    for layout, norm in zip(layout_list, norm_list)
]

dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset_list[0])
dataset_plotter.subplot_dataset()

"""
__Mask__

We apply a 2D mask which removes the FPR (e.g. all 200 pixels where the charge injection is performed).
"""
mask = ac.Mask2D.all_false(
    shape_native=dataset_list[0].shape_native,
    pixel_scales=dataset_list[0].pixel_scales,
)

mask = ac.Mask2D.masked_fpr_and_eper_from(
    mask=mask,
    layout=dataset_list[0].layout,
    settings=ac.SettingsMask2D(parallel_fpr_pixels=(0, 200)),
    pixel_scales=dataset_list[0].pixel_scales,
)

dataset_list = [dataset.apply_mask(mask=mask) for dataset in dataset_list]

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging_ci", "chaining", "parallel")

"""
__Clocking__

The `Clocker` models the CCD read-out, including CTI. 

For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""
clocker = ac.Clocker2D(
    parallel_express=5, parallel_roe=ac.ROEChargeInjection(), parallel_fast_mode=True
)

"""
__Model (Search 1)__

In Search 1 we fit a CTI model with:

 - One parallel `TrapInstantCapture`'s species [2 parameters].

 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.
"""
parallel_ccd = af.Model(ac.CCDPhase)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 200000.0

model = af.Collection(
    cti=af.Model(
        ac.CTI2D,
        parallel_trap_list=[af.Model(ac.TrapInstantCapture)],
        parallel_ccd=parallel_ccd,
    )
)

"""
__Settings + Search + Analysis + Model-Fit (Search 1)__

To reduce run-times, we trim the `ImagingCI` data from the high resolution data (e.g. 100 columns) to just 5 columns 
to speed up the model-fit at the expense of inferring larger errors on the CTI model.
"""
imaging_ci_trim_list = [
    dataset.apply_settings(settings=ac.SettingsImagingCI(parallel_pixels=(0, 5)))
    for dataset in dataset_list
]

search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[1]_parallel[x1]", nlive=50
)

analysis_list = [
    ac.AnalysisImagingCI(dataset=dataset, clocker=clocker)
    for dataset in imaging_ci_trim_list
]

analysis = sum(analysis_list)
analysis.n_cores = 1

result_1_list = search.fit(model=model, analysis=analysis)

"""
__Model (Search 2)__

We use the results of search 1 to create the CTI model fitted in search 2, with:

 - Two or more parallel `TrapInstantCapture`'s species [4+ parameters: prior on density initialized from search 1].

 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter: priors initialized 
 from search 1].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5 or more.
"""
parallel_trap_0 = af.Model(ac.TrapInstantCapture)
parallel_trap_1 = af.Model(ac.TrapInstantCapture)

parallel_trap_0.density = af.UniformPrior(
    lower_limit=0.0,
    upper_limit=result_1_list[0].instance.cti.parallel_trap_list[0].density,
)
parallel_trap_1.density = af.UniformPrior(
    lower_limit=0.0,
    upper_limit=result_1_list[0].instance.cti.parallel_trap_list[0].density,
)

parallel_ccd = result_1_list[0].model.cti.parallel_ccd

model = af.Collection(
    cti=af.Model(
        ac.CTI2D,
        parallel_trap_list=[parallel_trap_0, parallel_trap_1],
        parallel_ccd=parallel_ccd,
    )
)

"""
__Settings + Search + Analysis + Model-Fit (Search 2)__

We use a non-linear search with slower more thorough settings, so it can robustly sample the complex parameter space. 
This is necessary given that  many parameters in the model are not yet initialized and assume broad uniform priors. 

We again use the trimmed `ImagingCI` data to speed up run-times.
"""
imaging_ci_trim_list = [
    dataset.apply_settings(settings=ac.SettingsImagingCI(parallel_pixels=(0, 5)))
    for dataset in dataset_list
]

search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[2]_parallel[multi]", nlive=50
)

analysis_list = [
    ac.AnalysisImagingCI(dataset=dataset, clocker=clocker)
    for dataset in imaging_ci_trim_list
]

analysis = sum(analysis_list)
analysis.n_cores = 1

result_2_list = search.fit(model=model, analysis=analysis)

"""
__Model (Search 3)__

In Search 3 we fit a CTI model with:

 - The same number of trap species as search 2 [4+ parameters: priors initialized from search 2].

 - The same `CCD` volume filling parametrization as search 2 [1 parameter: priors initialized from search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5 or more.
"""
parallel_ccd = af.Model(ac.CCDPhase)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 200000.0

model = af.Collection(
    cti=af.Model(
        ac.CTI2D,
        parallel_trap_list=result_2_list[0].model.cti.parallel_trap_list,
        parallel_ccd=result_2_list[0].model.cti.parallel_ccd,
    )
)

"""
__Settings + Model + Search + Analysis + Model-Fit (Search 3)__

Now the value of every parameter is initialized (ensuring a more accurate and efficient non-linear search) and we do 
not trim the data to only 50 parallel columns and again use thorough non-linear search settings.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[3]_parallel[multi]", nlive=50
)

analysis_list = [
    ac.AnalysisImagingCI(dataset=dataset, clocker=clocker) for dataset in dataset_list
]

analysis = sum(analysis_list)
analysis.n_cores = 1

result_3_list = search.fit(model=model, analysis=analysis)

"""
Finish.
"""
