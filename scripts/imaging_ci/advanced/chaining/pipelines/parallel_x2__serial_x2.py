"""
Pipelines: Parallel CTI
========================

By chaining together three searches this script fits A CTI model using `ImagingCI`, where in the final model:

 - The CTI model consists of an input number of parallel trap species and an input number of serial trap species.
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
dataset_name = "parallel_x2__serial_x2"
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
"""
mask = ac.Mask2D.all_false(
    shape_native=shape_native, pixel_scales=dataset_list[0].pixel_scales
)

dataset_masked_list = [dataset.apply_mask(mask=mask) for dataset in dataset_list]

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging_ci", "chaining", "parallel_x2_serial_x2")

"""
__Model (Search 1)__

We use the results of search 1 to create the CTI model fitted in search 1, with:

 - Two or more parallel `TrapInstantCapture`'s species [4+ parameters].

 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5 or more.
"""
parallel_trap_0 = af.Model(ac.TrapInstantCapture)
parallel_trap_1 = af.Model(ac.TrapInstantCapture)

parallel_trap_0.add_assertion(
    parallel_trap_0.release_timescale < parallel_trap_1.release_timescale
)

parallel_trap_list = [parallel_trap_0, parallel_trap_1]

parallel_ccd = af.Model(ac.CCDPhase)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 200000.0

model = af.Collection(
    cti=af.Model(
        ac.CTI2D, parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd
    )
)

"""
__Search + Analysis + Model-Fit (Search 1)__

To reduce run-times, we trim the `ImagingCI` data from the high resolution data (e.g. 100 columns) to just 5 columns 
to speed up the model-fit at the expense of inferring larger errors on the CTI model.
"""
parallel_clocker = ac.Clocker2D(
    parallel_express=5, parallel_roe=ac.ROEChargeInjection(), parallel_fast_mode=True
)

search = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]_parallel[multi]",
    n_live=150,
)

imaging_ci_trim_list = [
    dataset.apply_settings(settings=ac.SettingsImagingCI(parallel_pixels=(50, 90)))
    for dataset in dataset_list
]

analysis_list = [
    ac.AnalysisImagingCI(dataset=dataset, clocker=parallel_clocker)
    for dataset in imaging_ci_trim_list
]

analysis = sum(analysis_list)
analysis.n_cores = 1

result_1_list = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

We use the results of search 3 to create the CTI model fitted in search 3, with:

 - Two or more serial `TrapInstantCapture`'s species [4+ parameters: prior on density initialized from search 1].

 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter: priors initialized 
 from search 1].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5 or more.
"""
serial_trap_0 = af.Model(ac.TrapInstantCapture)
serial_trap_1 = af.Model(ac.TrapInstantCapture)

serial_trap_0.add_assertion(
    serial_trap_0.release_timescale < serial_trap_1.release_timescale
)

serial_trap_list = [serial_trap_0, serial_trap_1]

serial_ccd = af.Model(ac.CCDPhase)
serial_ccd.well_notch_depth = 0.0
serial_ccd.full_well_depth = 200000.0

model = af.Collection(
    cti=af.Model(
        ac.CTI2D,
        parallel_trap_list=result_1_list.instance.cti.parallel_trap_list,
        parallel_ccd=result_1_list.instance.cti.parallel_ccd,
        serial_trap_list=serial_trap_list,
        serial_ccd=serial_ccd,
    )
)

"""
__Dataset (Search 2)__

In the second search we only fit a serial CTI model. 

However, it is benefitial if our pre-CTI data includes parallel CTI, as this will improve the accuracy of our inferred
serial CTI model. 

To achieve this, we can update the `ImagingCI` object to use the maximum likelihood post-CTI data inferred from the 
fit above.
"""
# dataset_list = [
#     ac.ImagingCI(
#         image=dataset.data,
#         noise_map=dataset.noise_map,
#         pre_cti_data=result_1.max_log_likelihood_full_fit.post_cti_data,
#         layout=dataset.layout,
#     )
#     for dataset, result_1 in zip(dataset_list, result_1_list)
# ]

"""
__Search + Analysis + Model-Fit (Search 2)__

We now perform the model-fit on this dataset, as per usual.
"""
search = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]_serial[multi]",
    n_live=150,
)

serial_clocker = ac.Clocker2D(
    parallel_express=5,
    parallel_roe=ac.ROEChargeInjection(),
    serial_express=5,
    serial_fast_mode=True,
)

analysis_list = [
    ac.AnalysisImagingCI(dataset=dataset, clocker=serial_clocker)
    for dataset in dataset_list
]

analysis = sum(analysis_list)
analysis.n_cores = 1

result_2_list = search.fit(model=model, analysis=analysis)

"""
__Model (Search 3)__

We use the results of searches 2 & 4 to create the CTI model fitted in search 5, with:

 - Two or more parallel `TrapInstantCapture`'s species [4+ parameters: prior on density initialized from search 2].

 - Two or more serial `TrapInstantCapture`'s species [4+ parameters: prior on density initialized from search 4].

 - A simple `CCD` volume filling parametrization for parallel clocking [1 parameter: priors initialized from search 2].

 - A simple `CCD` volume filling parametrization for serial clocking [1 parameter: priors initialized from search 4].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10 or more.
"""
model = af.Collection(
    cti=af.Model(
        ac.CTI2D,
        parallel_trap_list=result_2_list[0].model.cti.parallel_trap_list,
        parallel_ccd=result_2_list[0].model.cti.parallel_ccd,
        serial_trap_list=result_2_list[0].model.cti.serial_trap_list,
        serial_ccd=result_2_list[0].model.cti.serial_ccd,
    )
)

"""
__Search + Dataset + Analysis + Model-Fit (Search 3)__

We use a non-linear search with slower more thorough settings, so it can robustly sample the complex parameter space. 
This is necessary because although the parallel and serial CTI models have been initialized pretty well, they are not
yet perfect and there is a high probability the CTI model will shift from the previous estimate. 

To accurately clock parallel and serial CTI we cannot trim the data, thus the `ImagingCI` data at native resolution is
used.
"""
parallel_serial_clocker = ac.Clocker2D(
    parallel_express=5,
    parallel_roe=ac.ROEChargeInjection(),
    serial_express=5,
    parallel_fast_mode=True,
)

search = af.Nautilus(
    path_prefix=path_prefix, name="search[3]_parallel[multi]_serial[multi]", n_live=150
)

analysis_list = [
    ac.AnalysisImagingCI(dataset=dataset, clocker=parallel_serial_clocker)
    for dataset in dataset_list
]

analysis = sum(analysis_list)
analysis.n_cores = 1

result_5_list = search.fit(model=model, analysis=analysis)

"""
Finish.
"""
