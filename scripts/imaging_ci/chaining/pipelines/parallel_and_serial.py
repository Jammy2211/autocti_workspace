"""
Pipelines: Parallel CTI
========================

By chaining together five searches this script fits A CTI model using `ImagingCI`, where in the final model:

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

The paths pointing to the dataset we will use for cti modeling.
"""
dataset_name = "parallel_x2_serial_x2"
dataset_path = path.join("dataset", "imaging_ci", "uniform", dataset_name)

"""
__Layout__

Set up the 2D layout of the charge injection data and load it using this layout.
"""
shape_native = (2000, 100)

parallel_overscan = ac.Region2D((1980, 2000, 5, 95))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, 95, 100))

regions_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

normalization_list = [100, 5000, 25000, 84700]

layout_list = [
    ac.ci.Layout2DCIUniform(
        shape_2d=shape_native,
        region_list=regions_list,
        normalization=normalization,
        parallel_overscan=parallel_overscan,
        serial_prescan=serial_prescan,
        serial_overscan=serial_overscan,
    )
    for normalization in normalization_list
]

imaging_ci_list = [
    ac.ci.ImagingCI.from_fits(
        image_path=path.join(dataset_path, f"image_{layout.normalization}.fits"),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{layout.normalization}.fits"
        ),
        pre_cti_image_path=path.join(
            dataset_path, f"pre_cti_image_{layout.normalization}.fits"
        ),
        layout=layout,
        pixel_scales=0.1,
    )
    for layout in layout_list
]

imaging_ci_plotter = aplt.ImagingCIPlotter(imaging=imaging_ci_list[0])
imaging_ci_plotter.subplot_imaging_ci()

"""
__Masking__
"""
mask = ac.ci.Mask2DCI.unmasked(
    shape_native=shape_native, pixel_scales=imaging_ci_list[0].pixel_scales
)

imaging_ci_masked_list = [
    imaging_ci.apply_mask(mask=mask) for imaging_ci in imaging_ci_list
]

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging_ci", "chaining", "parallel_x2_serial_x2")

"""
__Parallel Clocking (Searches 1 & 2)__

The `Clocker` models the CCD read-out, including CTI. We use different clockers for different searches:

 Searches 1 & 2) Parallel only clocking (including 'charge injection mode').
 Searches 3 & 4) Serial only clocking.
 Searches 5,6 & 7) Parallel and serial joint clocking. 
"""
parallel_clocker = ac.Clocker(parallel_express=2, parallel_charge_injection_mode=True)
serial_clocker = ac.Clocker(serial_express=2)
parallel_serial_clocker = ac.Clocker(
    parallel_express=2, parallel_charge_injection_mode=True, serial_express=2
)

"""
__Model + Search + Analysis + Model-Fit (Search 1)__

In Search 1 we fit a CTI model with:

 - One parallel `TrapInstantCapture`'s species [2 parameters].

 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.
"""
parallel_ccd = af.PriorModel(ac.CCDPhase)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 84700.0

model = af.Collection(
    cti=af.Model(
        ac.CTI,
        parallel_traps=[af.PriorModel(ac.TrapInstantCapture)],
        parallel_ccd=parallel_ccd,
    )
)

"""
__Search + Analysis + Model-Fit (Search 1)__

To reduce run-times, we trim the `ImagingCI` data from the high resolution data (e.g. 2000 columns) to just 50 columns 
to speed up the model-fit at the expense of inferring larger errors on the CTI model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[1]_parallel[x1]", nlive=50
)

imaging_ci_trimmed_list = [
    imaging_ci.apply_settings(
        settings=ac.ci.SettingsImagingCI(parallel_columns=(0, 50))
    )
    for imaging_ci in imaging_ci_list
]

analysis = ac.AnalysisImagingCI(
    dataset_ci_list=imaging_ci_trimmed_list, clocker=parallel_clocker
)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Model (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, with:

 - Two or more parallel `TrapInstantCapture`'s species [4+ parameters: prior on density initialized from search 1].

 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter: priors initialized 
 from search 1].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5 or more.
"""
parallel_trap_0 = af.Model(ac.TrapInstantCapture)
parallel_trap_1 = af.Model(ac.TrapInstantCapture)

parallel_trap_0.density = af.UniformPrior(
    lower_limit=0.0, upper_limit=result_1.instance.cti.parallel_traps[0].density
)
parallel_trap_1.density = af.UniformPrior(
    lower_limit=0.0, upper_limit=result_1.instance.cti.parallel_traps[0].density
)

parallel_ccd = result_1.model.cti.parallel_ccd

model = af.Collection(
    cti=af.Model(
        ac.CTI,
        parallel_traps=[parallel_trap_0, parallel_trap_1],
        parallel_ccd=parallel_ccd,
    )
)

"""
__Search + Analysis + Model-Fit (Search 2)__

We use a non-linear search with slower more thorough settings, so it can robustly sample the complex parameter space. 
This is necessary given that  many parameters in the model are not yet initialized and assume broad uniform priors. 

We again use the trimmed `ImagingCI` data to speed up run-times.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[2]_parallel[multi]", nlive=50
)

analysis = ac.AnalysisImagingCI(
    dataset_ci_list=imaging_ci_trimmed_list, clocker=parallel_clocker
)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

In Search 3 we fit a CTI model with:

 - One serial `TrapInstantCapture`'s species [2 parameters].

 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.
"""
serial_ccd = af.PriorModel(ac.CCDPhase)
serial_ccd.well_notch_depth = 0.0
serial_ccd.full_well_depth = 84700.0

model = af.Collection(
    cti=af.Model(
        ac.CTI,
        serial_traps=[af.PriorModel(ac.TrapInstantCapture)],
        serial_ccd=serial_ccd,
    )
)

"""
__Search + Dataset + Analysis + Model-Fit (Search 3)__

To reduce run-times, we trim the `ImagingCI` data from the high resolution data (e.g. 2000 columns) to just 10 rows of 
every charge injection region to speed up the model-fit at the expense of inferring larger errors on the CTI model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[3]_serial[x1]", nlive=50
)

imaging_ci_trimmed_list = [
    imaging_ci.apply_settings(settings=ac.ci.SettingsImagingCI(serial_rows=(0, 10)))
    for imaging_ci in imaging_ci_list
]

analysis = ac.AnalysisImagingCI(
    dataset_ci_list=imaging_ci_trimmed_list, clocker=serial_clocker
)

result_3 = search.fit(model=model, analysis=analysis)

"""
__Model (Search 4)__

We use the results of search 3 to create the CTI model fitted in search 3, with:

 - Two or more serial `TrapInstantCapture`'s species [4+ parameters: prior on density initialized from search 1].

 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter: priors initialized 
 from search 1].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5 or more.
"""
serial_trap_0 = af.Model(ac.TrapInstantCapture)
serial_trap_1 = af.Model(ac.TrapInstantCapture)

serial_trap_0.density = af.UniformPrior(
    lower_limit=0.0, upper_limit=result_1.instance.cti.serial_traps[0].density
)
serial_trap_1.density = af.UniformPrior(
    lower_limit=0.0, upper_limit=result_1.instance.cti.serial_traps[0].density
)

serial_ccd = result_3.model.cti.serial_ccd

model = af.Collection(
    cti=af.Model(
        ac.CTI, serial_traps=[serial_trap_0, serial_trap_1], serial_ccd=serial_ccd
    )
)

"""
__Search + Dataset + Analysis + Model-Fit (Search 4)__

We use a non-linear search with slower more thorough settings, so it can robustly sample the complex parameter space. 
This is necessary given that  many parameters in the model are not yet initialized and assume broad uniform priors. 

We again use the trimmed `ImagingCI` data to speed up run-times.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[4]_serial[multi]", nlive=50
)

analysis = ac.AnalysisImagingCI(
    dataset_ci_list=imaging_ci_trimmed_list, clocker=serial_clocker
)

result_4 = search.fit(model=model, analysis=analysis)

"""
__Model (Search 5)__

We use the results of searches 2 & 4 to create the CTI model fitted in search 5, with:

 - Two or more parallel `TrapInstantCapture`'s species [4+ parameters: prior on density initialized from search 2].

 - Two or more serial `TrapInstantCapture`'s species [4+ parameters: prior on density initialized from search 4].

 - A simple `CCD` volume filling parametrization for parallel clocking [1 parameter: priors initialized from search 2].

 - A simple `CCD` volume filling parametrization for serial clocking [1 parameter: priors initialized from search 4].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10 or more.
"""
model = af.Collection(
    cti=af.Model(
        ac.CTI,
        parallel_traps=result_4.model.cti.parallel_traps,
        parallel_ccd=result_4.model.cti.parallel_ccd,
        serial_traps=result_4.model.cti.serial_traps,
        serial_ccd=result_4.model.cti.serial_ccd,
    )
)

"""
__Search + Dataset + Analysis + Model-Fit (Search 5)__

We use a non-linear search with slower more thorough settings, so it can robustly sample the complex parameter space. 
This is necessary because although the parallel and serial CTI models have been initialized pretty well, they are not
yet perfect and there is a high probability the CTI model will shift from the previous estimate. 

To accurately clock parallel and serial CTI we cannot trim the data, thus the `ImagingCI` data at native resolution is
used.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[5]_parallel[multi]_serial[multi]", nlive=100
)

analysis = ac.AnalysisImagingCI(
    dataset_ci_list=imaging_ci_masked_list, clocker=serial_clocker
)

result_4 = search.fit(model=model, analysis=analysis)


"""
Finish.
"""
