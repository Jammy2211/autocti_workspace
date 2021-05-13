"""
Pipelines: Serial CTI
=====================

By chaining together three searches this script fits A CTI model using `ImagingCI`, where in the final model:

 - The CTI model consists of an input number of serial trap species.
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
dataset_name = "serial_x2"
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
path_prefix = path.join("imaging_ci", "chaining", "serial")

"""
__Clocking__

The `Clocker` models the CCD read-out, including CTI. 
"""
clocker = ac.Clocker(serial_express=2)

"""
__Model + Search + Analysis + Model-Fit (Search 1)__

In Search 1 we fit a CTI model with:

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
__Search + Dataset + Analysis + Model-Fit (Search 1)__

To reduce run-times, we trim the `ImagingCI` data from the high resolution data (e.g. 2000 columns) to just 10 rows of 
every charge injection region to speed up the model-fit at the expense of inferring larger errors on the CTI model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[1]_serial[x1]", nlive=50
)

imaging_ci_trimmed_list = [
    imaging_ci.apply_settings(settings=ac.ci.SettingsImagingCI(serial_rows=(0, 10)))
    for imaging_ci in imaging_ci_list
]

analysis = ac.AnalysisImagingCI(
    dataset_ci_list=imaging_ci_trimmed_list, clocker=clocker
)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Model (Search 2)__

We use the results of search 1 to create the CTI model fitted in search 2, with:

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

serial_ccd = result_1.model.cti.serial_ccd

model = af.Collection(
    cti=af.Model(
        ac.CTI, serial_traps=[serial_trap_0, serial_trap_1], serial_ccd=serial_ccd
    )
)

"""
__Search + Dataset + Analysis + Model-Fit (Search 2)__

We use a non-linear search with slower more thorough settings, so it can robustly sample the complex parameter space. 
This is necessary given that  many parameters in the model are not yet initialized and assume broad uniform priors. 

We again use the trimmed `ImagingCI` data to speed up run-times.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[2]_serial[multi]", nlive=50
)

analysis = ac.AnalysisImagingCI(
    dataset_ci_list=imaging_ci_trimmed_list, clocker=clocker
)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

In Search 3 we fit a CTI model with:

 - The same number of trap species as search 2 [4+ parameters: priors initialized from search 2].

 - The same `CCD` volume filling parametrization as search 2 [1 parameter: priors initialized from search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5 or more.

The key difference from search 2 is that the value of every parameter is initialized (ensuring a more accurate and
efficient non-linear search) and that we do not trim the data to only 10 rows of serial trails such that are errors are 
representative of fitting all available data.
"""
serial_ccd = af.PriorModel(ac.CCDPhase)
serial_ccd.well_notch_depth = 0.0
serial_ccd.full_well_depth = 84700.0

model = af.Collection(
    cti=af.Model(
        ac.CTI,
        serial_traps=result_2.model.cti.serial_traps,
        serial_ccd=result_2.model.cti.serial_ccd,
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[3]_serial[multi]", nlive=50
)

analysis = ac.AnalysisImagingCI(dataset_ci_list=imaging_ci_masked_list, clocker=clocker)

result_3 = search.fit(model=model, analysis=analysis)

"""
Finish.
"""
