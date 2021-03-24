"""
Pipelines: Parallel CTI
========================

By chaining together three searches this script fits A CTI model using `CIImaging`, where in the final model:

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
__Dataset + Masking__ 

Load, plot and mask the `CIImaging` data, including the set up of its charge injection region, pattern, normalizations,
etc.
"""
dataset_name = "parallel_x2"
dataset_path = path.join("dataset", "ci_imaging", "uniform", dataset_name)

scans = ac.Scans(
    parallel_overscan=ac.Region2D((1980, 2000, 5, 95)),
    serial_prescan=ac.Region2D((0, 2000, 0, 5)),
    serial_overscan=ac.Region2D((0, 1980, 95, 100)),
)

ci_regions = [
    (0, 200, scans.serial_prescan[3], scans.serial_overscan[2]),
    (400, 600, scans.serial_prescan[3], scans.serial_overscan[2]),
    (800, 1000, scans.serial_prescan[3], scans.serial_overscan[2]),
    (1200, 1400, scans.serial_prescan[3], scans.serial_overscan[2]),
    (1600, 1800, scans.serial_prescan[3], scans.serial_overscan[2]),
]

normalizations = [100, 5000, 25000, 84700]

ci_patterns = [
    ac.ci.CIPatternUniform(normalization=normalization, regions=ci_regions)
    for normalization in normalizations
]

ci_imaging_list = [
    ac.ci.CIImaging.from_fits(
        image_path=path.join(dataset_path, f"image_{pattern.normalization}.fits"),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{pattern.normalization}.fits"
        ),
        ci_pre_cti_path=path.join(
            dataset_path, f"ci_pre_cti_{pattern.normalization}.fits"
        ),
        pixel_scales=0.1,
        ci_pattern=pattern,
        roe_corner=(1, 0),
    )
    for pattern in ci_patterns
]

imaging_plotter = aplt.CIImagingPlotter(imaging=ci_imaging_list[0])
imaging_plotter.subplot_imaging()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("ci_imaging", "chaining", "parallel")

"""
__Clocking__

The `Clocker` models the CCD read-out, including CTI. 

For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""
clocker = ac.Clocker(parallel_express=2, parallel_charge_injection_mode=True)

"""
__Model + Search + Analysis + Model-Fit (Search 1)__

In Search 1 we fit a CTI model with:

 - One parallel `TrapInstantCapture`'s species [2 parameters].

 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.
"""
parallel_ccd = af.PriorModel(ac.CCD)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 84700

model = af.Model(
    ac.CTI,
    parallel_traps=[af.PriorModel(ac.TrapInstantCapture)],
    parallel_ccd=parallel_ccd,
)

"""
__Search + Analysis + Model-Fit (Search 1)__

To reduce run-times, we trim the `CIImaging` data from the high resolution data (e.g. 2000 columns) to just 50 columns 
to speed up the model-fit at the expense of inferring larger errors on the CTI model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[1]_parallel[x1]", n_live_points=50
)

ci_imaging_trimmed_list = [
    ac.ci.MaskedCIImaging(
        ci_imaging=ci_imaging,
        mask=ac.ci.CIMask2D.unmasked(
            shape_native=ci_imaging.shape_native, pixel_scales=ci_imaging.pixel_scales
        ),
        settings=ac.ci.SettingsMaskedCIImaging(parallel_columns=(0, 50)),
    )
    for ci_imaging in ci_imaging_list
]

analysis = ac.AnalysisCIImaging(dataset_list=ci_imaging_list, clocker=clocker)

result_1 = search.fit(model=model, analysis=analysis)

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
    lower_limit=0.0, upper_limit=result_1.instance.cti.parallel_traps[0].density
)
parallel_trap_1.density = af.UniformPrior(
    lower_limit=0.0, upper_limit=result_1.instance.cti.parallel_traps[0].density
)

parallel_ccd = result_1.model.cti.parallel_ccd

model = af.Model(
    ac.CTI, parallel_traps=[parallel_trap_0, parallel_trap_1], parallel_ccd=parallel_ccd
)

"""
__Search + Analysis + Model-Fit (Search 2)__

We use a non-linear search with slower more thorough settings, so it can robustly sample the complex parameter space. 
This is necessary given that  many parameters in the model are not yet initialized and assume broad uniform priors. 

We again use the trimmed `CIImaging` data to speed up run-times.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[2]_parallel[multi]", n_live_points=50
)

analysis = ac.AnalysisCIImaging(dataset_list=ci_imaging_trimmed_list, clocker=clocker)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

In Search 3 we fit a CTI model with:

 - The same number of trap species as search 2 [4+ parameters: priors initialized from search 2].

 - The same `CCD` volume filling parametrization as search 2 [1 parameter: priors initialized from search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5 or more.

The key difference from search 2 is that the value of every parameter is initialized (ensuring a more accurate and
efficient non-linear search) and that we do not trim the data to only 50 parallel columns such that are errors are 
representative of fitting all available data.
"""
parallel_ccd = af.PriorModel(ac.CCD)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 84700

model = af.Model(
    ac.CTI,
    parallel_traps=result_2.model.cti.parallel_traps,
    parallel_ccd=result_2.model.cti.parallel_ccd,
)

search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[3]_parallel[multi]", n_live_points=50
)

analysis = ac.AnalysisCIImaging(dataset_list=ci_imaging_list, clocker=clocker)

result_3 = search.fit(model=model, analysis=analysis)

"""
Finish.
"""
