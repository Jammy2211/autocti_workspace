"""
Chaining: Noise Scaling
=======================

In this script, we chain two searches to fit `ImagingCI` with a CTI model where:

 - The final CTI model consists of two parallel `Trap` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.

The simulated data has Poisson trap densities, which lead to large chi-squareds in our model fit due to the
assumption of a single average trap density. Our second search therefore applies noise scaling, which increases
the noise in locations of the fit with large chi-squareds to down weight their contribution to the model inference.

The two searches break down as follows:

 1) Model CTI using a single parallel trap species and volume filling parameterization.
 2) Model CTI using the same model, but with noise scaling applied.

__Why Chain?__

The noise scaling maps must preferentially scale columns of data where there are large chi-squareds. These are
the columns whose densities are larger outliers.

In order to scale the nosie in this way, we therefore require an intiial fit that gives us the initial chi
squared map used to inform noise scaling.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import copy
from os import path
import autofit as af
import autocti as ac
import autocti.plot as aplt

"""
__Dataset__ 

The paths pointing to the dataset we will use for CTI modeling.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "imaging_ci", "poisson", dataset_name)

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

norm_list = [100, 200000]

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
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging_ci", "chaining", "noise_scaling")

"""
__Full__

Below, we are going to mask the data and extract a subset of the imaging data, which we will fit with a CTI model. 

Default visualization will be performed on this masked and extracted data, therefore not giving a complete picture of
how the model fits the overall data.

We create a deepcopy of the imaging data before masking / extraction, and visualization of the model-fit will also 
be performed on this full dataset, giving a complete  picture of the model-fit.
"""
imaging_ci_full_list = copy.deepcopy(dataset_list)

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
parallel_trap_0 = af.Model(ac.TrapInstantCapture)
parallel_trap_1 = af.Model(ac.TrapInstantCapture)

parallel_trap_0.add_assertion(
    parallel_trap_0.release_timescale < parallel_trap_1.release_timescale
)
parallel_trap_list = [parallel_trap_0, parallel_trap_1]

parallel_ccd = af.Model(ac.CCDPhase)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 200000.0

model_1 = af.Collection(
    cti=af.Model(
        ac.CTI2D, parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd
    )
)

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
__Settings + Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate CTI model.
"""
analysis_1_list = [
    ac.AnalysisImagingCI(dataset=dataset, clocker=clocker) for dataset in dataset_list
]

"""
By summing this list of analysis objects, we create an overall `Analysis` which we can use to fit the CTI model, where:

 - The log likelihood function of this summed analysis class is the sum of the log likelihood functions of each 
 individual analysis object.

 - The summing process ensures that tasks such as outputting results to hard-disk, visualization, etc use a 
 structure that separates each analysis.
"""
analysis_1 = sum(analysis_1_list)
analysis_1.n_cores = 1

search_1 = af.Nautilus(
    path_prefix=path_prefix, name="search[1]_species[x1]", n_live=100
)

result_1_list = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Noise Map Scaling__

We now set each charge injection imaging dataset with a noise scaling map, where this noise-scaling map is the
chi-squared values of different regions of the image.

For this script we only perform noise map scaling in one region of the data, the parallel EPERs.
"""
[
    imaging_ci.set_noise_scaling_map_dict(
        noise_scaling_map_dict=result_1.noise_scaling_map_dict
    )
    for dataset, result_1 in zip(dataset_list, result_1_list)
]

"""
__Model (Search 2)__

We use the results of search 1 to create the model fitted in search 2, with:

 - a free parameter in every analysis which fits the noise scaling in the parallel EPERs.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=2.
"""
model_2 = af.Collection(
    cti=af.Model(
        ac.CTI2D,
        parallel_trap_list=result_1_list.instance.cti.parallel_trap_list,
        parallel_ccd=result_1_list.instance.cti.parallel_ccd,
    ),
    hyper_noise=af.Model(
        ac.HyperCINoiseCollection,
        regions_ci=ac.HyperCINoiseScalar,
        parallel_eper=ac.HyperCINoiseScalar,
    ),
)

"""
__Settings + Search + Analysis + Model-Fit (Search 2)__

We now create the non-linear search, analysis and perform the model-fit using this model.

Whereas in the previous search we reduced run-times by trimming the data to just 5 columns, when we perform search
chaining we can increase this in the second search. The run times will slow down, but the model we infer will be
more accurate and precise.
"""
analysis_2_list = [
    ac.AnalysisImagingCI(dataset=dataset, clocker=clocker) for dataset in dataset_list
]

"""
By summing this list of analysis objects, we create an overall `Analysis` which we can use to fit the CTI model, where:

 - The log likelihood function of this summed analysis class is the sum of the log likelihood functions of each 
 individual analysis object.

 - The summing process ensures that tasks such as outputting results to hard-disk, visualization, etc use a 
 structure that separates each analysis.
"""
analysis_2 = sum(analysis_2_list)
analysis_2.n_cores = 1

"""
Make noise map scaling parameter free across every dataset.
"""
analysis_2 = analysis_2.with_free_parameters(
    model_2.hyper_noise.regions_ci, model_2.hyper_noise.parallel_eper
)

"""
The `info` attribute shows the model, including how parameters and priors were passed from `result_1`.
"""
print(model_2.info)

search_2 = af.Nautilus(
    path_prefix=path_prefix, name="search[2]_species[x2]", n_live=100
)

# result_2_list = search_2.fit(model=model_2, analysis=analysis_2)

result_2_list = search_2.fit_sequential(model=model_2, analysis=analysis_2)

"""
__Model (Search 3)__

We use the results of search 1 and 2 to create the model fitted in search 3, with:

 - Two parallel `TrapInstantCapture`'s species [4 parameters: prior on density initialized from search 1].

 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter: priors initialized 
 from search 1].
 
 - The noise map scaling applied.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.
"""
parallel_trap_0 = af.Model(ac.TrapInstantCapture)
parallel_trap_1 = af.Model(ac.TrapInstantCapture)
parallel_trap_list = [parallel_trap_0, parallel_trap_1]

parallel_ccd = af.Model(ac.CCDPhase)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 200000.0

model_3 = af.Collection(
    cti=af.Model(
        ac.CTI2D, parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd
    ),
    hyper_noise=af.Model(
        ac.HyperCINoiseCollection,
        regions_ci=ac.HyperCINoiseScalar,
        parallel_eper=ac.HyperCINoiseScalar,
    ),
)

"""
The `info` attribute shows the model, including how parameters and priors were passed from `result_1` and `result_2`.
"""
print(model_3.info)

"""
__Settings + Search + Analysis + Model-Fit (Search 3)__

We now create the non-linear search, analysis and perform the model-fit using this model.

Whereas in the previous search we reduced run-times by trimming the data to just 5 columns, when we perform search
chaining we can increase this in the second search. The run times will slow down, but the model we infer will be
more accurate and precise.
"""
analysis_3_list = [
    ac.AnalysisImagingCI(dataset=dataset, clocker=clocker) for dataset in dataset_list
]

"""
By summing this list of analysis objects, we create an overall `Analysis` which we can use to fit the CTI model, where:

 - The log likelihood function of this summed analysis class is the sum of the log likelihood functions of each 
 individual analysis object.

 - The summing process ensures that tasks such as outputting results to hard-disk, visualization, etc use a 
 structure that separates each analysis.
"""
analysis_3 = sum(analysis_3_list)
analysis_3.n_cores = 1

analysis_3 = analysis_3.with_free_parameters()

search_3 = af.Nautilus(
    path_prefix=path_prefix, name="search[3]_species[x2]", n_live=100
)

result_3_list = search_3.fit(model=model_3, analysis=analysis_3)

"""
__Wrap Up__

In this example, we passed used prior passing to initialize a CTI model with multiple trap species with a sensible
prior for the total density of trap_list based on a fit using a single species. We also pass information on the CCD volume
filling behaviour.

__Pipelines__

Advanced search chaining uses `pipelines` that chain together multiple searches to perform complex CTI modeling in a 
robust and efficient way. 

The following example pipelines fits a two trap species CTI model, using the same approach demonstrated in this script 
of first fitting a single species:

 `autocti_workspace/imaging/chaining/pipelines/parallel.py`
 `autocti_workspace/imaging/chaining/pipelines/serial.py`
"""
