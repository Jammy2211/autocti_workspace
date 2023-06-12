"""
Chaining: x1 Species to x2 Species
==================================

In this script, we chain two searches to fit `ImagingCI` with a CTI model where:

 - The final CTI model consists of two parallel `Trap` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.

The two searches break down as follows:

 1) Model CTI using a single parallel trap species and volume filling parameterization.
 2) Model CTI using two parallel trap species and volume filling parameterization.

__Why Chain?__

A CTI model with two more trap species is slower and more difficult to fit than model with one trap species, because:

 1) It has more free parameters and therefore a higher dimensionality non-linear parameter space.
 2) Degeneracies between the trap species release time parameters can be challenging for the non-linear search to
 sample accurately and efficiently.

By first fitting a CTI model containing just one species, we can make estimates of some aspects of the CTI model, which
we then use to initialize the second search in the right regions of parameter space. For example, the first search
will provide a reasonably accurate estimate of the total density of trap_list and the volume filling parameters of the CCD.
These results are not perfect, but they can be obtained quickly and are "good enough" to initialize the second
search's model-fit with two (or more) trap species.
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
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging_ci", "chaining", "x1_species_to_x2_species")

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
parallel_ccd = af.Model(ac.CCDPhase)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 200000.0

model_1 = af.Collection(
    cti=af.Model(
        ac.CTI2D,
        parallel_trap_list=[af.Model(ac.TrapInstantCapture)],
        parallel_ccd=parallel_ccd,
    )
)

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
__Settings + Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

To reduce run-times, we trim the `ImagingCI` data from the high resolution data (e.g. 100 columns) to just 5 columns 
to speed up the model-fit at the expense of inferring larger errors on the CTI model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate CTI model.
"""
imaging_ci_trim_list = [
    dataset.apply_settings(settings=ac.SettingsImagingCI(parallel_pixels=(0, 5)))
    for dataset in dataset_list
]

analysis_1_list = [
    ac.AnalysisImagingCI(dataset=dataset, clocker=clocker)
    for dataset in imaging_ci_trim_list
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

search_1 = af.DynestyStatic(
    path_prefix=path_prefix, name="search[1]_species[x1]", nlive=50
)

result_1_list = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Model (Search 2)__

We use the results of search 1 to create the model fitted in search 2, with:

 - Two parallel `TrapInstantCapture`'s species [4 parameters: prior on density initialized from search 1].

 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter: priors initialized 
 from search 1].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.

The first search gives an accurate estimate of the total density of trap_list. It is therefore reasonable to use this as 
the upper limit on the density of every individual trap in this model.

The term `model` below passes the CTI model's `parallel_ccd` as a model-component that is to be fitted for by the 
non-linear search.  
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

model_2 = af.Collection(
    cti=af.Model(
        ac.CTI2D,
        parallel_trap_list=[parallel_trap_0, parallel_trap_1],
        parallel_ccd=parallel_ccd,
    )
)

"""
The `info` attribute shows the model, including how parameters and priors were passed from `result_1`.
"""
print(model_2.info)

"""
__Settings + Search + Analysis + Model-Fit (Search 2)__

We now create the non-linear search, analysis and perform the model-fit using this model.

Whereas in the previous search we reduced run-times by trimming the data to just 5 columns, when we perform search
chaining we can increase this in the second search. The run times will slow down, but the model we infer will be
more accurate and precise.
"""
imaging_ci_trim_list = [
    dataset.apply_settings(settings=ac.SettingsImagingCI(parallel_pixels=(0, 25)))
    for dataset in dataset_list
]

analysis_2_list = [
    ac.AnalysisImagingCI(dataset=dataset, clocker=clocker)
    for dataset in imaging_ci_trim_list
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

search_2 = af.DynestyStatic(
    path_prefix=path_prefix, name="search[2]_species[x2]", nlive=50
)

result_2_list = search_2.fit(model=model_2, analysis=analysis_2)

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
