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
will provide a reasonably accurate estimate of the total density of traps and the volume filling parameters of the CCD.
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

The paths pointing to the dataset we will use for cti modeling.
"""
dataset_name = "parallel_x2"
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
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging_ci", "chaining", "x1_species_to_x2_species")

"""
__Clocking__

The `Clocker` models the CCD read-out, including CTI. 

For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""
clocker = ac.Clocker(parallel_express=2, parallel_charge_injection_mode=True)

"""
__Model (Search 1)__

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

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate CTI model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[1]_species[x1]", nlive=50
)

analysis = ac.AnalysisImagingCI(dataset_ci_list=imaging_ci_list, clocker=clocker)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Model (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, with:

 - Two parallel `TrapInstantCapture`'s species [4 parameters: prior on density initialized from search 1].

 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter: priors initialized 
 from search 1].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.

The first search gives an accurate estimate of the total density of traps. It is therefore reasonable to use this as 
the upper limit on the density of every individual trap in this model.

The term `model` below passes the CTI model's `parallel_ccd` as a model-component that is to be fitted for by the 
non-linear search.  
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

We now create the non-linear search, analysis and perform the model-fit using this model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[2]_species[x2]", nlive=50
)

analysis = ac.AnalysisImagingCI(dataset_ci_list=imaging_ci_list, clocker=clocker)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Wrap Up__

In this example, we passed used prior passing to initialize a CTI model with multiple trap species with a sensible
prior for the total density of traps based on a fit using a single species. We also pass information on the CCD volume
filling behaviour.

__Pipelines__

Advanced search chaining uses `pipelines` that chain together multiple searches to perform complex CTI modeling in a 
robust and efficient way. 

The following example pipelines fits a two trap species CTI model, using the same approach demonstrated in this script 
of first fitting a single species:

 `autocti_workspace/imaging/chaining/pipelines/parallel.py`
 `autocti_workspace/imaging/chaining/pipelines/serial.py`
"""
