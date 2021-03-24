"""
Chaining: x1 Species to x2 Species
==================================

In this script, we chain two searches to fit `CIImaging` with a CTI model where:

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
path_prefix = path.join("ci_imaging", "chaining", "x1_species_to_x2_species")

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

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate CTI model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[1]_species[x1]", n_live_points=50
)

analysis = ac.AnalysisCIImaging(dataset_list=ci_imaging_list, clocker=clocker)

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

model = af.Model(
    ac.CTI, parallel_traps=[parallel_trap_0, parallel_trap_1], parallel_ccd=parallel_ccd
)

"""
__Search + Analysis + Model-Fit (Search 2)__

We now create the non-linear search, analysis and perform the model-fit using this model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[2]_species[x2]", n_live_points=50
)

analysis = ac.AnalysisCIImaging(dataset_list=ci_imaging_list, clocker=clocker)

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
