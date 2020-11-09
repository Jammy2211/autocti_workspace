"""
__Example: Modeling__

To fit a CTI model to a dataset, we must perform CTI modeling, which uses a `NonLinearSearch` to fit many
different CTI models to the dataset.

Model-fitting is handled by our project **PyAutoFit**, a probabilistic programming language for non-linear model
fitting. The setting up on configuration files is performed by our project **PyAutoConf**. We'll need to import
both to perform the model-fit.
"""

"""
In this example script, we will fit charge injection imaging which has been subjected to CTI, where:

 - The CTI model consists of two parallel `Trap` species.
 - The `CCD` volume fill parameterization is a simple form with just a `well_fill_power` parameter.
 - The `CIImaging` is simulated with uniform charge injection lines and no cosmic rays.
"""

"""
Load the charge injection dataset 'ci_imaging/uniform/parallel_x2' 'from .fits files, which is the dataset we will
use to perform CTI modeling.

This is the same dataset we fitted in the 'autocti/intro/fitting.py' example.
"""

import autofit as af
import autocti as ac
import autocti.plot as aplt

dataset_name = "parallel_x2"
dataset_path = f"dataset/ci_imaging/uniform/{dataset_name}"

"""The locations of the overscans on the image, which is used to visualize the `CIImaging` during the model-fit."""

scans = ac.Scans(
    parallel_overscan=ac.Region((1980, 2000, 5, 95)),
    serial_prescan=ac.Region((0, 2000, 0, 5)),
    serial_overscan=ac.Region((0, 1980, 95, 100)),
)

"""Specify the charge injection regions on the CCD, which in this case is 3 equally spaced rectangular blocks."""

ci_regions = [
    (0, 450, scans.serial_prescan[3], scans.serial_overscan[2]),
    (650, 1100, scans.serial_prescan[3], scans.serial_overscan[2]),
    (1300, 1750, scans.serial_prescan[3], scans.serial_overscan[2]),
]

"""
We require the normalization of every charge injection image, as the names of the files are tagged with the charge
injection normalization level.
"""

normalizations = [100.0, 5000.0, 25000.0, 84700.0]

"""Use the charge injection normalizations and regions to create `CIPatternUniform` of every image we'll fit. The
`CIPattern` is used for visualizing the model-fit."""

ci_patterns = [
    ac.ci.CIPatternUniform(normalization=normalization, regions=ci_regions)
    for normalization in normalizations
]

"""
We can now load every image, noise-map and pre-CTI charge injection image as instances of the `CIImaging` object.
"""

imagings = [
    ac.ci.CIImaging.from_fits(
        image_path=f"{dataset_path}/image_{int(pattern.normalization)}.fits",
        noise_map_path=f"{dataset_path}/noise_map_{int(pattern.normalization)}.fits",
        ci_pre_cti_path=f"{dataset_path}/ci_pre_cti_{int(pattern.normalization)}.fits",
        pixel_scales=0.1,
        ci_pattern=pattern,
        roe_corner=(1, 0),
    )
    for pattern in ci_patterns
]

"""Lets plot the first `CIImaging` with its `Mask2D`"""
aplt.Imaging.subplot_imaging(imaging=imagings[0])

"""
The `Clocker` models the CCD read-out, including CTI. 

For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""

clocker = ac.Clocker(parallel_express=2, parallel_charge_injection_mode=True)

"""
__Phase__

To perform lens modeling, we create a *PhaseCIImaging* object, which comprises:

   - The `Trap`'s and `CCD` models used to fit the data.
   - The `SettingsPhase` which customize how the model is fitted to the data.
   - The `NonLinearSearch` used to sample parameter space.

Once we have create the phase, we 'run' it by passing it the data and mask.
"""

"""
__Model__

We compose our lens model using `Trap` and `CCD` objects, which are what add CTI to our images during clocking and 
read out. In this example our CTI model is:

 - Two parallel `TrapInstantCapture`'s which capture electrons during clocking instantly in the parallel direction.
 - A simple `CCD` volume beta parametrization.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.
"""

parallel_trap_0 = af.PriorModel(ac.TrapInstantCapture)
parallel_trap_1 = af.PriorModel(ac.TrapInstantCapture)
parallel_traps = [parallel_trap_0, parallel_trap_1]
parallel_ccd = af.PriorModel(ac.CCD)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 84700

"""
__Settings__

Next, we specify the *SettingsPhaseCIImaging*, which describes how the model is fitted to the data in the 
log likelihood function. Below, we specify:

Different `SettingsPhase` are used in different example model scripts and a full description of all `SettingsPhase`
can be found in the example script 'autoccti_workspace/examples/model/customize/settings.py' and the following 
link -> <link>
"""

settings = ac.SettingsPhaseCIImaging()

"""
__Search__

The lens model is fitted to the data using a `NonLinearSearch`, which we specify below. In this example, we use the
nested sampling algorithm Dynesty (https://dynesty.readthedocs.io/en/latest/), with:

 - 50 live points.

The script 'autocti_workspace/examples/model/customize/non_linear_searches.py' gives a description of the types of
non-linear searches that can be used with **PyAutoCTI**. If you do not know what a `NonLinearSearch` is or how it 
operates, I recommend you complete chapters 1 and 2 of the HowToCTI lecture series.
"""

search = af.MultiNest(
    path_prefix=f"examples/{dataset_name}", name="phase_parallel_x2", n_live_points=50
)

"""
__Phase__

We can now combine the model, settings and `NonLinearSearch` above to create and run a phase, fitting our data with
the lens model.

The name and folders inputs below specify the path where results are stored in the output folder:  

 '/autolens_workspace/output/examples/beginner/mass_sie__source_bulge/phase__mass_sie__source_bulge'.
"""

phase = ac.PhaseCIImaging(
    search=search,
    parallel_traps=parallel_traps,
    parallel_ccd=parallel_ccd,
    settings=settings,
)

"""
We can now begin the fit by passing the dataset and mask to the phase, which will use the `NonLinearSearch` to fit
the model to the data. 

The fit outputs visualization on-the-fly, so checkout the path 
'/path/to/autolens_workspace/output/examples/phase__mass_sie__source_bulge' to see how your fit is doing!
"""

result = phase.run(datasets=imagings, clocker=clocker)

"""
The phase above returned a result, which, for example, includes the lens model corresponding to the maximum
log likelihood solution in parameter space.
"""

print(result.max_log_likelihood_instance)

"""
It also contains instances of the maximum log likelihood Tracer and FitImaging, which can be used to visualize
the fit.
"""

aplt.CIFit.subplot_ci_fit(fit=result.max_log_likelihood_fit)

"""
Checkout '/path/to/autocti_workspace/examples/model/results.py' for a full description of the result object.
"""
