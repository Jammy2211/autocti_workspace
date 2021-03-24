"""
Modeling: Charge Injection Uniform
==================================

In this script, we will fit charge injection imaging to calibrate CTI, where:

 - The CTI model consists of two parallel `TrapInstantCapture` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.
 - The `CIImaging` is simulated with uniform charge injection lines and no cosmic rays.
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

Load the charge injection dataset 'ci_imaging/uniform/parallel_x2' 'from .fits files, which is the dataset we will
use to perform CTI modeling.
"""
dataset_name = "parallel_x2"
dataset_path = path.join("dataset", "ci_imaging", "uniform", dataset_name)

"""
The locations of the prescans and overscans on the image, which is used to visualize the `CIImaging` during the 
model-fit.
"""
scans = ac.Scans(
    parallel_overscan=ac.Region2D((1980, 2000, 5, 95)),
    serial_prescan=ac.Region2D((0, 2000, 0, 5)),
    serial_overscan=ac.Region2D((0, 1980, 95, 100)),
)

"""
Specify the charge injection regions on the CCD, which in this case is 3 equally spaced rectangular blocks.
"""
ci_regions = [
    (0, 200, scans.serial_prescan[3], scans.serial_overscan[2]),
    (400, 600, scans.serial_prescan[3], scans.serial_overscan[2]),
    (800, 1000, scans.serial_prescan[3], scans.serial_overscan[2]),
    (1200, 1400, scans.serial_prescan[3], scans.serial_overscan[2]),
    (1600, 1800, scans.serial_prescan[3], scans.serial_overscan[2]),
]


"""
We require the normalization of every charge injection image, as the names of the files are tagged with the charge
injection normalization level.
"""
normalizations = [100, 5000, 25000, 84700]

"""
Use the charge injection normalizations and regions to create `CIPatternUniform` of every image we'll fit. The
`CIPattern` is used for visualizing the model-fit.
"""
ci_patterns = [
    ac.ci.CIPatternUniform(normalization=normalization, regions=ci_regions)
    for normalization in normalizations
]

"""
We can now load every image, noise-map and pre-CTI charge injection image as instances of the `CIImaging` object.
"""
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
    )
    for pattern in ci_patterns
]

"""
Lets plot the first `CIImaging`.
"""
ci_imaging_plotter = aplt.CIImagingPlotter(imaging=ci_imaging_list[0])
ci_imaging_plotter.subplot_ci_imaging()

"""
__Clocking__

The `Clocker` models the CCD read-out, including CTI. 

For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""
clocker = ac.Clocker(parallel_express=2, parallel_charge_injection_mode=True)

"""
__Model__

We now compose our CTI model, which represents the trap species and CCD volume filling behaviour used to fit the charge 
injection data. In this example we fit a CTI model with:

 - Two parallel `TrapInstantCapture`'s which capture electrons during clocking instantly in the parallel direction
 [4 parameters].
 
 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.
"""
parallel_trap_0 = af.PriorModel(ac.TrapInstantCapture)
parallel_trap_1 = af.PriorModel(ac.TrapInstantCapture)
parallel_traps = [parallel_trap_0, parallel_trap_1]
parallel_ccd = af.PriorModel(ac.CCD)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 84700

model = af.Model(
    ac.CTI, parallel_traps=[parallel_trap_0, parallel_trap_1], parallel_ccd=parallel_ccd
)

"""
__Search__

The CTI model is fitted to the data using a `NonLinearSearch`. In this example, we use the
nested sampling algorithm Dynesty (https://dynesty.readthedocs.io/en/latest/).

The script 'autocti_workspace/examples/modeling/customize/non_linear_searches.py' gives a description of the types of
non-linear searches that can be used with **PyAutoCTI**. If you do not know what a `NonLinearSearch` is or how it 
operates, checkout chapter 2 of the HowToCTI lecture series.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/ci_imaging/parallel[x2]`.
"""
search = af.MultiNest(
    path_prefix=path.join("ci_imaging", dataset_name),
    name="parallel[x2]",
    n_live_points=50,
)

"""
__Analysis__

The `AnalysisCIImaging` object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `MaskedCIImaging`dataset.
"""
analysis = ac.AnalysisCIImaging(dataset_list=[ci_imaging_list], clocker=clocker)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the folder `autocti_workspace/output/ci_imaging/parallel[x2]` for live outputs 
of the results of the fit, including on-the-fly visualization of the best fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which includes: 

 - The charge injection fit corresponding to the maximum log likelihood solution in parameter space.
"""
print(result.max_log_likelihood_instance)

ci_fit_plotter = aplt.CIFitPlotter(fit=result.max_log_likelihood_fit)
ci_fit_plotter.subplot_fit_imaging()

"""
Checkout `autocti_workspace/notebooks/ci_imaging/modeling/results.py` for a full description of the result object.
"""
