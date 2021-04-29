"""
Modeling: Charge Injection Uniform
==================================

In this script, we will fit charge injection imaging to calibrate CTI, where:

 - The CTI model consists of two parallel `TrapInstantCapture` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.
 - The `ImagingCI` is simulated with uniform charge injection lines and no cosmic rays.
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

Load the charge injection dataset 'imaging_ci/uniform/parallel_x2' 'from .fits files, which is the dataset we will
use to perform CTI modeling.
"""
dataset_name = "parallel_x2"
dataset_path = path.join("dataset", "imaging_ci", "uniform", dataset_name)

"""
The locations of the serial prescan and serial overscan on the image, which is used to visualize the `ImagingCI` during the 
model-fit.
"""
layout_list = ac.Scans(
    parallel_overscan=ac.Region2D((1980, 2000, 5, 95)),
    serial_prescan=ac.Region2D((0, 2000, 0, 5)),
    serial_overscan=ac.Region2D((0, 1980, 95, 100)),
)

"""
Specify the charge injection regions on the CCD, which in this case is 3 equally spaced rectangular blocks.
"""
regions_ci = [
    (0, 200, layout_list.serial_prescan[3], layout_list.serial_overscan[2]),
    (400, 600, layout_list.serial_prescan[3], layout_list.serial_overscan[2]),
    (800, 1000, layout_list.serial_prescan[3], layout_list.serial_overscan[2]),
    (1200, 1400, layout_list.serial_prescan[3], layout_list.serial_overscan[2]),
    (1600, 1800, layout_list.serial_prescan[3], layout_list.serial_overscan[2]),
]


"""
We require the normalization of every charge injection image, as the names of the files are tagged with the charge
injection normalization level.
"""
normalization_list = [100, 5000, 25000, 84700]

"""
Use the charge injection normalization_list and regions to create `PatternCIUniform` of every image we'll fit. The
`PatternCI` is used for visualizing the model-fit.
"""
pattern_cis = [
    ac.ci.PatternCIUniform(normalization=normalization, regions=regions_ci)
    for normalization in normalization_list
]

"""
We can now load every image, noise-map and pre-CTI charge injection image as instances of the `ImagingCI` object.
"""
imaging_ci_list = [
    ac.ci.ImagingCI.from_fits(
        image_path=path.join(dataset_path, f"image_{pattern.normalization}.fits"),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{pattern.normalization}.fits"
        ),
        pre_cti_image_path=path.join(
            dataset_path, f"pre_cti_image_{pattern.normalization}.fits"
        ),
        pixel_scales=0.1,
        pattern_ci=pattern,
    )
    for pattern in pattern_cis
]

"""
Lets plot the first `ImagingCI`.
"""
imaging_ci_plotter = aplt.ImagingCIPlotter(imaging=imaging_ci_list[0])
imaging_ci_plotter.subplot_imaging_ci()

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

 `/autolens_workspace/output/imaging_ci/parallel[x2]`.
"""
search = af.MultiNest(
    path_prefix=path.join("imaging_ci", dataset_name),
    name="parallel[x2]",
    n_live_points=50,
)

"""
__Analysis__

The `AnalysisImagingCI` object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `ImagingCI`dataset.
"""
analysis = ac.AnalysisImagingCI(dataset_ci_list=[imaging_ci_list], clocker=clocker)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the folder `autocti_workspace/output/imaging_ci/parallel[x2]` for live outputs 
of the results of the fit, including on-the-fly visualization of the best fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which includes: 

 - The charge injection fit corresponding to the maximum log likelihood solution in parameter space.
"""
print(result.max_log_likelihood_instance)

fit_ci_plotter = aplt.FitImagingCIPlotter(fit=result.max_log_likelihood_fit)
fit_ci_plotter.subplot_fit_imaging()

"""
Checkout `autocti_workspace/notebooks/imaging_ci/modeling/results.py` for a full description of the result object.
"""
