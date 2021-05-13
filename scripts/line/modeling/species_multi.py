"""
Modeling: Charge Injection Uniform
==================================

In this script, we will fit a CTI line to calibrate a CTI model, where:

 - The CTI model consists of multiple parallel `TrapInstantCapture` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.
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

Load the cti line dataset 'line/species_x2' 'from .fits files, which is the dataset we will use to perform CTI modeling.
"""
dataset_name = "species_x2"
dataset_path = path.join("dataset", "line", dataset_name)

"""
The locations of the serial prescan and serial overscan on the image, which is used to visualize the cti-line during the model-fit
and customize aspects of the model-fit.
"""
layout_list = ac.Scans(
    parallel_overscan=ac.Region2D((1980, 2000, 5, 95)),
    serial_prescan=ac.Region2D((0, 2000, 0, 5)),
    serial_overscan=ac.Region2D((0, 1980, 95, 100)),
)

"""
Specify the charge regions on the cti line, corresponding to where a signal is contained that has its electrons 
captured and trailed by CTI.
"""
regions = [(10, 20, 0, 1)]

"""
We require the normalization of the charge in every cti line dataset, as the names of the files are tagged with this.
"""
normalization_list = [100, 5000, 25000, 84700]

"""
We use the regions and normalization_list above to create the `Pattern` of every cti-line we fit. This pattern is used for
visualizing the model-fit.
"""
patterns = [
    ac.ci.PatternCIUniform(normalization=normalization, regions=regions)
    for normalization in normalization_list
]

"""
We now load every cti-line dataset, including a noise-map and pre-CTI line containing the data before read-out and
therefore without CTI. This uses a`LineDataset` object.
"""
imaging_list = [
    ac.ci.ImagingCI.from_fits(
        image_path=path.join(dataset_path, f"image_{pattern.normalization}.fits"),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{pattern.normalization}.fits"
        ),
        pre_cti_path=path.join(dataset_path, f"pre_cti_{pattern.normalization}.fits"),
        pixel_scales=0.1,
        pattern=pattern,
    )
    for pattern in patterns
]

"""
Lets plot the first `LineDataset`.
"""
imaging_plotter = aplt.ImagingCIPlotter(imaging=imaging_list[0])
imaging_plotter.subplot_imaging()

"""
__Clocking__

The `Clocker` models the CCD read-out, including CTI. 
"""
clocker = ac.Clocker(parallel_express=2)

"""
__Model__

We now compose our CTI model, which represents the trap species and CCD volume filling behaviour used to fit the cti 
line. In this example we fit a CTI model with:

 - Two parallel `TrapInstantCapture`'s which capture electrons during clocking instantly in the parallel direction
 [4 parameters].
 
 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.
"""
parallel_trap_0 = af.PriorModel(ac.TrapInstantCapture)
parallel_trap_1 = af.PriorModel(ac.TrapInstantCapture)
parallel_traps = [parallel_trap_0, parallel_trap_1]
parallel_ccd = af.PriorModel(ac.CCDPhase)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 84700.0

model = af.Collection(cti=af.Model(
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

 `/autocti_workspace/output/line/parallel[x2]`.
"""
search = af.DynestyStatic(
    path_prefix=path.join("line", dataset_name), name="parallel[x2]", nlive=50
)

"""
__Analysis__

The `AnalysisLineDataset` object defines the `log_likelihood_function` used by the non-linear search to fit the model 
to the `LineDataset`dataset.
"""
analysis = ac.AnalysisImagingCI(dataset_ci_list=[imaging_list], clocker=clocker)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the folder `autocti_workspace/output/line/parallel[x2]` for live outputs of the results of the fit, 
including on-the-fly visualization of the best fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which includes: 

 - The fit corresponding to the maximum log likelihood solution in parameter space.
"""
print(result.max_log_likelihood_instance)

fit_plotter = aplt.FitImagingCIPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit_imaging()

"""
Checkout `autocti_workspace/notebooks/line/modeling/results.py` for a full description of the result object.
"""
