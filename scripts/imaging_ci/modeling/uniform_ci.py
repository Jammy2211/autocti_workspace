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

The paths pointing to the dataset we will use for cti modeling.
"""
dataset_name = "parallel_x2"
dataset_path = path.join("dataset", "imaging_ci", "uniform", dataset_name)

"""
__Layout__

The 2D shape of the images.
"""
shape_native = (2000, 100)

"""
The locations (using NumPy array indexes) of the parallel overscan, serial prescan and serial overscan on the image.
"""
parallel_overscan = ac.Region2D((1980, 2000, 5, 95))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, 95, 100))

"""
The charge injection regions on the CCD, which in this case is 5 equally spaced rectangular blocks.
"""
regions_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

"""
The normalization of every charge injection image.
"""
normalization_list = [100, 5000, 25000, 84700]

"""
Create the layout of the charge injection pattern for every charge injection normalization.
"""
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

"""
We can now load every image, noise-map and pre-CTI charge injection image as instances of the `ImagingCI` object.
"""
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
parallel_ccd = af.PriorModel(ac.CCDPhase)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 84700.0

model = af.Collection(
    cti=af.Model(
        ac.CTI,
        parallel_traps=[parallel_trap_0, parallel_trap_1],
        parallel_ccd=parallel_ccd,
    )
)

"""
__Search__

The CTI model is fitted to the data using a `NonLinearSearch`. In this example, we use the
nested sampling algorithm Dynesty (https://dynesty.readthedocs.io/en/latest/).

The script 'autocti_workspace/scripts/imaging_ci/modeling/customize/non_linear_searches.py' gives a description of the 
non-linear searches that can be used with **PyAutoCTI**. If you do not know what a `NonLinearSearch` is or how it 
operates, checkout chapter 2 of the HowToCTI lecture series.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autocti_workspace/output/imaging_ci/parallel[x2]`.
"""
search = af.DynestyStatic(
    path_prefix=path.join("imaging_ci", dataset_name), name="parallel[x2]", nlive=50
)

"""
__Analysis__

The `AnalysisImagingCI` object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `ImagingCI`dataset.

To reduce run-times, we trim the `ImagingCI` data from the high resolution data (e.g. 2000 columns) to just 50 columns 
to speed up the model-fit at the expense of inferring larger errors on the CTI model.
"""
imaging_ci_trimmed_list = [
    imaging_ci.apply_settings(settings=ac.ci.SettingsImagingCI(parallel_columns=(0, 1)))
    for imaging_ci in imaging_ci_list
]

analysis = ac.AnalysisImagingCI(
    dataset_ci_list=imaging_ci_trimmed_list, clocker=clocker
)

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
