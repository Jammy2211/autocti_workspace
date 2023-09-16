"""
Modeling: Charge Injection Uniform
==================================

__Model__

In this script, we will fit charge injection imaging to calibrate CTI, where:

 - The CTI model consists of two parallel `TrapInstantCapture` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.
 - The `ImagingCI` is simulated with uniform charge injection lines and no cosmic rays.

__Start Here Notebook__

If any code in this script is unclear, refer to the modeling `start_here.ipynb` notebook for more detailed comments.
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
dataset_name = "non_uniform"
dataset_path = path.join("dataset", "imaging_ci", dataset_name)

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
region_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

"""
The normalization of every charge injection image, which determines how many images are simulated.
"""
norm_list = [100, 5000, 25000, 200000]

"""
The total number of charge injection images that are fitted.
"""
total_datasets = len(norm_list)

"""
Create the layout of the charge injection pattern for every charge injection normalization.
"""
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

"""
We can now load every image, noise-map and pre-CTI charge injection image as instances of the `ImagingCI` object.
"""
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

"""
Lets plot the first `ImagingCI`.
"""
dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset_list[0])
dataset_plotter.subplot_dataset()

"""
__Clocking__

The `Clocker` models the CCD read-out, including CTI. 

For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""
clocker = ac.Clocker2D(
    parallel_express=5, parallel_roe=ac.ROEChargeInjection(), parallel_fast_mode=True
)

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
__Settings__

To reduce run-times, we trim the `ImagingCI` data from the high resolution data (e.g. 100 columns) to just 5 columns 
to speed up the model-fit at the expense of inferring larger errors on the CTI model.
"""
imaging_ci_trim_list = [
    dataset.apply_settings(settings=ac.SettingsImagingCI(parallel_pixels=(0, 5)))
    for dataset in dataset_list
]


"""
__Model__

We now compose our CTI model, which represents the trap species and CCD volume filling behaviour used to fit the charge 
injection data. In this example we fit a CTI model with:

 - Two parallel `TrapInstantCapture`'s which capture electrons during clocking instantly in the parallel direction
 [4 parameters].
 
 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.
"""
parallel_trap_list = [af.Model(ac.TrapInstantCapture), af.Model(ac.TrapInstantCapture)]
parallel_ccd = af.Model(ac.CCDPhase)
parallel_ccd.well_notch_depth = 0.0
parallel_ccd.full_well_depth = 200000.0

model = af.Collection(
    cti=af.Model(
        ac.CTI2D, parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd
    )
)

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (https://nautilus.readthedocs.io/en/latest/).

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autocti_workspace/output/imaging_ci/parallel[x2]`.
"""
search = af.Nautilus(
    path_prefix=path.join("imaging_ci", dataset_name),
    name="parallel[x1]",
    n_live=100,
)

"""
__Analysis__

The `AnalysisImagingCI` object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `ImagingCI`dataset.
"""
analysis_list = [
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
analysis = sum(analysis_list)

"""
We can parallelize the likelihood function of these analysis classes, whereby each evaluation will be performed on a 
different CPU.
"""
analysis.n_cores = 1

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the folder `autocti_workspace/output/imaging_ci/parallel[x2]` for live outputs 
of the results of the fit, including on-the-fly visualization of the best fit model!
"""
result_list = search.fit(model=model, analysis=analysis)

"""
__Result__

The result object returned by the fit provides information on the results of the non-linear search. 

The `info` attribute shows the result in a readable format.
"""
print(result_list.info)

"""
The result object also contains the fit corresponding to the maximum log likelihood solution in parameter space,
which can be used to visualizing the results.
"""
print(result_list[0].max_log_likelihood_instance.cti.parallel_trap_list[0].density)
print(result_list[0].max_log_likelihood_instance.cti.parallel_ccd.well_fill_power)

for result in result_list:
    fit_plotter = aplt.FitImagingCIPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()
"""
Checkout `autocti_workspace/*/imaging_ci/modeling/results.py` for a full description of the result object.
"""
