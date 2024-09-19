"""
Results: Fits
=============

This tutorial inspects the model's fit to the data using the `FitDataset1D` object inferred by the non-linear search, 
for example visualizing and interpreting its results.

This includes inspecting the residuals, chi-squared and other goodness-of-fit quantities.

__Plot Module__

This example uses the **PyAutoCTI** plot module to plot the results, including `Plotter` objects that make
the figures and `MatPlot` objects that wrap matplotlib to customize the figures.

The visualization API is straightforward but is explained in the `autocti_workspace/*/plot` package in full. This 
includes detailed guides on how to customize every aspect of the figures, which can easily be combined with the
code outlined in this tutorial.

__Units__

In this example, all quantities are **PyAutoCTI**'s internal unit coordinates, with spatial coordinates in
arc seconds, luminosities in electrons per second and mass quantities (e.g. convergence) are dimensionless.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autofit as af
import autocti as ac
import autocti.plot as aplt

"""
__Model Fit__

The code below performs a model-fit using nautilus. 

You should be familiar with modeling already, if not read the `modeling/start_here.py` script before reading this one!
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "dataset_1d", dataset_name)

shape_native = (200,)

prescan = ac.Region1D(region=(0, 10))
overscan = ac.Region1D(region=(190, 200))

region_list = [(10, 20)]

norm_list = [100, 5000, 25000, 200000]

total_datasets = len(norm_list)

layout_list = [
    ac.Layout1D(
        shape_1d=shape_native,
        region_list=region_list,
        prescan=prescan,
        overscan=overscan,
    )
    for i in range(total_datasets)
]

dataset_list = [
    ac.Dataset1D.from_fits(
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

mask = ac.Mask1D.all_false(
    shape_slim=dataset_list[0].shape_slim,
    pixel_scales=dataset_list[0].pixel_scales,
)

mask = ac.Mask1D.masked_fpr_and_eper_from(
    mask=mask,
    layout=dataset_list[0].layout,
    settings=ac.SettingsMask1D(fpr_pixels=(0, 10)),
    pixel_scales=dataset_list[0].pixel_scales,
)

dataset_list = [dataset.apply_mask(mask=mask) for dataset in dataset_list]

clocker = ac.Clocker1D(express=5)

trap_0 = af.Model(ac.TrapInstantCapture)
trap_1 = af.Model(ac.TrapInstantCapture)

trap_0.add_assertion(trap_0.release_timescale < trap_1.release_timescale)

trap_list = [trap_0, trap_1]

ccd = af.Model(ac.CCDPhase)
ccd.well_notch_depth = 0.0
ccd.full_well_depth = 200000.0

model = af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))

search = af.Nautilus(
    path_prefix=path.join("dataset_1d", dataset_name), name="species[x2]", n_live=100
)

analysis_list = [
    ac.AnalysisDataset1D(dataset=dataset, clocker=clocker) for dataset in dataset_list
]

analysis = sum(analysis_list)

result_list = search.fit(model=model, analysis=analysis)

"""
__Max Likelihood Fit__

As seen elsewhere in the workspace, the result contains a `max_log_likelihood_fit` which we can visualize.
"""
fit = result_list[0].max_log_likelihood_fit

fit_plotter = aplt.FitDataset1DPlotter(fit=fit)
fit_plotter.subplot_fit()

"""
__Fit Quantities__

The maximum log likelihood fit contains many 1D and 2D arrays showing the fit.

These use the `slim` and `native` API discussed in the previous results tutorial.

There is a `model_image`, which is the image of the plane we inspected in the previous tutorial blurred with the 
imaging data's PSF. 

This is the image that is fitted to the data in order to compute the log likelihood and therefore quantify the 
goodness-of-fit.
"""
print(fit.model_data.slim)
print(fit.model_data.native)

"""
There are numerous ndarrays showing the goodness of fit: 

 - `residual_map`: Residuals = (Data - Model_Data).
 - `normalized_residual_map`: Normalized_Residual = (Data - Model_Data) / Noise
 - `chi_squared_map`: Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
"""
print(fit.residual_map.slim)
print(fit.residual_map.native)

print(fit.normalized_residual_map.slim)
print(fit.normalized_residual_map.native)

print(fit.chi_squared_map.slim)
print(fit.chi_squared_map.native)

"""
__Figures of Merit__

There are single valued floats which quantify the goodness of fit:

 - `chi_squared`: The sum of the `chi_squared_map`.
 - `noise_normalization`: The normalizing noise term in the likelihood function 
    where [Noise_Term] = sum(log(2*pi*[Noise]**2.0)).
 - `log_likelihood`: The log likelihood value of the fit where [LogLikelihood] = -0.5*[Chi_Squared_Term + Noise_Term].
"""
print(fit.chi_squared)
print(fit.noise_normalization)
print(fit.log_likelihood)


"""
__Wrap Up__

In this tutorial, we saw how to inspect the quality of a model fit using the fit imaging object.

If you are modeling galaxies using interferometer data we cover the corresponding fit object in tutorial 6.
"""
