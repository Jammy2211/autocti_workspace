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
print(fit.model_image.slim)
print(fit.model_image.native)

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
__Galaxy Quantities__

The `FitImaging` object has specific quantities which break down each image of each galaxy:

 - `model_images_of_galaxies_list`: Model-images of each individual galaxy, which in this example is a model image of 
 the two galaxies in the model. Both images are convolved with the imaging's PSF.
 
 - `subtracted_images_of_galaxies_list`: Subtracted images of each individual galaxy, which are the data's image with
 all other galaxy's model-images subtracted. For example, the first subtracted image has the second galaxy's model image
 subtracted and therefore is of only the right galaxy's emission.
"""
print(fit.model_images_of_galaxies_list[0].slim)
print(fit.model_images_of_galaxies_list[1].slim)

print(fit.subtracted_images_of_galaxies_list[0].slim)
print(fit.subtracted_images_of_galaxies_list[1].slim)

"""
__Unmasked Quantities__

All of the quantities above are computed using the mask which was used to fit the data.

The `FitImaging` can also compute the unmasked blurred image of each plane.
"""
print(fit.unmasked_blurred_image.native)
print(fit.unmasked_blurred_image_of_galaxies_list[0].native)
print(fit.unmasked_blurred_image_of_galaxies_list[1].native)

"""
__Mask__

We can use the `Mask2D` object to mask regions of one of the fit's maps and estimate quantities of it.

Below, we estimate the average absolute normalized residuals within a 1.0" circular mask, which would inform us of
how accurate the model fit is in the central regions of the data.
"""
mask = ag.Mask2D.circular(
    shape_native=fit.dataset.shape_native,
    pixel_scales=fit.dataset.pixel_scales,
    radius=1.0,
)

normalized_residuals = fit.normalized_residual_map.apply_mask(mask=mask)

print(np.mean(np.abs(normalized_residuals.slim)))

"""
__Pixel Counting__

An alternative way to quantify residuals like the galaxy light residuals is pixel counting. For example, we could sum
up the number of pixels whose chi-squared values are above 10 which indicates a poor fit to the data.

Whereas computing the mean above the average level of residuals, pixel counting informs us how spatially large the
residuals extend. 
"""
mask = ag.Mask2D.circular(
    shape_native=fit.dataset.shape_native,
    pixel_scales=fit.dataset.pixel_scales,
    radius=1.0,
)

chi_squared_map = fit.chi_squared_map.apply_mask(mask=mask)

print(np.sum(chi_squared_map > 10.0))

"""
__Outputting Results__

You may wish to output certain results to .fits files for later inspection. 

For example, one could output the galaxy subtracted image of the second galaxy to a .fits file such that
we could fit this image again with an independent modeling script.
"""
galaxy_subtracted_data = fit.subtracted_images_of_galaxies_list[1]
galaxy_subtracted_data.output_to_fits(
    file_path=path.join(dataset_path, "galaxy_subtracted_image.fits"), overwrite=True
)

"""
__Refitting__

Using the API introduced in the first tutorial, we can also refit the data locally. 

This allows us to inspect how the fit changes for models with similar log likelihoods. Below, we refit and plot
the fit of the 100th last accepted model by nautilus.
"""
samples = result.samples

instance = samples.from_sample_index(sample_index=-10)

plane = ag.Plane(galaxies=instance.galaxies)

fit = ag.FitImaging(dataset=dataset, plane=plane)

fit_plotter = aplt.FitDataset1DPlotter(fit=fit)
fit_plotter.subplot_fit()


"""
__Wrap Up__

In this tutorial, we saw how to inspect the quality of a model fit using the fit imaging object.

If you are modeling galaxies using interferometer data we cover the corresponding fit object in tutorial 6.
"""
