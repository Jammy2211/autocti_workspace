"""
Results: Start Here
===================

This script is the starting point for investigating the results of modeling and it provides
an overview of the modeling API.

After reading this script, the `examples` folder provides more detailed examples for analysing the different aspects of
performing modeling results outlined here.

__Model__

We begin by fitting a quick model to a simple dataset, which we will use to illustrate the modeling
results API.

If you are not familiar with the modeling API and process, checkout the `autocti_workspace/imaging_ci/modeling`
folder for examples.
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
__Info__

As seen throughout the workspace, the `info` attribute shows the result in a readable format.
"""
print(result_list.info)

"""
__Samples__

The result's `Samples` object contains the complete set of non-linear search nautilus samples, where each sample 
corresponds to a set of model parameters that were evaluated and accepted. 

The examples script `autocti_workspace/*/dataset_1d/results/examples/samples.py` provides a detailed description of 
this object, including:

 - Extracting the maximum likelihood model.
 - Using marginalized PDFs to estimate errors on the model parameters.
 - Deriving errors on derived quantities, such as the Einstein radius.

Below, is an example of how to use the `Samples` object to estimate the mass model parameters which are 
the median of the probability distribution function and its errors at 3 sigma confidence intervals.
"""
samples = result_list.samples

median_pdf_instance = samples.median_pdf()

print("Median PDF Model Instances: \n")
print(median_pdf_instance.cti.trap_list[0])
print()

ue3_instance = samples.values_at_upper_sigma(sigma=3.0)
le3_instance = samples.values_at_lower_sigma(sigma=3.0)

print("Errors Instances: \n")
print(ue3_instance.cti.trap_list[0], "\n")
print(le3_instance.cti.trap_list[0], "\n")

"""
__Fits__

The result's maximum likelihood `FitDataset1D` object contains everything necessary to inspect the model fit to the 
data.

The examples script `autocti_workspace/*/dataset_1d/results/examples/fits.py` provides a detailed description of this 
object, including:

 - How to inspect the residuals, chi-squared, likelihood and other quantities.
 - Outputting resulting images (e.g. the CTI corrected data) to hard-disk.
 - Refitting the data with other models from the `Samples` object, to investigate how sensitive the fit is to
   different models.

Below, is an example of how to use the `FitDataset1D` object to output the CTI reconstruction to print the 
chi-squared and log likelihood values.
"""
fit = result_list[0].max_log_likelihood_fit

print(fit.chi_squared)
print(fit.log_likelihood)

"""
__CTI__

The CTI model can be inspected to quantify whether certain science requirements are met by the quality of model.

The examples script `autocti_workspace/*/dataset_1d/results/examples/cti.py` provides a detailed description 
of this, including:

 - How to translate the inferred CTI model errors to the spurious elliticity of galaxy weak lensing measurements
   due to CTI. 
 
"""
