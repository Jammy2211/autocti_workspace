"""
Customize: Priors
=================

This example demonstrates how to customize the priors of a model-fit, for example if you are modeling a dataset where
certain parameters are known beforehand.

**Benefits:**: This will result in a faster more robust model-fit.

__Disadvantages__

The priors on your model determine the errors you infer. Overly tight priors may lead to over
confidence in the inferred parameters.

The `autocti_workspace/*/imaging/modeling/customize/start_point.ipynb` shows an alternative API, which
customizes where the non-linear search starts its search of parameter space.

This cannot be used for a nested sampling method like `nautilus` (whose parameter space search is dictated by priors)
but can be used for the maximum likelihood estimator / MCMC methods PyAutoGalaxy supports.

The benefit of the starting point API is that one can tell the non-linear search where to look in parameter space
(ensuring a fast and robust fit) but retain uninformative priors which will not lead to over-confident errors.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
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

Load the CTI dataset 'dataset_1d/simple' 'from .fits files, which is the dataset we will use to perform CTI modeling.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "dataset_1d", dataset_name)

"""
__Shape__

The 1D shape of each 1D dataset.
"""
shape_native = (200,)

"""
__Regions__

The locations of the prescan and overscan on the 1D data, which is used to visualize the 1D CTI dataset during the 
model-fit and customize aspects of the model-fit.
"""
prescan = ac.Region1D((0, 10))
overscan = ac.Region1D((190, 200))

"""
Specify the charge regions on the 1D CTI Dataset, corresponding to where a signal is contained that has its electrons 
captured and trailed by CTI.
"""
region_list = [(10, 20)]

"""
__Normalizations__

We require the normalization of the charge in every CTI dataset, as the names of the files are tagged with this.
"""
norm_list = [100, 5000, 25000, 200000]

"""
__Layout__

We use the regions and norm_list above to create the `Layout1D` of every 1D CTI dataset we fit. This is used 
for visualizing the model-fit.
"""
layout_list = [
    ac.Layout1D(
        shape_1d=shape_native,
        region_list=region_list,
        prescan=prescan,
        overscan=overscan,
    )
    for norm in norm_list
]


"""
__Dataset__

We now load every cti-dataset, including a noise-map and pre-cti data containing the data before read-out and
therefore without CTI.
"""
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

"""
__Mask__

We now mask every 1D dataset, removing the FPR of each dataset so we use only the EPER to calibrate the CTI model.
"""
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

"""
Lets plot the first dataset.
"""
dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0])
dataset_plotter.subplot_dataset()

"""
__Clocking__

The `Clocker` models the CCD read-out, including CTI. 
"""
clocker = ac.Clocker1D(express=5)

"""
__Model__

We now compose our CTI model, which represents the trap species and CCD volume filling behaviour used to fit the CTI 
1D data. In this example we fit a CTI model with:

 - Two `TrapInstantCapture`'s which capture electrons during clocking instantly in the parallel direction
 [4 parameters].
 
 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.

__Prior Customization__
 
We customize the parameter of every prior to values near the true values, using the following priors:

- UniformPrior: The values of a parameter are randomly drawn between a `lower_limit` and `upper_limit`. For example,
the `well_fill_power` of the `CCD` object assumes a uniform prior between 0.0 and 1.0.

- LogUniformPrior: Like a `UniformPrior` this randomly draws values between a `limit_limit` and `upper_limit`, but the
values are drawn from a distribution with base 10. This is not currently used for any model parameters in **PyAutoCTI**.

- GaussianPrior: The values of a parameter are randomly drawn from a Gaussian distribution with a `mean` and width
 `sigma`. In this example we use `GaussianPrior`'s on the trap densities, as if we have some previous knowledge of
 their expected value.
"""
trap_0 = af.Model(ac.TrapInstantCapture)
trap_1 = af.Model(ac.TrapInstantCapture)

trap_0.density = af.GaussianPrior(mean=0.13, sigma=0.1)
trap_0.release_timescale = af.GaussianPrior(mean=1.25, sigma=0.5)

trap_1.density = af.GaussianPrior(mean=0.25, sigma=0.1)
trap_1.release_timescale = af.GaussianPrior(mean=4.4, sigma=0.5)

trap_list = [trap_0, trap_1]

ccd = af.Model(ac.CCDPhase)

ccd.well_fill_power = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

ccd.well_notch_depth = 0.0
ccd.full_well_depth = 200000.0

"""
Assertions must be added after all priors are customized, you cannot change the priors after adding an assertion.
"""
trap_0.add_assertion(trap_0.release_timescale < trap_1.release_timescale)

model = af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))

"""
The `info` attribute shows the model in a readable format, including the customized priors above.
"""
print(model.info)

"""
__Alternative API__

The priors can also be customized after the `CTI1D` model object is created instead.
"""
trap_0 = af.Model(ac.TrapInstantCapture)
trap_1 = af.Model(ac.TrapInstantCapture)

trap_list = [trap_0, trap_1]

ccd = af.Model(ac.CCDPhase)

model = af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))

model.cti.trap_list[0].density = af.GaussianPrior(mean=0.13, sigma=0.1)
model.cti.trap_list[0].release_timescale = af.GaussianPrior(mean=1.25, sigma=0.5)

model.cti.trap_list[1].density = af.GaussianPrior(mean=0.25, sigma=0.1)
model.cti.trap_list[1].release_timescale = af.GaussianPrior(mean=4.4, sigma=0.5)

model.cti.ccd.well_fill_power = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

model.cti.ccd.well_notch_depth = 0.0
model.cti.ccd.full_well_depth = 200000.0

model.cti.trap_list[0].add_assertion(
    model.cti.trap_list[0].release_timescale < model.cti.trap_list[1].release_timescale
)

"""
The `info` attribute shows the model in a readable format, including the customized priors above.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Nautilus (https://nautilus.readthedocs.io/en/latest/).

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autocti_workspace/output/dataset_1d/species[x2]`.
"""
search = af.Nautilus(
    path_prefix=path.join("dataset_1d", dataset_name), name="species[x2]", n_live=100
)

"""
__Analysis__

The `AnalysisDataset1D` object defines the `log_likelihood_function` used by the non-linear search to fit the model 
to the `Dataset1D`dataset.
"""
analysis_list = [
    ac.AnalysisDataset1D(dataset=dataset, clocker=clocker) for dataset in dataset_list
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

Checkout the folder `autocti_workspace/output/dataset_1d/species[x2]` for live outputs of the results of the fit, 
including on-the-fly visualization of the best fit model!
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
print(result_list[0].max_log_likelihood_instance.cti.trap_list[0].density)
print(result_list[0].max_log_likelihood_instance.cti.ccd.well_fill_power)

for result in result_list:
    fit_plotter = aplt.FitDataset1DPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

"""
Checkout `autocti_workspace/*/dataset_1d/modeling/results.py` for a full description of the result object.
"""
