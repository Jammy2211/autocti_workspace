"""
Results: Samples
================

After fitting a CTI model a search returns a `result` variable. We have used this throughout the examples scripts to 
plot the maximum log likelihood fits.

This `Result` object contains a lot more information which the results tutorials illustrate.

This script describes the non-linear search samples of a model-fit, which for a `DynestyStatic` fit corresponds
to every accepted live point in parameter space. 

These are used to compute quantities like the maximum likelihood CTI model, the errors on the parameters and 
visualization showing the parameter degeneracies.

__Units__

In this example, all quantities are **PyAutoCTI**'s internal unit coordinates, with spatial coordinates in pixels and
pixel values in electrons.
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

The code below performs a model-fit using dynesty. 

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

"""
__Model Composition__

The code below composes the model fitted to the data (the API is described in the `modeling/start_here.py` example).

The way the model is composed below (e.g. that the model is called `cti` and includes a `trap_list` and `ccd`) should 
be noted, as it will be important when inspecting certain results later in this example.
"""
clocker = ac.Clocker1D(express=5)

trap_0 = af.Model(ac.TrapInstantCapture)
trap_1 = af.Model(ac.TrapInstantCapture)

trap_0.add_assertion(trap_0.release_timescale < trap_1.release_timescale)

trap_list = [trap_0, trap_1]

ccd = af.Model(ac.CCDPhase)
ccd.well_notch_depth = 0.0
ccd.full_well_depth = 200000.0

model = af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))

search = af.DynestyStatic(
    path_prefix=path.join("dataset_1d", dataset_name), name="species[x2]", nlive=50
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
__Plot__

We now have the `Result` object we will cover in this script. 

As a reminder, in the `modeling` scripts we use the `max_log_likelihood_instance` and `max_log_likelihood_fit` to print 
and plot the results of the fit.
"""
print(result_list[0].max_log_likelihood_instance.cti.trap_list[0].density)
print(result_list[0].max_log_likelihood_instance.cti.ccd.well_fill_power)

for result in result_list:
    fit_plotter = aplt.FitDataset1DPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

"""
The results tutorial `fits.py` expands `max_log_likelihood_fit`, showing how it can be used to inspect many aspects 
of a model.

__Samples__

This tutorial covers the result's `Samples` object, which contains all of the non-linear search samples. Each sample 
corresponds to a set of model parameters that were evaluated and accepted by our non linear search, in this example 
dynesty. 

This also includes their log likelihoods, which are used for computing additional information about the model-fit,
for example the error on every parameter. 

Our model-fit used the nested sampling algorithm Dynesty, so the `Samples` object returned is a `SamplesNest` object.
"""
samples = result.samples

print("Nest Samples: \n")
print(samples)

"""
__Parameters__

The `Samples` class contains all the parameter samples, which is a list of lists where:

 - The outer list is the size of the total number of samples.
 - The inner list is the size of the number of free parameters in the fit.
"""
print("All parameters of the very first sample")
print(samples.parameter_lists[0])
print("The fourth parameter of the tenth sample")
print(samples.parameter_lists[9][3])

"""
__Figures of Merit__

The `Samples` class contains the log likelihood, log prior, log posterior and weight_list of every sample, where:

 - The log likelihood is the value evaluated from the likelihood function (e.g. -0.5 * chi_squared + the noise 
   normalization).

 - The log prior encodes information on how the priors on the parameters maps the log likelihood value to the log
   posterior value.

 - The log posterior is log_likelihood + log_prior.

 - The weight gives information on how samples should be combined to estimate the posterior. The weight values 
   depend on the sampler used. For example for an MCMC search they will all be 1`s whereas for the nested sampling
   method used in this example they are weighted as a combination of the log likelihood value and prior.
"""
print("log(likelihood), log(prior), log(posterior) and weight of the tenth sample.")
print(samples.log_likelihood_list[9])
print(samples.log_prior_list[9])
print(samples.log_posterior_list[9])
print(samples.weight_list[9])

"""
__Instances__

The `Samples` contains many results which are returned as an instance of the model, using the Python class structure
of the model composition.

For example, we can return the model parameters corresponding to the maximum log likelihood sample.
"""
max_lh_instance = samples.max_log_likelihood()
print("Maximum Log Likelihood Model Instance: \n")
print(max_lh_instance, "\n")

"""
A model instance contains all the model components of our fit, for example the `CTI1D` object we specified during 
model composition with the name `cti`.
"""
print(max_lh_instance.cti)

"""
This object was composed with `trap_list` and `ccd` entries, which are also contained in the results instance.
"""
print(max_lh_instance.cti.trap_list)
print(max_lh_instance.cti.ccd)

"""
__Vectors__

All results can alternatively be returned as a 1D vector of values, by passing `as_instance=False`:
"""
max_lh_vector = samples.max_log_likelihood(as_instance=False)
print("Max Log Likelihood Model Parameters: \n")
print(max_lh_vector, "\n\n")

"""
__Labels__

Vectors return a lists of all model parameters, but do not tell us which values correspond to which parameters.

The following quantities are available in the `Model`, where the order of their entries correspond to the parameters 
in the `ml_vector` above:
 
 - `paths`: a list of tuples which give the path of every parameter in the `Model`.
 - `parameter_names`: a list of shorthand parameter names derived from the `paths`.
 - `parameter_labels`: a list of parameter labels used when visualizing non-linear search results (see below).

For simple models like the one fitted in this tutorial, the quantities below are somewhat redundant. For the
more complex models illustrated in other tutorials their utility will become clear.
"""
model = samples.model

print(model.paths)
print(model.parameter_names)
print(model.parameter_labels)
print(model.model_component_and_parameter_names)
print("\n")


"""
__Posterior / PDF__

PDF stands for "Probability Density Function" and it quantifies the probability of each model sampled. It 
therefore enables error estimation via a process called marginalization.

We can access the `median_pdf` model, which is the model computed by marginalizing over the samples of every 
parameter in 1D and taking the median of this PDF.
"""
median_pdf_instance = samples.median_pdf()

print("Median PDF Model Instances: \n")
print(median_pdf_instance, "\n")
print(median_pdf_instance.cti.trap_list)
print()

median_pdf_vector = samples.median_pdf(as_instance=False)

print("Median PDF Model Parameter Lists: \n")
print(median_pdf_vector, "\n")

"""
__Errors__

We can compute the model parameters at a given sigma value (e.g. at 3.0 sigma limits).

These parameter values and error estimates do not fully account for covariance between model parameters, which is
explained in the "Derived Errors" section below.

The `uv3` below signifies this is an upper value at 3 sigma confidence, with `lv3` indicating a the lower value.
"""
uv3_instance = samples.values_at_upper_sigma(sigma=3.0)
lv3_instance = samples.values_at_lower_sigma(sigma=3.0)

print("Errors Instances: \n")
print(uv3_instance, "\n")
print(lv3_instance, "\n")

"""
We can compute the upper and lower errors on each parameter at a given sigma limit.

The `ue3` below signifies the upper error at 3 sigma. 
"""
ue3_instance = samples.errors_at_upper_sigma(sigma=3.0)
le3_instance = samples.errors_at_lower_sigma(sigma=3.0)

print("Errors Instances: \n")
print(ue3_instance, "\n")
print(le3_instance, "\n")

"""
__Search Plots__

The Probability Density Functions of the results can be plotted using Dynesty's in-built visualization tools, 
which are wrapped via the `DynestyPlotter` object.
"""
search_plotter = aplt.DynestyPlotter(samples=result.samples)
search_plotter.cornerplot()

"""
__Sample Instance__

A dynesty search retains every model that is accepted during the model-fit.

We can create an instance of any model -- for example the 100th last model accepted -- and can compare its parameters 
to the maximum log likelihood model.
"""
instance = samples.from_sample_index(sample_index=-10)

print(max_lh_instance.cti.trap_list)
print(instance.cti.trap_list)

"""
__Maximum Likelihood__

The maximum log likelihood value of the model-fit can be estimated by simple taking the maximum of all log
likelihoods of the samples.

If different models are fitted to the same dataset, this value can be compared to determine which model provides
the best fit (e.g. which model has the highest maximum likelihood)?
"""
print("Maximum Log Likelihood: \n")
print(max(samples.log_likelihood_list))

"""
__Bayesian Evidence__

Nested sampling algorithms like dynesty also estimate the Bayesian evidence (estimated via the nested sampling 
algorithm).

The Bayesian evidence accounts for "Occam's Razor", whereby it penalizes models for being more complex (e.g. if a model
has more parameters it needs to fit the da

The Bayesian evidence is a better quantity to use to compare models, because it penalizes models with more parameters
for being more complex ("Occam's Razor"). Comparisons using the maximum likelihood value do not account for this and
therefore may unjustly favour more complex models.

Using the Bayesian evidence for model comparison is well documented on the internet, for example the following
wikipedia page: https://en.wikipedia.org/wiki/Bayes_factor
"""
print("Maximum Log Likelihood and Log Evidence: \n")
print(samples.log_evidence)

"""
__Latex__

**PyAutoFit**'s latex tools can create latex table code which you can copy to a .tex document (e.g. a paper).

By combining this with the filtering tools below, specific parameters can be included or removed from the latex.

The superscripts of each parameter's latex string are loaded from the config file `notation/label.yaml`. Editing this
config file provides high levels of customization for each parameter appears in the latex table. 

This is especially useful if your CTI model uses the same model components with the same parameter (e.g. parallel and
serial CTI), which therefore need to be distinguished via superscripts.
"""
latex = af.text.Samples.latex(
    samples=result.samples,
    median_pdf_model=True,
    sigma=3.0,
    name_to_label=True,
    include_name=True,
    include_quickmath=True,
    prefix="Example Prefix ",
    suffix=r"\\[-2pt]",
)

print(latex)

"""
__Wrap Up__

This tutorial illustrated how to analyse the non-linear samples of a model-fit. We saw that the API used for model 
composition produced the instances of the results after fitting (for example, how we accessed the galaxy model 
as `instance.cti.trap_list`).

The remainder of this script provides advanced samples calculations which:

 - Calculate the most probable value with errors of a derived quantity by computing its PDF. For example, computing 
   the total trap density of the model with errors (the total trap denisty is not directly accessible from the model).
 
 - Filtering `Samples` objects to remove certain parameters or components from the model. This can help with,
   for example, creating LaTex tables of specific parts of a model.

__Derived Errors (Advanced)__

Computing the errors of a quantity like the `release_timescale` is simple, because it is sampled by the non-linear 
search. Errors are accessible using the `Samples` object's `errors_from` methods, which marginalize over the 
parameters via the 1D Probability Density Function (PDF).

Computing errors on derived quantities is more tricky, because it is not sampled directly by the non-linear search. 
For example, what if we want the error on the total trap density of the CTI model? In order to do this we need to create 
the PDF of that derived quantity, which we can then marginalize over using the same function we use to marginalize 
model parameters.

Below, we compute the total trap density of every accepted model sampled by the non-linear search and use this 
determine the PDF of the total trap density. When combining the axis-ratio's we weight each value by its `weight`. 
For Dynesty, a nested sampling algorithm, the weight of every sample is different and thus must be included.

In order to pass these samples to the function `marginalize`, which marginalizes over the PDF of the axis-ratio to 
compute its error, we also pass the weight list of the samples.
"""
total_density_list = []

for sample in samples.sample_list:
    instance = sample.instance_for_model(model=samples.model)

    trap_list = instance.galaxies.cti.trap_list

    total_density = sum([trap.density for trap in trap_list])

    total_density_list.append(total_density)

median_total_density, upper_total_density, lower_total_density = af.marginalize(
    parameter_list=total_density_list, sigma=3.0, weight_list=samples.weight_list
)

print(
    f"total_density = {median_total_density} ({upper_total_density} {lower_total_density}"
)


"""
__Samples Filtering (Advanced)__

The samples object has the results for all model parameter. It can be filtered to contain the results of specific 
parameters of interest.

The basic form of filtering specifies parameters via their path, which was printed above via the model and is printed 
again below.
"""
samples = result.samples

print("Parameter paths in the model which are used for filtering:")
print(samples.model.paths)

print("All parameters of the very first sample")
print(samples.parameter_lists[0])

samples = samples.with_paths(
    [
        ("cti", "trap_list", "density"),
    ]
)

print(
    "All parameters of the very first sample (containing only the trap list densities)."
)
print(samples.parameter_lists[0])

print(
    "Maximum Log Likelihood Model Instances (containing only the trap list densities):\n"
)
print(samples.max_log_likelihood(as_instance=False))

"""
Above, we specified each path as a list of tuples of strings. 

This is how the **PyAutoFit** source code stores the path to different components of the model, but it is not in-line 
with the **PyAutoCTI** API used to compose a model.

We can alternatively use the following API:
"""
samples = result.samples

samples = samples.with_paths(["cti.trap_list.density", "cti.ccd.well_fill_power"])

print(
    "All parameters of the very first sample (containing only the galaxy bulge's effective radius and sersic index)."
)

"""
We can alternatively filter the `Samples` object by removing all parameters with a certain path. 

Below, we remove the `density` parameters of the CTI model to be left with 4 parameters.
"""
samples = result.samples

print("Parameter paths in the model which are used for filtering:")
print(samples.model.paths)

print("Parameters of first sample")
print(samples.parameter_lists[0])

print(samples.model.prior_count)

samples = samples.without_paths(["cti.trap_list.density"])

print("Parameters of first sample without trap densities.")
print(samples.parameter_lists[0])

"""
We can keep and remove entire paths of the samples, for example keeping only the parameters of the trap densities.
"""
# samples = result.samples
# samples = samples.with_paths(["cti.trap_list"])
# print("Parameters of the first sample of the trap list")
# print(samples.parameter_lists[0])

samples = result.samples
samples = samples.without_paths(["cti.trap_list"])
print("Parameters of the first sample without the trap list")
print(samples.parameter_lists[0])

"""
Fin.
"""
