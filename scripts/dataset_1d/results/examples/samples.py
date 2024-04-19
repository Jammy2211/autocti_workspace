"""
Results: Samples
================

After a non-linear search has completed, it returns a `Result` object that contains information on samples of
the non-linear search, such as the maximum likelihood model instance, the errors on each parameter and the 
Bayesian evidence.

This script illustrates how to use the result to inspect the non-linear search samples.

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

To illustrate results, we need to perform a model-fit in order to create a `Result` object.

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
Results tutorials `cti.py` and `fits.py` expand on the `max_log_likelihood_instance` and `max_log_likelihood_fit`, 
showing how they can be used to inspect many aspects of a model.

__Samples__

The result contains a `Samples` object, which contains all samples of the non-linear search.

Each sample corresponds to a set of model parameters that were evaluated and accepted by the non linear search, 
in this example `nautilus`. 

This includes their log likelihoods, which are used for computing additional information about the model-fit,
for example the error on every parameter. 

Our model-fit used the nested sampling algorithm Nautilus, so the `Samples` object returned is a `SamplesNest` object.
"""
samples = result.samples

print("Nest Samples: \n")
print(samples)

"""
__Parameters__

The parameters are stored as a list of lists, where:

 - The outer list is the size of the total number of samples.
 - The inner list is the size of the number of free parameters in the fit.
"""
print("All parameters of the very first sample")
print(samples.parameter_lists[0])
print("The fourth parameter of the tenth sample")
print(samples.parameter_lists[9][3])

"""
__Figures of Merit__

The `Samples` class contains the log likelihood, log prior, log posterior and weight_list of every accepted sample, where:

- The `log_likelihood` is the value evaluated in the `log_likelihood_function`.

- The `log_prior` encodes information on how parameter priors map log likelihood values to log posterior values.

- The `log_posterior` is `log_likelihood + log_prior`.

- The `weight` gives information on how samples are combined to estimate the posterior, which depends on type of search
  used (for `nautilus` they are all non-zero values which sum to 1).

Lets inspect these values for the tenth sample.
"""
print("log(likelihood), log(prior), log(posterior) and weight of the tenth sample.")
print(samples.log_likelihood_list[9])
print(samples.log_prior_list[9])
print(samples.log_posterior_list[9])
print(samples.weight_list[9])

"""
__Maximum Likelihood__

Many results can be returned as an instance of the model, using the Python class structure of the model composition.

For example, we can return the model parameters corresponding to the maximum log likelihood sample.
"""
instance = samples.max_log_likelihood()
print("Maximum Log Likelihood Model Instance: \n")
print(instance, "\n")

"""
The attributes of the `instance` (e.g. `cti`, `trap_list`) have these names due to how we composed the `CTI1D` object
and its traps and CCD via the `Collection` and `Model` above. 

A model instance contains all the model components of our fit, for example the `CTI1D` object we specified during 
model composition with the name `cti`.
"""
print(instance.cti)

"""
This object was composed with `trap_list` and `ccd` entries, which are also contained in the results instance.
"""
print(instance.cti.trap_list)
print(instance.cti.ccd)

"""
This allows individual parameters to be printed.
"""
print(instance.cti.trap_list[0].density)


"""
__Posterior / PDF__

The result contains the full posterior information of our non-linear search, which can be used for parameter 
estimation. 

PDF stands for "Probability Density Function" and it quantifies probability of each model parameter having values
that are sampled. It therefore enables error estimation via a process called marginalization.

The median pdf vector is available, which estimates every parameter via 1D marginalization of their PDFs.
"""
instance = samples.median_pdf()

print("Median PDF Model Instances: \n")
print(instance, "\n")
print(instance.cti.trap_list)
print()

vector = samples.median_pdf(as_instance=False)

print("Median PDF Model Parameter Lists: \n")
print(vector, "\n")

"""
__Errors__

Methods for computing error estimates on all parameters are provided. 

This again uses 1D marginalization, now at an input sigma confidence limit. 

By inputting `sigma=3.0` margnialization find the values spanning 99.7% of 1D PDF. Changing this to `sigma=1.0`
would give the errors at the 68.3% confidence limit.
"""
instance_upper_sigma = samples.values_at_upper_sigma(sigma=3.0)
instance_lower_sigma = samples.values_at_lower_sigma(sigma=3.0)

print("Errors Instances: \n")
print(instance_upper_sigma.cti.trap_list, "\n")
print(instance_lower_sigma.cti.trap_list, "\n")

"""
They can also be returned at the values of the parameters at their error values.
"""
instance_upper_values = samples.errors_at_upper_sigma(sigma=3.0)
instance_lower_values = samples.errors_at_lower_sigma(sigma=3.0)

print("Errors Instances: \n")
print(instance.cti.trap_list, "\n")
print(instance.cti.trap_list, "\n")

"""
__Sample Instance__

A non-linear search retains every model that is accepted during the model-fit.

We can create an instance of any model -- below we create an instance of the last accepted model.
"""
instance = samples.from_sample_index(sample_index=-1)

print(instance.cti.trap_list)
print(instance.cti.trap_list)

"""
__Search Plots__

The Probability Density Functions (PDF's) of the results can be plotted using the non-linear search in-built 
visualization tools.

This fit used `nautilus` therefore we use the `NestPlotter` for visualization, which wraps `nautilus`'s in-built
visualization tools.

The `autofit_workspace/*/plots` folder illustrates other packages that can be used to make these plots using
the standard output results formats (e.g. `GetDist.py`).
"""
plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_cornerpy()

"""
__Instances__

The maximum log likelihood value of the model-fit can be estimated by simple taking the maximum of all log
likelihoods of the samples.

If different models are fitted to the same dataset, this value can be compared to determine which model provides
the best fit (e.g. which model has the highest maximum likelihood)?
"""
print("Maximum Log Likelihood: \n")
print(max(samples.log_likelihood_list))

"""
__Bayesian Evidence__

Nested sampling algorithms like nautilus also estimate the Bayesian evidence (estimated via the nested sampling 
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
__Lists__

All results can alternatively be returned as a 1D list of values, by passing `as_instance=False`:
"""
max_lh_list = samples.max_log_likelihood(as_instance=False)
print("Max Log Likelihood Model Parameters: \n")
print(max_lh_list, "\n\n")

"""
The list above does not tell us which values correspond to which parameters.

The following quantities are available in the `Model`, where the order of their entries correspond to the parameters 
in the `ml_vector` above:

 - `paths`: a list of tuples which give the path of every parameter in the `Model`.
 - `parameter_names`: a list of shorthand parameter names derived from the `paths`.
 - `parameter_labels`: a list of parameter labels used when visualizing non-linear search results (see below).

For simple models like the one fitted in this tutorial, the quantities below are somewhat redundant. For the
more complex models they are important for tracking the parameters of the model.
"""
model = samples.model

print(model.paths)
print(model.parameter_names)
print(model.parameter_labels)
print(model.model_component_and_parameter_names)
print("\n")

"""
All the methods above are available as lists.
"""
instance = samples.median_pdf(as_instance=False)
values_at_upper_sigma = samples.values_at_upper_sigma(sigma=3.0, as_instance=False)
values_at_lower_sigma = samples.values_at_lower_sigma(sigma=3.0, as_instance=False)
errors_at_upper_sigma = samples.errors_at_upper_sigma(sigma=3.0, as_instance=False)
errors_at_lower_sigma = samples.errors_at_lower_sigma(sigma=3.0, as_instance=False)

"""
__Latex__

If you are writing modeling results up in a paper, you can use inbuilt latex tools to create latex table code which 
you can copy to your .tex document.

By combining this with the filtering tools below, specific parameters can be included or removed from the latex.

Remember that the superscripts of a parameter are loaded from the config file `notation/label.yaml`, providing high
levels of customization for how the parameter names appear in the latex table. This is especially useful if your model
uses the same model components with the same parameter, which therefore need to be distinguished via superscripts.
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
__Derived Errors (Advanced)__

Computing the errors of a quantity like the `release_timescale` is simple, because it is sampled by the non-linear 
search. Errors are accessible using the `Samples` object's `errors_from` methods, which marginalize over the 
parameters via the 1D Probability Density Function (PDF).

Computing errors on derived quantities is more tricky, because they are not sampled directly by the non-linear search. 
For example, what if we want the error on the total trap density of the CTI model? In order to do this we need to create 
the PDF of that derived quantity, which we can then marginalize over using the same function we use to marginalize 
model parameters.

Below, we compute the total trap density of every accepted model sampled by the non-linear search and use this 
determine the PDF of the total trap density. When combining the axis-ratio's we weight each value by its `weight`. 
For Nautilus, a nested sampling algorithm, the weight of every sample is different and thus must be included.

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

Our samples object has the results for all three parameters in our model. However, we might only be interested in the
results of a specific parameter.

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
We specified each path as a list of tuples of strings. 

This is how the source code internally stores the path to different components of the model, but it is not 
consistent with the API used to compose a model.

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
