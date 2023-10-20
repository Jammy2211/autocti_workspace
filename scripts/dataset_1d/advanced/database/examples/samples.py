"""
Database: Samples
=================

In the script `autocti_workspace/*/advanced/database/start_here.py` we performed a fit which fitted 3
datasets and stored the results in a sqlite database. 

In this example, we'll load results from this database and show how to manipulate the non-linear search's samples,
for example to inspect the maximum log likelihood models or get errors on parameters.

__Samples via Result__

A fraction of this example repeats the API for manipulating samples given in the
`autogalaxy_workspace/*/results/examples/samples.py` example.

This is done so users can directly copy and paste Python code which loads results from the database and manipulates
the samples.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autocti.plot as aplt

"""
__Files__

In the `start_here.py` script, we discussed the `files` that are output by the non-linear search. The 
following files correspond to the information loaded when loading the non-linear search samples from the database:

 - `model`: The `model` defined above and used in the model-fit (`model.json`).
 - `samples`: The non-linear search samples (`samples.csv`).
 - `samples_info`: Additional information about the samples (`samples_info.json`).
 - `samples_summary`: A summary of key results of the samples (`samples_summary.json`).
 - `covariance`: The inferred covariance matrix (`covariance.csv`).

The `samples` and `samples_summary` results contain a lot of repeated information. The `samples` result contains
the full non-linear search samples, for example every parameter sample and its log likelihood. The `samples_summary`
contains a summary of the results, for example the maximum log likelihood model and error estimates on parameters
at 1 and 3 sigma confidence.

Accessing results via the `samples_summary` is much faster, because as it does reperform calculations using the full 
list of samples. Therefore, if the result you want is accessible via the `samples_summary` you should use it
but if not you can revert to the `samples.

__Database File__

The results are not contained in the `output` folder after each search completes. Instead, they are
contained in the `database.sqlite` file, which we can load using the `Aggregator`.
"""
database_file = "database.sqlite"
agg = af.Aggregator.from_database(filename=database_file)

"""
__Generators__

The `start_here.py` database example gives an explanation of what Python generators are and why and how they are used.
Refer back to that example if you are unsure.
"""
samples_gen = agg.values("samples")

"""
When we print this the length of this generator converted to a list of outputs we see 3 different `SamplesNest`
instances. 

These correspond to each fit of each search to each of our 3 images.
"""
print("NestedSampler Samples: \n")
print(samples_gen)
print()
print("Total Samples Objects = ", len(agg), "\n")

"""
__Samples__

The result contains a `Samples` object, which contains all samples of the non-linear search, which is accessible
via the database and aggregator.

Each sample corresponds to a set of model parameters that were evaluated and accepted by the non linear search, 
in this example `nautilus`. 

This includes their log likelihoods, which are used for computing additional information about the model-fit,
for example the error on every parameter. 
"""
print("Samples: \n")
print(agg.values("samples"))
print()
print("Total Samples Objects = ", len(agg), "\n")

"""
__Parameters__

The parameters are stored as a list of lists, where:

 - The outer list is the size of the total number of samples.
 - The inner list is the size of the number of free parameters in the fit
"""
for samples in agg.values("samples"):
    print("All parameters of the very first sample")
    print(samples.parameter_lists[0])
    print("The third parameter of the tenth sample")
    print(samples.parameter_lists[9][2])
    print()

"""
__Figures of Merit__

The `Samples` class contains the log likelihood, log prior, log posterior and weight_list of every accepted sample, where:

- The `log_likelihood` is the value evaluated in the `log_likelihood_function`.

- The `log_prior` encodes information on how parameter priors map log likelihood values to log posterior values.

- The `log_posterior` is `log_likelihood + log_prior`.

- The `weight` gives information on how samples are combined to estimate the posterior, which depends on type of search
  used (for `nautilus` they are all non-zero values which sum to 1).

Lets inspect these values for the tenth sample of each of the 3 model-fits.
"""
for samples in agg.values("samples"):
    print("log(likelihood), log(prior), log(posterior) and weight of the tenth sample.")
    print(samples.log_likelihood_list[9])
    print(samples.log_prior_list[9])
    print(samples.log_posterior_list[9])
    print(samples.weight_list[9])

"""
__Samples Summary__

The samples summary contains a subset of results access via the `Samples`, for example the maximum likelihood model
and parameter error estimates.

Using the samples method above can be slow, as the quantities have to be computed from all non-linear search samples
(e.g. computing errors requires that all samples are marginalized over). This information is stored directly in the
samples summary and can therefore be accessed instantly.
"""
# for samples_summary in agg.values("samples_summary"):
#
#     instance = samples_summary.max_log_likelihood()
#
#     print("Max Log Likelihood `Gaussian` Instance:")
#     print("Centre = ", instance.centre)
#     print("Normalization = ", instance.normalization)
#     print("Sigma = ", instance.sigma, "\n")

"""
__Maximum Likelihood Model__

We can use the outputs to create a list of the maximum log likelihood model of each fit to our three images.
"""
ml_list = [
    samps.max_log_likelihood(as_instance=False) for samps in agg.values("samples")
]

print("Max Log Likelihood Model Parameter Lists: \n")
print(ml_list, "\n\n")

"""
__Parameter Names__

Vectors return a lists of all model parameters, but do not tell us which values correspond to which parameters.

The following quantities are available in the `Model`, where the order of their entries correspond to the parameters 
in the `ml_list` above:
 
 - `paths`: a list of tuples which give the path of every parameter in the `Model`.
 - `parameter_names`: a list of shorthand parameter names derived from the `paths`.
 - `parameter_labels`: a list of parameter labels used when visualizing non-linear search results (see below).
"""
for samples in agg.values("samples"):
    model = samples.model
    print(model)
    print(model.paths)
    print(model.parameter_names)
    print(model.parameter_labels)
    print()

"""
These lists will be used later for visualization, how it is often more useful to create the model instance of every fit.
"""
ml_instances = [samps.max_log_likelihood() for samps in agg.values("samples")]
print("Maximum Log Likelihood Model Instances: \n")
print(ml_instances, "\n")

"""
__Instances__

We can use the `Aggregator` to create a list of instances of the model, using the Python class structure of the 
model composition.

For example, we can return a list of the model instances corresponding to the maximum log likelihood sample.
"""
print(ml_instances[0].cti)
print(ml_instances[1].cti)
print(ml_instances[2].cti)

"""
These galaxies will be named according to the model composed and fitted by the search (in this case, only `cti`).
"""
print(ml_instances[0].cti.trap_list)
print()
print(ml_instances[1].cti.ccd)

"""
Their individual parameters can be printed.
"""
print(ml_instances[0].cti.trap_list[0].density)
print(ml_instances[1].cti.ccd.well_fill_power)

"""
__Posterior / PDF__

The result contains the full posterior information of our non-linear search, which can be used for parameter 
estimation. 

PDF stands for "Probability Density Function" and it quantifies probability of each model parameter having values
that are sampled. It therefore enables error estimation via a process called marginalization.

The median pdf vector is available, which estimates every parameter via 1D marginalization of their PDFs.
"""
mp_instances = [samps.median_pdf() for samps in agg.values("samples")]

print("Median PDF Model Instances: \n")
print(mp_instances, "\n")
print(mp_instances[0].cti.trap_list[0])
print()

"""
__Errors__

Methods for computing error estimates on all parameters are provided. 

This again uses 1D marginalization, now at an input sigma confidence limit. 

By inputting `sigma=3.0` margnialization find the values spanning 99.7% of 1D PDF. Changing this to `sigma=1.0`
would give the errors at the 68.3% confidence limit.
"""
uv3_lists = [samps.values_at_upper_sigma(sigma=3.0) for samps in agg.values("samples")]

uv3_instances = [
    samps.values_at_upper_sigma(sigma=3.0) for samps in agg.values("samples")
]

lv3_lists = [samps.values_at_lower_sigma(sigma=3.0) for samps in agg.values("samples")]

lv3_instances = [
    samps.values_at_lower_sigma(sigma=3.0) for samps in agg.values("samples")
]

print("Errors Lists: \n")
print(uv3_lists, "\n")
print(lv3_lists, "\n")
print("Errors Instances: \n")
print(uv3_instances, "\n")
print(lv3_instances, "\n")

"""
We can compute the upper and lower errors on each parameter at a given sigma limit.

The `ue3` below signifies the upper error at 3 sigma. 
"""
try:
    ue3_lists = [samps.errors_at_upper_sigma(sigma=3.0) for samps in agg.values("samples")]

    # ue3_instances = [
    #     samps.errors_at_upper_sigma(sigma=3.0) for samps in agg.values("samples")
    # ]

    le3_lists = [samps.errors_at_lower_sigma(sigma=3.0) for samps in agg.values("samples")]
    # le3_instances = [
    #     samps.errors_at_lower_sigma(sigma=3.0) for samps in agg.values("samples")
    # ]

    print("Errors Lists: \n")
    print(ue3_lists, "\n")
    print(le3_lists, "\n")
    print("Errors Instances: \n")
    # print(ue3_instances, "\n")
    # print(le3_instances, "\n")
except af.exc.FitException:
    pass

"""
__Sample Instance__

A non-linear search retains every model that is accepted during the model-fit.

We can create an instance of any model -- below we create an instance of the last accepted model.
"""
for samples in agg.values("samples"):
    instance = samples.from_sample_index(sample_index=-1)

    print(instance.cti.trap_list)

"""
__Search Plots__

The Probability Density Functions (PDF's) of the results can be plotted using the non-linear search in-built 
visualization tools.

This fit used `nautilus` therefore we use the `NautilusPlotter` for visualization, which wraps `nautilus`'s in-built
visualization tools.

The `autofit_workspace/*/plots` folder illustrates other packages that can be used to make these plots using
the standard output results formats (e.g. `GetDist.py`).
"""
for samples in agg.values("samples"):
    search_plotter = aplt.NautilusPlotter(samples=samples)
#  search_plotter.cornerplot()

"""
__Maximum Likelihood__

The maximum log likelihood value of the model-fit can be estimated by simple taking the maximum of all log
likelihoods of the samples.

If different models are fitted to the same dataset, this value can be compared to determine which model provides
the best fit (e.g. which model has the highest maximum likelihood)?
"""
print([max(samps.log_likelihood_list) for samps in agg.values("samples")])

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
print("Log Evidences: \n")
# print([samps.log_evidence for samps in agg.values("samples")])

"""
__Lists__

All results can alternatively be returned as a 1D list of values, by passing `as_instance=False`:
"""
for samples in agg.values("samples"):
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
for model in agg.values("model"):
    print(model.paths)
    print(model.parameter_names)
    print(model.parameter_labels)
    print(model.model_component_and_parameter_names)
    print("\n")

"""
__Latex__

If you are writing modeling results up in a paper, you can use inbuilt latex tools to create latex table code which 
you can copy to your .tex document.

By combining this with the filtering tools below, specific parameters can be included or removed from the latex.

Remember that the superscripts of a parameter are loaded from the config file `notation/label.yaml`, providing high
levels of customization for how the parameter names appear in the latex table. This is especially useful if your model
uses the same model components with the same parameter, which therefore need to be distinguished via superscripts.
"""
for samples in agg.values("samples"):
    latex = af.text.Samples.latex(
        samples=samples,
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
__Ordering__

The default ordering of the results can be a bit random, as it depends on how the sqlite database is built. 

The `order_by` method can be used to order by a property of the database that is a string, for example by ordering 
using the `unique_tag` (which we set up in the search as the `dataset_name`) the database orders results alphabetically
according to dataset name.
"""
agg = agg.order_by(agg.search.unique_tag)

"""
We can also order by a bool, for example making it so all completed results are at the front of the aggregator.
"""
agg = agg.order_by(agg.search.is_complete)


"""
__Samples Filtering__

The samples object has the results for all model parameter. It can be filtered to contain the results of specific 
parameters of interest.

The basic form of filtering specifies parameters via their path, which was printed above via the model and is printed 
again below.
"""
samples = list(agg.values("samples"))[0]

print("Parameter paths in the model which are used for filtering:")
print(samples.model.paths)

print("All parameters of the very first sample")
print(samples.parameter_lists[0])

# samples = samples.with_paths(
#     [
#         ("cti", "trap_list", "density"),
#     ]
# )
#
# print(
#     "All parameters of the very first sample (containing only the CTI trap list's densities)."
# )
# print(samples.parameter_lists[0])
#
# print(
#     "Maximum Log Likelihood Model Instances (containing only the CTI trap list's densities):\n"
# )
# print(samples.max_log_likelihood(as_instance=False))

"""
Above, we specified each path as a list of tuples of strings. 

This is how the source code internally stores the path to different components of the model, but it is not in-line 
with the PyAutoCTI API used to compose a model.

We can alternatively use the following API:
"""
# samples = list(agg.values("samples"))[0]
#
# samples = samples.with_paths(
#     ["cti.trap_list.release_timescale", "cti.ccd.well_fill_power"]
# )
#
# print(
#     "All parameters of the very first sample (containing only the CTI model's release timescales and well fill power)."
# )

"""
Above, we filtered the `Samples` but asking for all parameters which included the
path ("cti", "trap_list", "density").

We can alternatively filter the `Samples` object by removing all parameters with a certain path. Below, we remove
the centres of the mass model to be left with 10 parameters.
"""
samples = list(agg.values("samples"))[0]

print("Parameter paths in the model which are used for filtering:")
print(samples.model.paths)

print("Parameters of first sample")
print(samples.parameter_lists[0])

print(samples.model.prior_count)

# samples = samples.without_paths(
#     [
#         # ("cti", "trap_list", "density"),
#         "cti.trap_list.density",
#     ]
# )
#
# print("Parameters of first sample without the densities of the traps")
# print(samples.parameter_lists[0])

"""
We can keep and remove entire paths of the samples, for example keeping only the parameters of the trap list.
"""
samples = list(agg.values("samples"))[0]
samples = samples.with_paths(["cti.trap_list"])
print("Parameters of the first sample of CTI trap list")
print(samples.parameter_lists[0])


"""
__Latex__

If you are writing results up in a paper, you can use inbuilt latex tools to create latex 
table code which you can copy to your .tex document.

By combining this with the filtering tools below, specific parameters can be included or removed from the latex.

Remember that the superscripts of a parameter are loaded from the config file `notation/label.yaml`, providing high
levels of customization for how the parameter names appear in the latex table. This is especially useful if your 
model uses the same model components with the same parameter, which therefore need to be distinguished via superscripts.
"""
latex = af.text.Samples.latex(
    samples=list(agg.values("samples"))[0],
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
Finished.
"""
