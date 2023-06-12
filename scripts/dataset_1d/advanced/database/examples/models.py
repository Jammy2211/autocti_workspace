"""
Database: Models
================

In this tutorial, we use the database to load models and `CTI` objects from a non-linear search. This allows us to
visualize and interpret its results.

We then show how the database also allows us to load many `CTI` objects correspond to many samples of the non-linear
search. This allows us to compute the errors on quantities that the `CTI` object contains, but were not sampled
directly by the non-linear search.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autocti as ac
import autocti.plot as aplt

"""
__Database File__

First, set up the aggregator as we did in the previous tutorial.
"""
agg = af.Aggregator.from_database("database.sqlite")

"""
__CTI via Database__

Having performed a model-fit, we now want to interpret and visualize the results. In this example, we want to inspect
the `CTI` object objects that gave good fits to the data. 

Using the API shown in the `start_here.py` example this would require us to create a `Samples` object and manually 
compose our own `CTI` object. For large datasets, this would require us to use generators to ensure it is memory-light,
which are cumbersome to write.

This example therefore uses the `CTIAgg` object, which conveniently loads the `CTI` objects of every fit via 
generators for us. Explicit examples of how to do this via generators is given in the `advanced/manual_generator.py` 
tutorial.

We get a CTI generator via the `ac.agg.CTIAgg` object, where this `cti_gen` contains the maximum log
likelihood `CTI `object of every model-fit.
"""
cti_agg = ac.agg.CTIAgg(aggregator=agg)
cti_gen = cti_agg.max_log_likelihood_gen_from()

"""
We can now iterate over our `CTI `object generator to make the plots we desire.
"""
for CTI in cti_gen:
    print(cti)

"""
__CTI Delta Ellipticity Requirement__

Each `CTI `object can inform us, based on its error, what the induced spurious ellipticity on galaxy shape 
measurements is.
"""
cti_gen = cti_agg.max_log_likelihood_gen_from()

print("Maximum Log Likelihood Spurious Ellipticity:")

for CTI in cti_gen:
    delta_ellipticity = cti.delta_ellipticity

    print("Delta Ellipticity = ", delta_ellipticity)

"""
__Errors (PDF from samples)__

In this example, we will compute the errors on the delta ellipticity of a model. Computing the errors on a quantity 
like the trap `density` is simple, because it is sampled by the non-linear search. The errors are therefore accessible
via the `Samples`, by marginalizing over all over parameters via the 1D Probability Density Function (PDF).

Computing the errors on the delta ellipticity is more tricky, because it is a derived quantity. It is a parameter or 
measurement that we want to calculate but was not sampled directly by the non-linear search. The `CTIAgg` object 
object has everything we need to compute the errors of derived quantities.

Below, we compute the delta ellipticity of every model sampled by the non-linear search and use this determine the PDF 
of the delta ellipticity. When combining each delta ellipticity we weight each value by its `weight`. For Dynesty, 
the nested sampler used by the fit, this ensures models which gave a bad fit (and thus have a low weight) do not 
contribute significantly to the delta ellipticity error estimate.

We set `minimum_weight=`1e-4`, such that any sample with a weight below this value is discarded when computing the 
error. This speeds up the error computation by only using a small fraction of the total number of samples. Computing
a delta ellipticity is cheap, and this is probably not necessary. However, certain quantities have a non-negligible
computational overhead is being calculated and setting a minimum weight can speed up the calculation without 
significantly changing the inferred errors.

Below, we use the `CTIAgg` object to get the `CTI` object of every Dynesty sample in each model-fit. We extract from 
each `CTI `object the model's delta ellipticity, store them in a list and find the value via the PDF and quantile 
method. This again uses generators, ensuring minimal memory use. 

In order to use these samples in the function `quantile`, we also need the weight list of the sample weights. We 
compute this using the ``CTIAgg`'s function `weights_above_gen_from`, which computes generators of the weights of 
all points above this minimum value. This again ensures memory use in minimal.
"""
cti_agg = ac.agg.CTIAgg(aggregator=agg)
cti_list_gen = cti_agg.all_above_weight_gen_from(minimum_weight=1e-4)
weight_list_gen = cti_agg.weights_above_gen_from(minimum_weight=1e-4)

for cti_gen, weight_gen in zip(cti_list_gen, weight_list_gen):
    delta_ellipticity_list = []

    for CTI in cti_gen:
        delta_ellipticity = cti.delta_ellipticity

        delta_ellipticity_list.append(delta_ellipticity)

    weight_list = [weight for weight in weight_gen]

    (
        median_delta_ellipticity,
        upper_delta_ellipticity,
        lower_delta_ellipticity,
    ) = af.marginalize(
        parameter_list=delta_ellipticity_list, sigma=2.0, weight_list=weight_list
    )

    print(
        f"delta ellipticity = {median_delta_ellipticity} ({upper_delta_ellipticity} {lower_delta_ellipticity}"
    )

"""
__Errors (Random draws from PDF)__

An alternative approach to estimating the errors on a derived quantity is to randomly draw samples from the PDF 
of the non-linear search. For a sufficiently high number of random draws, this should be as accurate and precise
as the method above. However, it can be difficult to be certain how many random draws are necessary.

The weights of each sample are used to make every random draw. Therefore, when we compute the delta ellipticity and its 
errors we no longer need to pass the `weight_list` to the `quantile` function.
"""
cti_agg = ac.agg.CTIAgg(aggregator=agg)
cti_list_gen = cti_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

for cti_gen in cti_list_gen:
    delta_ellipticity_list = []

    for CTI in cti_gen:
        delta_ellipticity = cti.delta_ellipticity

        delta_ellipticity_list.append(delta_ellipticity)

    (
        median_delta_ellipticity,
        upper_delta_ellipticity,
        lower_delta_ellipticity,
    ) = af.marginalize(parameter_list=delta_ellipticity_list, sigma=3.0)

    print(
        f"delta ellipticity = {median_delta_ellipticity} ({upper_delta_ellipticity} {lower_delta_ellipticity}"
    )

"""
Finish.
"""
