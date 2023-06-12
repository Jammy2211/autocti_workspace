"""
Database: Data Fitting
======================

In this tutorial, we use the aggregator to load models and data from a non-linear search and use them to perform
fits to the data.

All examples use just a single instance of a `FitDataset1D` object corresponding to the maximum log likelihood sample.

__Charge Injection Imaging__

This script can easily be adapted to analyse the results of charge injection imaging model-fits.

The only entries that needs changing are: 

 - `Dataset1DAgg` -> `ImagingCIAgg`.
 - `FitDataset1DAgg` -> `FitImagingCIAgg`.
 - `Clocker1D` -> `Clocker2D`.
 - `SettingsDataset1D` -> `SettingsImagingCI`.
 - `Dataset1DPlotter` -> `ImagingCIPlotter`.
 - `FitDataset1DPlotter` -> `FitImagingCIPlotter`.
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
__Database File__

First, set up the aggregator as we did in the previous tutorial.
"""
agg = af.Aggregator.from_database("database.sqlite")

"""
__Fits via Database__

Having performed a model-fit, we now want to interpret and visualize the results. In this example, we want to inspect
the `Dataset1D` objects that gave good fits to the data. 

Using the tools introduced in the previous tutorials, this would require us to create a `Samples` object and manually 
compose our own `Dataset1D`. To ensure this is memory-light, this would have to be done via generators, which can be 
tricky to write. There is a optional database tutorial which how these datasets can be retrieved manually, but here we 
use convenience objects instead.

We first get a dataset 1D generator via the `ac.agg.Dataset1D` object, where this `dataset_gen` is a list containing
each dataset that was fitted. This can be useful for reminding ourselves of the datasets we fitted, before plotting 
the fits themselves.
"""
dataset_1d_agg = ac.agg.Dataset1DAgg(aggregator=agg)
dataset_gen = dataset_1d_agg.dataset_list_gen_from()

for dataset_list in dataset_gen:
    print(dataset_list)

    dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0])
    dataset_plotter.subplot_dataset()

"""
We now use the database to load a generator containing a list of the fits of the maximum log likelihood model to 
each dataset.
"""
fit_agg = ac.agg.FitDataset1DAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    fit_plotter = aplt.FitDataset1DPlotter(fit=fit_list[0])
    fit_plotter.subplot_fit()

"""
__Modification__

The `Dataset1DAgg` allow us to modify the fit settings. By default, it uses the `Clocker1D` and `SettingsDataset1D` 
that were used during the model-fit. 

However, we can change these settings such that the fit is performed differently. For example, what if I wanted to see 
how the fit looks with a clocker that uses a different express parameter?

You can do this by passing the objects, which overwrite the ones used by the analysis.
"""
fit_agg = ac.agg.FitDataset1DAgg(
    aggregator=agg,
    clocker_list=[ac.Clocker1D(express=2)] * 4,
)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    fit_plotter = aplt.FitDataset1DPlotter(fit=fit_list[0])
    fit_plotter.subplot_fit()

"""
__Visualization Customization__

The benefit of inspecting fits using the aggregator, rather than the files outputs to the hard-disk, is that we can 
customize the plots using the `MatPlot1D` and `MatPlot2D` objects..

Below, we create a new function to apply as a generator to do this. However, we use a convenience method available 
in the aggregator package to set up the fit.
"""
fit_agg = ac.agg.FitDataset1DAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    mat_plot = aplt.MatPlot1D(
        figure=aplt.Figure(figsize=(12, 12)),
        title=aplt.Title(label="Custom Image", fontsize=24),
        yticks=aplt.YTicks(fontsize=24),
        xticks=aplt.XTicks(fontsize=24),
        cmap=aplt.Cmap(norm="log", vmax=1.0, vmin=1.0),
        colorbar_tickparams=aplt.ColorbarTickParams(labelsize=20),
        units=aplt.Units(in_kpc=True),
    )

    fit_plotter = aplt.FitDataset1DPlotter(fit=fit_list[0], mat_plot_1d=mat_plot)
    fit_plotter.figures_1d(data=True)

"""
Making this plot for a paper? You can output it to hard disk.
"""
fit_agg = ac.agg.FitDataset1DAgg(aggregator=agg)
fit_gen = fit_agg.max_log_likelihood_gen_from()

for fit_list in fit_gen:
    mat_plot = aplt.MatPlot2D(
        title=aplt.Title(label="Hey"),
        output=aplt.Output(
            path=path.join("output", "path", "of", "file"),
            filename="publication",
            format="png",
        ),
    )


"""
The info dictionary we passed is also available.
"""
print("Info:")
info_gen = agg.values("info")
print([info for info in info_gen])

"""
Finished.
"""
