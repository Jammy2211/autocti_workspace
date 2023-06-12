"""
Plots CCD: Database
===================

CTI calibration is typically performed on a CCD using many images (8-32 or more), where the images vary in the level of
charge injected into the CCD (the charge injection level).

Visualizing the results of a CTI calibration in a way that shows the results across all injection levels is
challenging, as there is a lot of information to convey.

The `autocti_workspace/*/plot/ccd` package provides tools for simulating CTI calibration data, fitting it in a 
realistic calibration setting and plotting the results of the fit.

__Database__

This script creates an .sqlite database file from the results of the model-fit performed
in the `plot/ccd/imaging_ci/fit.py` script and uses this database to create figures of the fit.

If you are not familiar with the database functionality, checkout
the `autocti_workspace/*/imaging_ci/advanced/database` package.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import copy
import os
from os import path
import autofit as af
import autocti as ac
import autocti.plot as aplt


"""
__Building a Database File From an Output Folder__

The fits performed in the `plot/ccd/imaging_ci/fit.py` script output results to hard-disk, in the `output` folder.

The code belows creates a new .sqlite database file from these results, which is used to load the results of the 
model-fit in this example script.
"""
database_name = "plot_ccd_imaging_ci"

try:
    os.remove(path.join("output", f"{database_name}.sqlite"))
except FileNotFoundError:
    pass

agg = af.Aggregator.from_database(
    filename=f"{database_name}.sqlite", completed_only=False, top_level_only=True
)

agg.add_directory(directory=path.join("output", "plot_ccd", "imaging_ci"))

"""
__Output__

The image of each fit are output to the directory below, and we update their filenames before making each plot.
"""
mat_plot = aplt.MatPlot1D(
    output=aplt.Output(
        path=path.join("scripts", "plot", "images", "ccd", "imaging_ci"),
        format="png",
    )
)

"""
__Dataset__

The model-fit was performed to 32 1D datasets consisting of 8 charge injection levels across 4 quadrants on each
CCD. 

We first seek to plot these 32 datasets, on an 8 x 4 matplotlib figure, so we can cleanly see them all at once.

We use the `ImagingCIAgg` object to create a generator of every dataset, which we iterate over to create a 
figure of all 32 datasets via the `ImagingCIPlotter` object.
"""
dataset_agg = ac.agg.ImagingCIAgg(aggregator=agg)
dataset_gen = dataset_agg.dataset_list_gen_from()

dataset_plotter_list = []

for dataset_list in dataset_gen:
    for dataset in dataset_list:
        dataset_plotter_list.append(
            aplt.ImagingCIPlotter(dataset=dataset, mat_plot_1d=mat_plot)
        )

dataset_plotter_list[0].set_filename(filename="dataset_via_database")

multi_plotter = aplt.MultiFigurePlotter(plotter_list=dataset_plotter_list)

multi_plotter.subplot_of_figure(
    func_name="figures_1d", figure_name="data", region="parallel_eper"
)


"""
The model-fit masked the FPR, and the visualization above therefore does not show the FPR (the values of 0's in its
location indicate it has been masked).

When performing the fit a full dataset was passed to the `Analysis` object for visualization via the `dataset_full`
input. This can also be loaded from the database and plotted, by passing the `ImagingCIAgg` object the input
`use_dataset_full=True.
"""
dataset_agg = ac.agg.ImagingCIAgg(aggregator=agg, use_dataset_full=True)
dataset_gen = dataset_agg.dataset_list_gen_from()

dataset_plotter_list = []

for dataset_list in dataset_gen:
    for dataset in dataset_list:
        dataset_plotter_list.append(
            aplt.ImagingCIPlotter(dataset=dataset, mat_plot_1d=mat_plot)
        )

dataset_plotter_list[0].set_filename(filename="dataset_full_via_database")

multi_plotter = aplt.MultiFigurePlotter(plotter_list=dataset_plotter_list)

multi_plotter.subplot_of_figure(
    func_name="figures_1d", figure_name="data", region="parallel_fpr"
)


"""
__Fits__

Visualization of model fits can be performed using the `FitImagingCIAgg` object and an analogous generator to the
API above.

Below, we produce a subplot of all 32 fits to the full unmasked data, for the FPR and EPER regions.
"""
fit_agg = ac.agg.FitImagingCIAgg(aggregator=agg, use_dataset_full=True)

for region in ["parallel_fpr", "parallel_eper"]:
    fit_gen = fit_agg.max_log_likelihood_gen_from()

    fit_plotter_list = []

    for fit_list in fit_gen:
        for fit in fit_list:
            fit_plotter_list.append(
                aplt.FitImagingCIPlotter(fit=fit, mat_plot_1d=mat_plot)
            )

    fit_plotter_list[0].set_filename(filename=f"fit_{region}_via_database")

    multi_plotter = aplt.MultiFigurePlotter(plotter_list=fit_plotter_list)

    multi_plotter.subplot_of_figure(
        func_name="figures_1d", figure_name="data", region=region
    )

"""
__Classic Aggregator__

The tasks above can also use an older implementation of the aggregator, which is now deprecated but still supported.

This is used for CTI calibration in the Euclid coding framework (which does not support sqlite and thus has to
use this implementation).

The code below is purely for testing / legacy purposes and probably not of interest to most users.
"""
from autofit.aggregator import Aggregator as ClassicAggregator

aggregator = ClassicAggregator(directory=path.join("output", "plot_ccd"))

fit_agg = ac.agg.FitImagingCIAgg(aggregator=agg, use_dataset_full=True)

for region in ["parallel_fpr", "parallel_eper", "serial_fpr", "serial_eper"]:
    fit_gen = fit_agg.max_log_likelihood_gen_from()

    fit_plotter_list = []

    for fit_list in fit_gen:
        for fit in fit_list:
            fit_plotter_list.append(
                aplt.FitImagingCIPlotter(fit=fit, mat_plot_1d=mat_plot)
            )

    fit_plotter_list[0].set_filename(filename=f"fit_{region}_via_classic_aggregator")

    multi_plotter = aplt.MultiFigurePlotter(plotter_list=fit_plotter_list)

    multi_plotter.subplot_of_figure(
        func_name="figures_1d", figure_name="data", region=region
    )

# """
# __Max LH Fits__
# """
# ml_instances = [samps.max_log_likelihood() for samps in agg.values("samples")]
#
# fit_list = [ac.FitImagingCI(dataset=)]
