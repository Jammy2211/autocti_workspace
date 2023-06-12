"""
Plots FPA: Database
===================

CTI calibration is typically performed independently on many CCD's over a telescope's focal plane array (FPA).
For example, for Euclid, the FPA is a 6x6 grid of CCDs.

Visualizing the results of a CTI calibration in a way that shows the results across all CCDs in the FPA is
challenging, as there is a lot of information to convey.

Fits to each CCD are also performed independently, meaning that the model-fit of a given CCD does not have any
information on the CTI best-fit of its neighboring CCDs, meaning that visualizing the results of a CTI calibration
across the FPA is not a trivial task.

The `autocti_workspace/*/plot/fpa` package provides tools for simulating an FPA of CTI calibration data, fitting it in
a realistic calibration setting and plotting the results of the fit on a single figure showing the whole FPA via
the database.

__Database__

This script creates an .sqlite database file from the results of the model-fit performed
in the `plot/ccd/dataset_1d/fit.py` script and uses this database to create figures of the fit to the FPA.

If you are not familiar with the database functionality, checkout
the `autocti_workspace/*/dataset_1d/advanced/database` package.
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

The fits performed in the `plot/foa/dataset_1d/fit.py` script output results to hard-disk, in the `output` folder.

The code belows creates a new .sqlite database file from these results, which is used to load the results of the 
model-fit in this example script.
"""
database_name = "plot_fpa"

try:
    os.remove(path.join("output", f"{database_name}.sqlite"))
except FileNotFoundError:
    pass

agg = af.Aggregator.from_database(
    filename=f"{database_name}.sqlite", completed_only=False
)

agg.add_directory(directory=path.join("output", database_name))

"""
__Output__

The image of each fit are output to the directory below, and we update their filenames before making each plot.
"""
mat_plot = aplt.MatPlot1D(
    output=aplt.Output(
        path=path.join("scripts", "plot", "images", "fpa"),
        format="png",
    )
)

"""
__Dataset__

The model-fit was performed to 36 1D datasets consisting of 1 charge injection level across the 6x6 FPA.

We first seek to plot these 36 datasets, on an 6 x 6 matplotlib figure, so we can cleanly see all datasets across
the FPA on at once.

We use the `Dataset1DAgg` object to create a generator of every dataset, which we iterate over to create a 
figure of all 36 datasets via the `Dataset1DPlotter` object.
"""
dataset_1d_agg = ac.agg.Dataset1DAgg(aggregator=agg)
dataset_gen = dataset_1d_agg.dataset_list_gen_from()

dataset_plotter_list = []


for dataset_list in dataset_gen:
    for dataset in dataset_list:
        dataset_plotter_list.append(
            aplt.Dataset1DPlotter(dataset=dataset, mat_plot_1d=mat_plot)
        )

dataset_plotter_list[0].set_filename(filename="dataset_via_database")

multi_plotter = aplt.MultiFigurePlotter(
    plotter_list=dataset_plotter_list, subplot_shape=(6, 6)
)

multi_plotter.subplot_of_figure(
    func_name="figures_1d",
    figure_name="data",
)


"""
The model-fit masked the FPR, and the visualization above therefore does not show the FPR (the values of 0's in its
location indicate it has been masked).

When performing the fit a full dataset was passed to the `Analysis` object for visualization via the `dataset_full`
input. This can also be loaded from the database and plotted, by passing the `Dataset1DAgg` object the input
`use_dataset_full=True.
"""
dataset_1d_agg = ac.agg.Dataset1DAgg(aggregator=agg, use_dataset_full=True)
dataset_gen = dataset_1d_agg.dataset_list_gen_from()

dataset_plotter_list = []

for dataset_list in dataset_gen:
    for dataset in dataset_list:
        dataset_plotter_list.append(
            aplt.Dataset1DPlotter(dataset=dataset, mat_plot_1d=mat_plot)
        )

dataset_plotter_list[0].set_filename(filename="dataset_full_via_database")

multi_plotter = aplt.MultiFigurePlotter(
    plotter_list=dataset_plotter_list, subplot_shape=(6, 6)
)

multi_plotter.subplot_of_figure(
    func_name="figures_1d",
    figure_name="data",
)

"""
Plots of the FPR and EPER regions of the CCD are also available.

We again use the full dataset, to ensure the FPR is plotted and not masked.
"""
for region in ["fpr", "eper"]:
    dataset_gen = dataset_1d_agg.dataset_list_gen_from()

    dataset_plotter_list = []

    for dataset_list in dataset_gen:
        for dataset in dataset_list:
            dataset_plotter_list.append(
                aplt.Dataset1DPlotter(dataset=dataset, mat_plot_1d=mat_plot)
            )

    dataset_plotter_list[0].set_filename(filename=f"dataset_{region}_via_database")

    multi_plotter = aplt.MultiFigurePlotter(
        plotter_list=dataset_plotter_list, subplot_shape=(6, 6)
    )

    multi_plotter.subplot_of_figure(
        func_name="figures_1d", figure_name="data", region=region
    )


"""
__Fits__

Visualization of model fits can be performed using the `FitImaging1DAgg` object and an analogous generator to the
API above.

Below, we produce a subplot of all 32 fits to the full unmasked data, for the FPR and EPER regions.
"""
fit_agg = ac.agg.FitDataset1DAgg(aggregator=agg, use_dataset_full=True)

for region in ["fpr", "eper"]:
    fit_gen = fit_agg.max_log_likelihood_gen_from()

    fit_plotter_list = []

    for fit_list in fit_gen:
        for fit in fit_list:
            fit_plotter_list.append(
                aplt.FitDataset1DPlotter(fit=fit, mat_plot_1d=mat_plot)
            )

    fit_plotter_list[0].set_filename(filename=f"fit_{region}_via_database")

    multi_plotter = aplt.MultiFigurePlotter(
        plotter_list=fit_plotter_list, subplot_shape=(6, 6)
    )

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

aggregator = ClassicAggregator(directory=path.join("output", "plot_fpa"))

fit_agg = ac.agg.FitDataset1DAgg(aggregator=agg, use_dataset_full=True)

for region in ["fpr", "eper"]:
    fit_gen = fit_agg.max_log_likelihood_gen_from()

    fit_plotter_list = []

    for fit_list in fit_gen:
        for fit in fit_list:
            fit_plotter_list.append(
                aplt.FitDataset1DPlotter(fit=fit, mat_plot_1d=mat_plot)
            )

    fit_plotter_list[0].set_filename(filename=f"fit_{region}_via_classic_aggregator")

    multi_plotter = aplt.MultiFigurePlotter(
        plotter_list=fit_plotter_list, subplot_shape=(6, 6)
    )

    multi_plotter.subplot_of_figure(
        func_name="figures_1d", figure_name="data", region=region
    )

# """
# __Max LH Fits__
# """
# ml_instances = [samps.max_log_likelihood() for samps in agg.values("samples")]
#
# fit_list = [ac.FitDataset1D(dataset=)]
