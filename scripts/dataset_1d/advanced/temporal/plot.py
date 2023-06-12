"""
Temporal: Plot
==============

The example script `fit.py` fits multiple 1D CTI calibration datasets, representative of data taken over the course
of a space mission where radiation damage increases therefore also increasing the level of CTI.

This script loads the model-fitting results into an .sqlite database file and performs visualization of the results
as a function of time.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import os
import autofit as af
import autocti as ac
import autocti.plot as aplt

"""
__Building a Database File From an Output Folder__

The fits performed in the `advanced/temporal/fits.py` script output results to hard-disk, in the `output` folder.

The code belows creates a new .sqlite database file from these results, which is used to load the results of the 
model-fit in this example script.
"""
database_name = "temporal"

try:
    os.remove(path.join("output", f"{database_name}.sqlite"))
except FileNotFoundError:
    pass

agg = af.Aggregator.from_database(
    filename=f"{database_name}.sqlite", completed_only=False
)

agg.add_directory(directory=path.join("output", database_name))

"""
__Density Versus Time__

A plot of trap density versus time informs us of how much CTI increases due to radiation damage over the course of a
space mission. 

We produce this plot by extracting the trap density from the median PDF model, with errors at the 3.0 sigma confidence
level. We also extract the time at which each dataset was acquired.
"""
mp_instances_list = [samps.median_pdf() for samps in agg.values("samples")]
density_mp_list = [instance.cti.trap_list[0].density for instance in mp_instances_list]

ue3_instance_list = [
    samps.errors_at_upper_sigma(sigma=3.0) for samps in agg.values("samples")
]
density_ue3_list = [instance.cti.trap_list[0].density for instance in ue3_instance_list]

le3_instance_list = [
    samps.errors_at_lower_sigma(sigma=3.0) for samps in agg.values("samples")
]
density_le3_list = [instance.cti.trap_list[0].density for instance in le3_instance_list]

time_list = [instance.time for instance in mp_instances_list]

mat_plot = aplt.MatPlot1D(
    output=aplt.Output(
        path=path.join("scripts", "dataset_1d", "advanced", "temporal", "images"),
        format="png",
    )
)

from autoarray.plot.auto_labels import AutoLabels

mat_plot_1d.plot_yx(
    y=density_mp_list,
    x=time_list,
    plot_axis_type_override="errorbar",
    visuals_1d=aplt.Visuals1D(),
    y_errors=[density_le3_list, density_ue3_list],
    auto_labels=AutoLabels(
        title=f"Density vs Time",
        yunit="",
        filename=f"density_versus_time",
    ),
)
