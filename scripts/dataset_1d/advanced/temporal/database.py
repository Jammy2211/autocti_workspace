"""
Temporal: Database
==================

The example script `fit.py` fits multiple 1D CTI calibration datasets, representative of data taken over the course
of a space mission where radiation damage increases therefore also increasing the level of CTI.

This script loads the model-fitting results into an .sqlite database file and performs interpolation on the results
to determine the evolution of CTI with time.
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

from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(
    directory=path.join("output", "temporal"), completed_only=False
)
# agg.add_directory(directory=path.join("output", database_name))

"""
__Instances__

Interpolation uses the maximum log likelihood model of each fit to build an interpolation model of the CTI as a
function of time. 

We therefore first create a list of instances of these maximum log likelihood models via the database.
"""
ml_instances_list = [samps.max_log_likelihood() for samps in agg.values("samples")]

"""
__Interpolation__

We use the `ml_instances_list` to build an interpolation model of the CTI as a function of time.

This is performed using the `LinearInterpolator` object, which interpolates the CTI model parameters as a function of
time linearly between the values computed by the model-fits above.

More advanced interpolation schemes are available and described in the `interpolation.py` example.
"""
interpolator = af.LinearInterpolator(instances=ml_instances_list)

"""
The model can be interpolated to any time, for example time=1.5.

This returns a new `instance` of the CTI model, as an instance of the `CTI1D` object, where the parameters are computed 
by interpolating between the values computed above.
"""
instance = interpolator[interpolator.time == 1.5]

"""
The `density` of the `TrapInstantCapture` at time 1.5 is between the value inferred for the first and second fits taken
at times 1.0 and 2.0.
"""
print(f"Trap density of fit 1 (t = 1): {ml_instances_list[0].cti.trap_list[0].density}")
print(f"Trap density of fit 2 (t = 2): {ml_instances_list[1].cti.trap_list[0].density}")

print(f"Trap Density interpolated at t = 1.5 {instance.cti.trap_list[0].density}")

"""
Finish.
"""
