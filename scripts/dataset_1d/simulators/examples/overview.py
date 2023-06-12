"""
Simulator: Dataset 1D With x2 traps
===================================

This script simulates a 1D dataset with CTI, where:

 - CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the direction uses the `CCD` class.

__Start Here Notebook__

If any code in this script is unclear, refer to the simulators `start_here.ipynb` notebook for more detailed comments.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import json
from os import path
import autocti as ac
import autocti.plot as aplt

"""
__Dataset Paths__

The path where the dataset will be output.
"""
dataset_type = "dataset_1d"
dataset_name = "overview"
dataset_path = path.join("dataset", dataset_name, dataset_type)

"""
__Layout__

The 1D shape of each data.
"""
shape_native = (30,)

"""
The locations (using NumPy array indexes) of the prescan and overscan on the data.

For the fiducial 1D dataset the prescan spans the first 10 pixels and overscan the last 10 pixels.
"""
prescan = ac.Region1D((0, 1))
overscan = ac.Region1D((25, 30))

"""
Specify the regions of the dataset where charge was present before CTI, called the First Pixel Response (FPR). 

For the fiducial 1D dataset this is 10 pixels after the prescan.
"""
region_list = [(1, 25)]

"""
The normalization of the charge region (e.g. the FPR) of every dataset.
"""
norm_list = [100, 5000, 25000, 200000]

"""
The `Layout1D` object for every 1D dataset, which is used for generating the simulation data.
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
__Clocker__

The `Clocker1D` object models the read-out process of every 1D dataset as if it were clocked out on a real CCD. This 
includes the addition of CTI. 
"""
clocker = ac.Clocker1D(express=5, roe=ac.ROEChargeInjection())

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the simulated data, which contains: 

 - 1 `TrapInstantCapture` species.
 - A simple CCDPhase volume filling parametrization.
"""
trap_0 = ac.TrapInstantCapture(density=10.0, release_timescale=5.0)
trap_list = [trap_0]

ccd = ac.CCDPhase(well_fill_power=0.5, well_notch_depth=0.0, full_well_depth=200000.0)

cti = ac.CTI1D(trap_list=trap_list, ccd=ccd)


"""
__Simulate__

To simulate data including CTI, we pass the 1D dataset layout to a `SimulatorDataset1D`, which adds CTI via arCTIc and 
read-noise to the data.
"""
simulator_list = [
    ac.SimulatorDataset1D(read_noise=1.0, pixel_scales=0.1, norm=norm)
    for norm in norm_list
]

"""
We now pass each layout to the simulator. This creates a list of instances of the `Dataset1D` class, which 
include the data (with CTI), noise-maps and the pre-cti data.
"""
dataset_list = [
    simulator.via_layout_from(clocker=clocker, layout=layout, cti=cti)
    for layout, simulator in zip(layout_list, simulator_list)
]

dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0])
dataset_plotter.subplot_dataset()

"""
__Output__

Output a subplot of the data, noise-map and pre CTI image to .png files.
"""
for dataset, norm in zip(dataset_list, norm_list):
    filename = f"dataset_1d_{int(norm)}"

    output_path = path.join(dataset_path, f"norm_{int(norm)}")

    mat_plot = aplt.MatPlot1D(
        output=aplt.Output(path=output_path, filename=filename, format="png")
    )

    dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset, mat_plot_1d=mat_plot)
    dataset_plotter.subplot_dataset()

"""
Output the data, noise-map and pre CTI image of the charge injection dataset to .fits files.
"""
[
    dataset.output_to_fits(
        data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
        noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
        pre_cti_data_path=path.join(
            dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
        ),
        overwrite=True,
    )
    for dataset, norm in zip(dataset_list, norm_list)
]

"""
__True Likelihood__

Fit the true model to the data and output the true `log_likelihood`, which can act as a verification of the quality of
model fits.
"""
fit_list = []

for dataset in dataset_list:
    post_cti_data = clocker.add_cti(data=dataset.pre_cti_data, cti=cti)

    fit_list.append(ac.FitDataset1D(dataset=dataset, post_cti_data=post_cti_data))

true_log_likelihood_list = [fit.log_likelihood for fit in fit_list]
true_log_likelihood = sum(true_log_likelihood_list)

true_log_likelihood_file = path.join(dataset_path, "true_log_likelihood.json")

with open(true_log_likelihood_file, "w+") as f:
    json.dump(true_log_likelihood_list, f, indent=4)
    json.dump(true_log_likelihood, f, indent=4)

"""
Finish.
"""
