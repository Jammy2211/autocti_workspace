"""
Simulator: 1D Data With x3 traps
================================

This script simulates a 1D dataset with CTI, where:

 - CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the direction uses the `CCD` class.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.
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
dataset_name = "species_x3"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Layout__

The 1D shape of each data.
"""
shape_native = (2000,)

"""
The locations (using NumPy array indexes) of the prescan and overscan on the data.

For the fiducial 1D dataset the prescan spans the first 10 pixels and overscan the last 10 pixels.
"""
prescan = ac.Region1D((0, 10))
overscan = ac.Region1D((1990, 2000))

"""
Specify the regions of the dataset where charge was present before CTI, called the First Pixel Response (FPR). 

For the fiducial 1D dataset this is 10 pixels after the prescan.
"""
region_list = [
    (10, 40),
    (310, 340),
    (610, 640),
    (910, 940),
    (1210, 1240),
    (1510, 1540),
]

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
clocker = ac.Clocker1D(express=5)

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the simulated data, which contains: 

 - 3 `TrapInstantCapture` species.
 
 - A simple CCDPhase volume filling parametrization.
"""
trap_0 = ac.TrapInstantCapture(density=0.07275, release_timescale=0.8)
trap_1 = ac.TrapInstantCapture(density=0.21825, release_timescale=4.0)
trap_2 = ac.TrapInstantCapture(density=6.54804, release_timescale=20.0)

trap_list = [trap_0, trap_1, trap_2]

ccd = ac.CCDPhase(well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0)

cti = ac.CTI1D(trap_list=trap_list, ccd=ccd)

"""
__Simulate__

To simulate data including CTI, we pass the 1D dataset layout to a `SimulatorDataset1D`, which adds CTI via arCTIc and 
read-noise to the data.
"""
simulator_list = [
    ac.SimulatorDataset1D(read_noise=4.0, pixel_scales=0.1, norm=norm)
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
    output = aplt.Output(
        path=path.join(dataset_path, f"norm_{int(norm)}"),
        filename="dataset_1d",
        format="png",
    )

    mat_plot = aplt.MatPlot1D(output=output)

    dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset, mat_plot_1d=mat_plot)
    dataset_plotter.subplot_dataset()

"""
Output plots of the EPER and FPR's binned up in 1D, so that electron capture and trailing can be
seen clearly.
"""
for dataset, norm in zip(dataset_list, norm_list):
    output = aplt.Output(
        path=path.join(dataset_path, f"norm_{int(norm)}", "binned_1d"), format="png"
    )

    mat_plot = aplt.MatPlot1D(output=output)

    dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset, mat_plot_1d=mat_plot)
    dataset_plotter.figures_1d(region="fpr", data=True, data_logy=True)
    dataset_plotter.figures_1d(region="eper", data=True, data_logy=True)

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
__CTI json__

Save the `Clocker1D` and `CTI1D` in the dataset folder as a .json file, ensuring the true traps and CCD settings 
are safely stored  and available to check how the dataset was simulated in the future. 

This can be loaded via the methods `cti = ac.from_json()` and `clocker = ac.from_json()`.
"""
40
ac.output_to_json(
    obj=clocker,
    file_path=path.join(dataset_path, "clocker.json"),
)

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
