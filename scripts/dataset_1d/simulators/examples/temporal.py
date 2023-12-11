"""
Simulator: Temporal
===================

This script simulates multiple 1D CTI calibration datasets, representative of data taken over the course of a space
mission where radiation damage increases therefore also increasing the level of CTI.

This simulated data is used to illustrate the temporal fitting, where a CTI model is fitted to each dataset individually
and can then be interpolated to estimate the CTI at any point in time for the correction of science data.

__Model__

This script simulates multiple 1D CTI calibration datasets with CTI, where:

 - CTI is added to the data using a 1 `Trap` species model.
 - The volume filling behaviour uses the `CCD` class.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autocti as ac
import autocti.plot as aplt

"""
__Dataset Paths__

The path where the dataset will be output.
"""
dataset_type = "dataset_1d"
dataset_label = "temporal"
dataset_path = path.join("dataset", dataset_type, dataset_label)

"""
__Layout__

The 1D shape of each data.
"""
shape_native = (200,)

"""
The locations (using NumPy array indexes) of the prescan and overscan on the data.

For the fiducial 1D dataset the prescan spans the first 10 pixels and overscan the last 10 pixels.
"""
prescan = ac.Region1D((0, 10))
overscan = ac.Region1D((190, 200))

"""
Specify the regions of the dataset where charge was present before CTI, called the First Pixel Response (FPR). 

For the fiducial 1D dataset this is 10 pixels after the prescan.
"""
region_list = [(10, 20)]

"""
The normalization of the charge region (e.g. the FPR) of every dataset.
"""
norm_list = [100, 5000, 25000, 200000]

"""
The total number of charge injection datas that are simulated.
"""
total_datasets = len(norm_list)

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
__Base CTI Model__

The CTI model used by arCTIc to add CTI to the input data, which contains: 

 - 1 `TrapInstantCapture` species.

 - A simple CCDPhase volume filling parametrization.
 
__Temporal CTI Model__

We will create 5 realisations of the above model, corresponding to CTI calibration data taken at five equally spaced
intervals of the space mission.

The density of the trap species for each dataset is computed via a linear relation between time and density, where:

 y = mx + c
 
 x = time
 m = density evolution
 c = density at mission start
 y = density at a given time
"""

density_evolution = 0.2
density_start = 1.0

time_list = range(0, 5)

for time in time_list:
    """
    __Density at Time__

    Compute the density of the trap from the linear relation defining its time evolution.
    """
    density = float((density_evolution * time) + density_start)

    trap_0 = ac.TrapInstantCapture(density=density, release_timescale=5.0)
    trap_list = [trap_0]

    ccd = ac.CCDPhase(
        well_fill_power=0.5, well_notch_depth=0.0, full_well_depth=200000.0
    )

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

    """
    __Output__
    
    We output each simulated dataset to a folder based on its number of times.
    
    Output a subplot of the simulated dataset to the dataset path as .png files.
    """
    dataset_time = f"time_{time}"
    dataset_output_path = path.join(dataset_path, dataset_time)

    mat_plot = aplt.MatPlot1D(
        output=aplt.Output(path=dataset_output_path, format="png")
    )

    for dataset, norm in zip(dataset_list, norm_list):
        output = aplt.Output(
            path=path.join(dataset_output_path, f"norm_{int(norm)}"),
            filename="dataset",
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
            path=path.join(dataset_output_path, f"norm_{int(norm)}", "binned_1d"),
            format="png",
        )

        mat_plot = aplt.MatPlot1D(output=output)

        dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset, mat_plot_1d=mat_plot)
        dataset_plotter.figures_1d(region="fpr", data=True)
        dataset_plotter.figures_1d(region="eper", data=True)

    """
    Output the data, noise-map and pre CTI data of the charge injection dataset to .fits files.
    """
    [
        dataset.output_to_fits(
            data_path=path.join(dataset_output_path, f"norm_{int(norm)}", "data.fits"),
            noise_map_path=path.join(
                dataset_output_path, f"norm_{int(norm)}", "noise_map.fits"
            ),
            pre_cti_data_path=path.join(
                dataset_output_path, f"norm_{int(norm)}", "pre_cti_data.fits"
            ),
            overwrite=True,
        )
        for dataset, norm in zip(dataset_list, norm_list)
    ]

    """
    Save the `TrapInstantCapture` in the dataset folder as a .json file, ensuring the true densities
    are safely stored and available to check how the dataset was simulated in the future. 

    This can be loaded via the method `TrapInstantCapture.from_json`.
    """
    ac.output_to_json(
        obj=cti,
        file_path=path.join(dataset_output_path, "cti.json"),
    )
    ac.output_to_json(
        obj=clocker,
        file_path=path.join(dataset_output_path, "clocker.json"),
    )

"""
Finished.
"""
