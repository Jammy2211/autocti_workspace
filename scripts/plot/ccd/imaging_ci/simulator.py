"""
Plots CCD: Simulator
====================

CTI calibration is typically performed on a CCD using many images (8-32 or more), where the images vary in the level of
charge injected into the CCD (the charge injection level).

Visualizing the results of a CTI calibration in a way that shows the results across all injection levels is
challenging, as there is a lot of information to convey.

The `autocti_workspace/*/plot/ccd` package provides tools for simulating CTI calibration data, fitting it in a
realistic calibration setting and plotting the results of the fit.

This script simulates the CTI calibration data that is fitted and plotted in the other modules, where `ImagingCI`
objects are used, which makes the fits slower but is representative of real CTI calibration data (e.g. Euclid).

__Simulation__

A total of 32 2D charge injection datasets are simulated, which are grouped in 4 sets of 8 different charge injection
levels.

This is representative of a real CTI calibration dataset from Euclid, where each CCD has four quadrants and each
quadrant is exposed at 8 different charge injection levels.

__Model__

This script simulates a 2D dataset with CTI, where:

 - Parallel CTI is added to the image using a 1 `Trap` species model.
 - The parallel volume filling behaviour in the direction uses the `CCD` class.

Serial CTI is omitted.
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
__Simulation__

The steps below repeat those found in the `dataset/simulators/start_here.py` script, so refer to comments in that
script for a description of the code below.

The two things to take note of are:

 - That the `norm_list` consists of 8 entries, meaning that 8 different charge injection levels are simulated.
 - That a for loop over range 4 is used to simulate 4 quadrants of data, where each quadrant is simulated at one of 
   the 8 charge injection levels.
"""
dataset_type = "imaging_ci"

shape_native = (100, 100)

parallel_overscan = ac.Region2D((95, 100, 5, 95))
serial_prescan = ac.Region2D((0, 100, 0, 5))
serial_overscan = ac.Region2D((0, 95, 95, 100))

region_list = [
    (5, 25, serial_prescan[3], serial_overscan[2]),
]

norm_list = [100, 500, 1000, 5000, 10000, 25000, 100000, 200000]
total_datasets = len(norm_list)

for quadrant_id in range(4):
    dataset_name = f"data_quad_{quadrant_id}"
    dataset_path = path.join("dataset", dataset_type, "ccd_plot", dataset_name)

    layout_list = [
        ac.Layout2DCI(
            shape_2d=shape_native,
            region_list=region_list,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
        )
        for i in range(total_datasets)
    ]

    clocker = ac.Clocker2D(
        parallel_express=5,
        parallel_roe=ac.ROEChargeInjection(),
    )
    parallel_trap_0 = ac.TrapInstantCapture(density=0.5, release_timescale=5.0)
    parallel_trap_list = [parallel_trap_0]

    parallel_ccd = ac.CCDPhase(
        well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
    )

    cti = ac.CTI2D(parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd)

    simulator_list = [
        ac.SimulatorImagingCI(read_noise=0.1, pixel_scales=0.1, norm=norm)
        for norm in norm_list
    ]

    dataset_list = [
        simulator.via_layout_from(clocker=clocker, layout=layout, cti=cti)
        for layout, simulator in zip(layout_list, simulator_list)
    ]

    dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset_list[0])
    dataset_plotter.subplot_dataset()

    [
        dataset.output_to_fits(
            data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
            noise_map_path=path.join(
                dataset_path, f"norm_{int(norm)}", "noise_map.fits"
            ),
            pre_cti_data_path=path.join(
                dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
            ),
            overwrite=True,
        )
        for dataset, norm in zip(dataset_list, norm_list)
    ]

    for dataset, norm in zip(dataset_list, norm_list):
        output = aplt.Output(
            path=path.join(dataset_path, f"norm_{int(norm)}"),
            filename="imaging_ci",
            format="png",
        )

        mat_plot = aplt.MatPlot2D(output=output)

        dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset, mat_plot_2d=mat_plot)
        dataset_plotter.subplot_dataset()

    for dataset, norm in zip(dataset_list, norm_list):
        output = aplt.Output(
            path=path.join(dataset_path, f"norm_{int(norm)}", "binned_1d"), format="png"
        )

        mat_plot = aplt.MatPlot1D(output=output)

        dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset, mat_plot_1d=mat_plot)
        dataset_plotter.figures_1d(region="parallel_fpr", data=True, data_logy=True)
        dataset_plotter.figures_1d(region="parallel_eper", data=True, data_logy=True)

    cti.output_to_json(file_path=path.join(dataset_path, "cti.json"))
    clocker.output_to_json(file_path=path.join(dataset_path, "clocker.json"))

"""
Finish.
"""
