"""
Plots FPA: Simulator
====================

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

This script simulates the FPA of CTI calibration data that is fitted and plotted in the other modules,
 where `ImagingCI` objects are used to make fitting run times fast and use of hard-disk space efficient.

__Simulation__

A total of 36 1D datasets are simulated, which represent a 6x6 FPA of data. For simplicity and efficient run times,
only a single charge injection level is simulated for each CCD, but in a realistic calibration dataset each CCD would
have data at multiple charge injection levels.

This is representative of a real CTI calibration dataset from Euclid, whose FPA consists of 36 CCDs in a 6x6 grid.

__Model__

This script simulates a 1D dataset with CTI, where:

 - CTI is added to the image using a 1 `Trap` species model.
 - The volume filling behaviour in the direction uses the `CCD` class.
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

The steps below repeat those found in the `imaging_ci/simulators/start_here.py` script, so refer to comments in that
script for a description of the code below.

The two things to take note of are:

 - That the `norm_list` consists of 1 entry, meaning that 1 charge injection levels is simulated per CCD for efficiency.
 - That two for loops over range 6 and 6 is used to simulate 36 CCDs, representing a 6x6 FPA of data.
"""
dataset_type = "imaging_ci"

shape_native = (100, 100)

parallel_overscan = ac.Region2D((95, 100, 5, 95))
serial_prescan = ac.Region2D((0, 100, 0, 5))
serial_overscan = ac.Region2D((0, 95, 95, 100))

region_list = [
    (5, 25, serial_prescan[3], serial_overscan[2]),
]

norm_list = [1000, 10000]
total_datasets = len(norm_list)

for fpa_i in range(6):
    for fpa_j in range(6):
        dataset_name = f"data_fpa_{fpa_i}_{fpa_j}"
        dataset_path = path.join("dataset", dataset_type, "fpa_plot", dataset_name)

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

            dataset_plotter = aplt.ImagingCIPlotter(
                dataset=dataset, mat_plot_2d=mat_plot
            )
            dataset_plotter.subplot_dataset()

        for dataset, norm in zip(dataset_list, norm_list):
            output = aplt.Output(
                path=path.join(dataset_path, f"norm_{int(norm)}", "binned_1d"),
                format="png",
            )

            mat_plot = aplt.MatPlot1D(output=output)

            dataset_plotter = aplt.ImagingCIPlotter(
                dataset=dataset, mat_plot_1d=mat_plot
            )
            dataset_plotter.figures_1d(region="parallel_fpr", data=True, data_logy=True)
            dataset_plotter.figures_1d(
                region="parallel_eper", data=True, data_logy=True
            )

        40
        ac.output_to_json(
            obj=clocker,
            file_path=path.join(dataset_path, "clocker.json"),
        )

"""
Finish.
"""
