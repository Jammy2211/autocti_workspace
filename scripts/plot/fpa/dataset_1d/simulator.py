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
 where `Dataset1D` objects are used to make fitting run times fast and use of hard-disk space efficient.

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

The steps below repeat those found in the `dataset_1d/simulators/start_here.py` script, so refer to comments in that
script for a description of the code below.

The two things to take note of are:

 - That the `norm_list` consists of 1 entry, meaning that 1 charge injection levels is simulated per CCD for efficiency.
 - That two for loops over range 6 and 6 is used to simulate 36 CCDs, representing a 6x6 FPA of data.
"""
dataset_type = "dataset_1d"

shape_native = (21,)

prescan = ac.Region1D((0, 1))
overscan = ac.Region1D((20, 21))

region_list = [(1, 10)]

norm_list = [1000]

for fpa_i in range(6):
    for fpa_j in range(6):
        dataset_name = f"data_fpa_{fpa_i}_{fpa_j}"
        dataset_path = path.join("dataset", dataset_type, "fpa_plot", dataset_name)

        layout_list = [
            ac.Layout1D(
                shape_1d=shape_native,
                region_list=region_list,
                prescan=prescan,
                overscan=overscan,
            )
            for norm in norm_list
        ]

        clocker = ac.Clocker1D(express=5)

        trap_0 = ac.TrapInstantCapture(density=0.5, release_timescale=5.0)
        trap_list = [trap_0]

        ccd = ac.CCDPhase(
            well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
        )

        cti = ac.CTI1D(trap_list=trap_list, ccd=ccd)

        simulator_list = [
            ac.SimulatorDataset1D(read_noise=0.01, pixel_scales=0.1, norm=norm)
            for norm in norm_list
        ]

        dataset_list = [
            simulator.via_layout_from(clocker=clocker, layout=layout, cti=cti)
            for layout, simulator in zip(layout_list, simulator_list)
        ]

        dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0])
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
                filename="dataset_1d",
                format="png",
            )

            mat_plot = aplt.MatPlot1D(output=output)

            dataset_plotter = aplt.Dataset1DPlotter(
                dataset=dataset, mat_plot_1d=mat_plot
            )
            dataset_plotter.subplot_dataset()

        for dataset, norm in zip(dataset_list, norm_list):
            output = aplt.Output(
                path=path.join(dataset_path, f"norm_{int(norm)}", "binned_1d"),
                format="png",
            )

            mat_plot = aplt.MatPlot1D(output=output)

            dataset_plotter = aplt.Dataset1DPlotter(
                dataset=dataset, mat_plot_1d=mat_plot
            )
            dataset_plotter.figures_1d(region="fpr", data=True, data_logy=True)
            dataset_plotter.figures_1d(region="eper", data=True, data_logy=True)

        40
        ac.output_to_json(
            obj=clocker,
            file_path=path.join(dataset_path, "clocker.json"),
        )

        fit_list = []

        for dataset in dataset_list:
            post_cti_data = clocker.add_cti(data=dataset.pre_cti_data, cti=cti)

            fit_list.append(
                ac.FitDataset1D(dataset=dataset, post_cti_data=post_cti_data)
            )

        true_log_likelihood_list = [fit.log_likelihood for fit in fit_list]
        true_log_likelihood = sum(true_log_likelihood_list)

        true_log_likelihood_file = path.join(dataset_path, "true_log_likelihood.json")

        with open(true_log_likelihood_file, "w+") as f:
            json.dump(true_log_likelihood_list, f, indent=4)
            json.dump(true_log_likelihood, f, indent=4)

"""
Finish.
"""
