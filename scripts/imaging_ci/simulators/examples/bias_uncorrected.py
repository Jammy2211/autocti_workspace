"""
Simulator: Bias Uncorrected
===========================

The data is used to illustrate the bias subtraction functionality illustrated
in `preprocess_optional_bias_subtraction.py`.

The method to simulate and add bias is therefore very simplistic and not necessarily that realistic. We simply add
a constant value of 2000e- to the data after the simulation procedure is complete.

__Model__

This script simulates charge injection imaging with CTI, where:

 - Parallel CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the parallel direction using the `CCD` class.

__Start Here Notebook__

If any code in this script is unclear, refer to the simulators `start_here.ipynb` notebook for more detailed comments.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autocti as ac
import autocti.plot as aplt

"""
__Dataset Paths__

The path where the dataset will be output.
"""
dataset_type = "imaging_ci"
dataset_name = "bias_uncorrected"

"""
Returns the path where the dataset will be output, which in this case is
'/autocti_workspace/dataset/imaging_ci/misc/bias_uncorrected'
"""
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
__Layout__

The 2D shape of the image.
"""
shape_native = (2000, 100)

"""
The locations (using NumPy array indexes) of the parallel overscan, serial prescan and serial overscan on the image.
"""
parallel_overscan = ac.Region2D((1980, 2000, 5, 95))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, 95, 100))

"""
Specify the charge injection regions on the CCD, which in this case is 5 equally spaced rectangular blocks.
"""
region_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

"""
The normalization of every charge injection image, which determines how many images are simulated.
"""
norm_list = [100, 5000, 25000, 200000]

"""
The total number of charge injection images that are simulated.
"""
total_datasets = len(norm_list)

"""
Create the layout of the charge injection pattern for every charge injection normalization.
"""
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

"""
__Clocker__

The `Clocker` models the CCD read-out, including CTI. 

For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""
clocker = ac.Clocker2D(
    parallel_express=5, parallel_roe=ac.ROEChargeInjection(), parallel_fast_mode=True
)

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

 - 2 `TrapInstantCapture` species in the parallel direction, which captures electrons during clocking instantly and 
 release them according to an exponential probability distribution defined by a single release times.
 - A simple CCDPhase volume filling parametrization.
"""
parallel_trap_0 = ac.TrapInstantCapture(density=1.0, release_timescale=5.0)
parallel_trap_list = [parallel_trap_0]

parallel_ccd = ac.CCDPhase(
    well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
)

cti = ac.CTI2D(parallel_trap_list=parallel_trap_list, parallel_ccd=parallel_ccd)

"""
__Simulate__

To simulate charge injection imaging, we pass the charge injection pattern to a `SimulatorImagingCI`, which adds CTI 
via arCTIc and read-noise to the data.

This creates instances of the `ImagingCI` class, which include the images, noise-maps and pre_cti_data images.
"""
simulator_list = [
    ac.SimulatorImagingCI(read_noise=4.0, pixel_scales=0.1, norm=norm)
    for norm in norm_list
]

"""
We now pass each charge injection pattern to the simulator. This generate the charge injection image of each exposure
and before passing each image to arCTIc does the following:

 - Uses an input read-out electronics corner to perform all rotations of the image before / after adding CTI.
 - Stores this corner so that if we output the files to .fits,they are output in their original and true orientation.
 - Includes information on the different scan regions of the image, such as the serial prescan and serial overscan.
"""
dataset_list = [
    simulator.via_layout_from(clocker=clocker, layout=layout, cti=cti)
    for layout, simulator in zip(layout_list, simulator_list)
]

"""
__Add Bias__

Manually add 2000e- to every pixel, in order to simulate the effect of an unsubtracted bias.
"""
for index in range(len(dataset_list)):
    dataset_list[index].data += 2000.0

"""
__Output__

Output subplots of the simulated dataset to the dataset path as .png files.
"""
for dataset, norm in zip(dataset_list, norm_list):
    output = aplt.Output(
        path=path.join(dataset_path, f"norm_{int(norm)}"),
        filename="imaging_ci",
        format="png",
    )

    mat_plot = aplt.MatPlot2D(output=output)

    dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset, mat_plot_2d=mat_plot)
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

    dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset, mat_plot_1d=mat_plot)
    dataset_plotter.figures_1d(region="parallel_fpr", data=True, data_logy=True)
    dataset_plotter.figures_1d(region="parallel_eper", data=True, data_logy=True)

"""
Output the image, noise-map and pre CTI image of the charge injection dataset to .fits files.
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

Save the `Clocker2D` and `CTI2D` in the dataset folder as a .json file, ensuring the true traps and CCD settings are 
safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `CTI2D.from_json`.
"""
cti.output_to_json(file_path=path.join(dataset_path, "cti.json"))
clocker.output_to_json(file_path=path.join(dataset_path, "clocker.json"))

"""
Finished.
"""
