"""
Simulator: Uniform Charge Injection With Cosmic Rays
====================================================

__Model__

This script simulates charge injection imaging with CTI, where:

 - Parallel CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the parallel direction using the `CCD` class.

__Start Here Notebook__

If any code in this script is unclear, refer to the `simulators/start_here.ipynb` notebook.
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

The 'dataset_name' describes the type of data being simulated (in this case, imaging data) and 'dataset_name' 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to '/autocti_workspace/dataset/dataset_name/dataset_name/image.fits'.
 - The noise-map will be output to '/autocti_workspace/dataset/dataset_name/dataset_name/noise_map.fits'.
 - The pre_cti_data will be output to '/autocti_workspace/dataset/dataset_name/dataset_name/pre_cti_data.fits'.
"""
dataset_name = "overview"
dataset_type = "uniform"

"""
Returns the path where the dataset will be output, which in this case is
'/autocti_workspace/dataset/imaging_ci/overview/uniform'
"""
dataset_path = path.join("dataset", dataset_name, "imaging_ci", dataset_type)

"""
__Layout__

The 2D shape of the image.
"""
shape_native = (2066, 2128)

"""
The locations (using NumPy array indexes) of the parallel overscan, serial prescan and serial overscan on the image.
"""
parallel_overscan = ac.Region2D((2108, 2128, 51, 2099))
serial_prescan = ac.Region2D((0, 2128, 0, 51))
serial_overscan = ac.Region2D((0, 2128, 2099, 2128))

"""
Specify the charge injection regions on the CCD, which in this case is 5 equally spaced rectangular blocks.
"""
region_list = [
    (100, 300, serial_prescan[3], serial_overscan[2]),
    (500, 700, serial_prescan[3], serial_overscan[2]),
    (900, 1100, serial_prescan[3], serial_overscan[2]),
    (1300, 1500, serial_prescan[3], serial_overscan[2]),
    (1700, 1900, serial_prescan[3], serial_overscan[2]),
]

"""
The normalization of every charge injection image, which determines how many images are simulated.
"""
norm_list = [100]

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
    parallel_express=5,
    parallel_roe=ac.ROEChargeInjection(),
    parallel_fast_mode=True,
    serial_express=5,
    serial_roe=ac.ROE(),
)

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

 - 1 `TrapInstantCapture` species in the parallel and serial directions, which captures electrons during clocking 
 instantly and release them according to an exponential probability distribution defined by a single release times.
 
 - A simple CCDPhase volume filling parametrization for parallel and serial clocking separately.
"""
parallel_trap_0 = ac.TrapInstantCapture(density=20.0, release_timescale=5.0)
parallel_trap_list = [parallel_trap_0]

parallel_ccd = ac.CCDPhase(
    well_fill_power=0.5, well_notch_depth=0.0, full_well_depth=200000.0
)

serial_trap_0 = ac.TrapInstantCapture(density=20.0, release_timescale=10.0)
serial_trap_list = [serial_trap_0]

serial_ccd = ac.CCDPhase(
    well_fill_power=0.75, well_notch_depth=0.0, full_well_depth=200000.0
)

cti = ac.CTI2D(
    parallel_trap_list=parallel_trap_list,
    parallel_ccd=parallel_ccd,
    serial_trap_list=serial_trap_list,
    serial_ccd=serial_ccd,
)

"""
__Simulate__

To simulate charge injection imaging, we pass the charge injection pattern to a `SimulatorImagingCI`, which adds CTI 
via arCTIc and read-noise to the data.

This creates instances of the `ImagingCI` class, which include the images, noise-maps and pre_cti_data images.
"""
simulator_list = [
    ac.SimulatorImagingCI(read_noise=1.0, pixel_scales=0.1, norm=norm)
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
40
ac.output_to_json(
    obj=clocker,
    file_path=path.join(dataset_path, "clocker.json"),
)

"""
Finished.
"""
