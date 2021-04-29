"""
Simulator: Uniform Charge Injection With Cosmic Rays
====================================================

This script simulates charge injection imaging with CTI, where:

 - Parallel CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the parallle direction using the `CCD` class.
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
The 'dataset_label' describes the type of data being simulated (in this case, imaging data) and 'dataset_name' 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/image.fits'.
 - The noise-map will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/noise_map.fits'.
 - The pre_cti_image will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/pre_cti_image.fits'.
"""
dataset_type = "imaging_ci"
dataset_label = "uniform"
dataset_name = "parallel_x2"

"""
Returns the path where the dataset will be output, which in this case is
'/autocti_workspace/dataset/imaging_ci/uniform/parallel_x2'
"""
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

"""
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
regions_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

"""
The normalization of every charge injection image, which determines how many images are simulated.
"""
normalization_list = [100, 5000, 25000, 84700]

"""
Create the layout of the charge injection pattern for every charge injection normalization.
"""
layout_list = [
    ac.ci.Layout2DCIUniform(
        shape_2d=(2000, 100),
        region_list=regions_list,
        normalization=normalization,
        parallel_overscan=parallel_overscan,
        serial_prescan=serial_prescan,
        serial_overscan=serial_overscan,
    )
    for normalization in normalization_list
]

"""
The `Clocker` models the CCD read-out, including CTI. 

For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""
clocker = ac.Clocker(parallel_express=2, parallel_charge_injection_mode=True)

"""
The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

 - 2 `Trap` species in the parallel direction.
 - A simple CCD volume beta parametrization.
"""
parallel_trap_0 = ac.TrapInstantCapture(density=0.13, release_timescale=1.25)
parallel_trap_1 = ac.TrapInstantCapture(density=0.25, release_timescale=4.4)
parallel_ccd = ac.CCD(well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700)

"""
To simulate charge injection imaging, we pass the charge injection pattern to a `SimulatorImagingCI`, which adds CTI 
via arCTIc and read-noise to the data.

This creates instances of the `ImagingCI` class, which include the images, noise-maps and pre_cti_image images.
"""
simulator = ac.ci.SimulatorImagingCI(read_noise=4.0, pixel_scales=0.1)

"""
We now pass each charge injection pattern to the simulator. This generate the charge injection image of each exposure
and before passing each image to arCTIc does the following:

 - Uses an input read-out electronics corner to perform all rotations of the image before / after adding CTI.
 - Stores this corner so that if we output the files to .fits,they are output in their original and true orientation.
 - Includes information on the different scan regions of the image, such as the serial prescan and serial overscan.
"""
dataset_ci_list = [
    simulator.from_layout(
        clocker=clocker,
        layout=layout_ci,
        parallel_traps=[parallel_trap_0, parallel_trap_1],
        parallel_ccd=parallel_ccd,
    )
    for layout_ci in layout_list
]

"""
Output a subplot of the simulated dataset to the dataset path as .png files.
"""
mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

imaging_ci_plotter = aplt.ImagingCIPlotter(
    imaging=dataset_ci_list[0], mat_plot_2d=mat_plot_2d
)
imaging_ci_plotter.subplot_imaging_ci()

"""
Finally output the image, noise-map and pre cti image of the charge injection dataset to .fits files.
"""
[
    dataset_ci.output_to_fits(
        image_path=path.join(
            dataset_path, f"image_{int(dataset_ci.pattern_ci.normalization)}.fits"
        ),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{int(dataset_ci.pattern_ci.normalization)}.fits"
        ),
        pre_cti_image_path=path.join(
            dataset_path,
            f"pre_cti_image_{int(dataset_ci.pattern_ci.normalization)}.fits",
        ),
    )
    for dataset_ci in dataset_ci_list
]

"""
Finished.
"""
