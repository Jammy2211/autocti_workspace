import autofit as af
import autocti as ac

"""
This script simulates charge injection imaging with CTI, where:

 - Serial CTI is added to the image using a 1 *Trap* species model.
 - The volume filling behaviour in the serial direction using the *CCD* class.
"""


"""
The 'dataset_label' describes the type of data being simulated (in this case, imaging data) and 'dataset_name' 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/image.fits'.
 - The noise-map will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/noise_map.fits'.
 - The ci_pre_cti will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/ci_pre_cti.fits'.
"""
dataset_type = "ci_imaging"
dataset_label = "non_uniform"
dataset_name = "serial_x1"

"""
Returns the path where the dataset will be output, which in this case is
'/autocti_workspace/dataset/ci_images_non_uniform/serial_x1'
"""
dataset_path = f"dataset/{dataset_type}/{dataset_label}/{dataset_name}"

"""The 2D shape of the image"""
shape_2d = (100, 2000)

"""The locations of the overscans on the image."""
scans = ac.Scans(
    parallel_overscan=None,
    serial_prescan=ac.Region((0, 100, 0, 20)),
    serial_overscan=ac.Region((0, 100, 1980, 2000)),
)

"""Specify the charge injection regions on the CCD, which in this case is the full CCD to maximize the serial EPERs."""
ci_regions = [(0, 10, scans.serial_prescan[3], scans.serial_overscan[2])]

"""The normalization of every charge injection image - this list determines how many images are simulated."""
normalizations = [100.0, 5000.0, 25000.0, 84700.0]

"""
The `Clocker` models the CCD read-out, including CTI. 
"""
clocker = ac.Clocker(serial_express=2)

"""
The CTI model used by arCTIc to add CTI to the input image in the serial direction, which contains: 

 - 1 *Trap* species in the serial direction.
 - A simple CCD volume beta parametrization.
"""
serial_trap = ac.TrapInstantCapture(density=0.5, release_timescale=4.0)
serial_ccd = ac.CCD(well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700)

"""Use the charge injection normalizations and regions to create *CIPatternNonUniform* of every image we'll simulate."""
column_sigmas = [100.0] * len(normalizations)
row_slopes = [0.0] * len(normalizations)
ci_patterns = [
    ac.ci.CIPatternNonUniform(
        normalization=normalization,
        regions=ci_regions,
        column_sigma=column_sigma,
        row_slope=row_slope,
    )
    for normalization, column_sigma, row_slope in zip(
        normalizations, column_sigmas, row_slopes
    )
]


"""
To simulate charge injection imaging, we pass the charge injection pattern to a *SimulatorCIImaging*, which adds CTI 
via arCTIc and read-noise to the data.

This creates instances of the *CIImaging* class, which include the images, noise-maps and ci_pre_cti images.
"""
simulator = ac.ci.SimulatorCIImaging(shape_2d=shape_2d, read_noise=4.0, scans=scans)

"""
We now pass each charge injection pattern to the simulator. This generate the charge injection image of each exposure
and before passing each image to arCTIc does the following:

 - Uses an input read-out electronics corner to perform all rotations of the image before / after adding CTI.
 - Stores this corner so that if we output the files to .fits,they are output in their original and true orientation.
 - Includes information on the different scan regions of the image, such as the serial prescan and overscans.
"""
ci_datasets = [
    simulator.from_ci_pattern(
        clocker=clocker,
        ci_pattern=ci_pattern,
        serial_traps=[serial_trap],
        serial_ccd=serial_ccd,
    )
    for ci_pattern in ci_patterns
]

"""
Finally output the image, noise-map and pre cti image of the charge injection dataset to .fits files.
"""
[
    ci_dataset.output_to_fits(
        image_path=f"{dataset_path}/image_{int(ci_dataset.ci_pattern.normalization)}",
        noise_map_path=f"{dataset_path}/noise_map_{int(ci_dataset.ci_pattern.normalization)}",
        ci_pre_cti_path=f"{dataset_path}/ci_pre_cti_{int(ci_dataset.ci_pattern.normalization)}",
    )
    for ci_dataset in ci_datasets
]
