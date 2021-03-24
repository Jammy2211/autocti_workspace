"""
Simulator: Non-Uniform Charge Injection With Cosmic Rays
========================================================

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

"""
The 'dataset_label' describes the type of data being simulated (in this case, imaging data) and 'dataset_name' 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/image.fits'.
 - The noise-map will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/noise_map.fits'.
 - The ci_pre_cti will be output to '/autocti_workspace/dataset/dataset_label/dataset_name/ci_pre_cti.fits'.
"""
dataset_type = "ci_imaging"
dataset_label = "non_uniform_cosmic_rays"
dataset_name = "parallel_x1"
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

"""
The 2D shape of the image.
"""
shape_native = (2000, 100)

"""
The locations of the prescans and overscans on the image.
"""
scans = ac.Scans(
    parallel_overscan=ac.Region2D((1980, 2000, 5, 95)),
    serial_prescan=ac.Region2D((0, 2000, 0, 5)),
    serial_overscan=ac.Region2D((0, 1980, 95, 100)),
)

"""
Specify the charge injection regions on the CCD, which in this case is 3 equally spaced rectangular blocks.
"""
ci_regions = [
    (0, 200, scans.serial_prescan[3], scans.serial_overscan[2]),
    (400, 600, scans.serial_prescan[3], scans.serial_overscan[2]),
    (800, 1000, scans.serial_prescan[3], scans.serial_overscan[2]),
    (1200, 1400, scans.serial_prescan[3], scans.serial_overscan[2]),
    (1600, 1800, scans.serial_prescan[3], scans.serial_overscan[2]),
]


"""
The normalization of every charge injection image, which determines how many images are simulated.
"""
normalizations = [100, 5000, 25000, 84700]

"""
The `Clocker` models the CCD read-out, including CTI. 

For parallel clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""
clocker = ac.Clocker(parallel_express=2, parallel_charge_injection_mode=True)

"""
The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

 - 1 `Trap` species in the parallel direction.
 - A simple CCD volume beta parametrization.
"""
parallel_trap = ac.TrapInstantCapture(density=0.5, release_timescale=4.0)
parallel_ccd = ac.CCD(well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700)

"""
Use the charge injection normalizations and regions to create *CIPatternNonUniform* of every image we'll simulate.
"""
column_sigmas = [100.0] * len(normalizations)
row_slopes = [0.0] * len(normalizations)
ci_patterns = [
    ac.ci.CIPatternNonUniform(
        normalization=normalization, regions=ci_regions, row_slope=row_slope
    )
    for (normalization, row_slope) in zip(normalizations, row_slopes)
]

"""
To simulate charge injection imaging, we pass the charge injection pattern to a *SimulatorCIImaging*, which adds CTI 
via arCTIc and read-noise to the data.

This creates instances of the *CIImaging* class, which include the images, noise-maps and ci_pre_cti images.
"""
simulator = ac.ci.SimulatorCIImaging(
    shape_native=shape_native, read_noise=4.0, scans=scans, pixel_scales=0.1
)

"""
We use the LA Cosmic algorithm to simulate and add cosmic rays to our ci pre cti image. This routine randomly
generates cosmimc rays based on realistic cosmic ray rates expected. These cosmic rays will then be added to our
ci pre-cti image in the simulaate function below, and subject to CTI according to the CTI model.
"""
cosmic_path = path.join("scripts", "cosmic_rays")

cosmic_ray_maker = cosmics.CosmicRays(
    shape=shape_native,
    cr_fluxscaling=1.0,
    cr_length_file=f"{cosmic_path}crlength_v2.fits",
    cr_distance_file=f"{cosmic_path}crdist.fits",
    log=logger,
)
cosmic_ray_maker.set_ifiles()
cosmic_ray_maps = list(
    map(
        lambda i: cosmic_ray_maker.drawEventsToCoveringFactor(
            limit=clocker.parallel.well_depth
        ),
        range(len(normalizations)),
    )
)

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
        parallel_traps=[parallel_trap],
        parallel_ccd=parallel_ccd,
        cosmic_ray_map=cosmic_ray_map,
    )
    for ci_pattern, cosmic_ray_map in zip(ci_patterns, cosmic_ray_maps)
]

"""
Finally output the image, noise-map and pre cti image of the charge injection dataset to .fits files.
"""
[
    ci_dataset.output_to_fits(
        image_path=path.join(
            dataset_path, f"image_{int(ci_dataset.ci_pattern.normalization)}"
        ),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{int(ci_dataset.ci_pattern.normalization)}"
        ),
        ci_pre_cti_path=path.join(
            dataset_path, f"ci_pre_cti_{int(ci_dataset.ci_pattern.normalization)}"
        ),
    )
    for ci_dataset in ci_datasets
]

"""
Finished.
"""
