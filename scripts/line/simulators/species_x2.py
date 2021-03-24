"""
Simulator: Line With x2 Traps
=============================

This script simulates a `Line` with CTI, where:

 - CTI is added to the image using a 2 `Trap` species model.
 - The volume filling behaviour in the direction uses the `CCD` class.
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

 - The image will be output to '/autocti_workspace/dataset/dataset_name/image.fits'.
 - The noise-map will be output to '/autocti_workspace/dataset/dataset_name/noise_map.fits'.
 - The line_pre_cti will be output to '/autocti_workspace/dataset/dataset_name/line_pre_cti.fits'.
"""
dataset_type = "line"
dataset_name = "species_x2"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
The 1D shape of the `Line`.
"""
shape_native = (200, 1)

"""
The locations of the prescans and overscans on the `Line`.
"""
scans = ac.Scans(
    serial_prescan=ac.Region2D((0, 10, 0, 1)),
    serial_overscan=ac.Region2D((190, 200, 0, 1)),
)

"""
Specify the regions on the `Line` where charge was present before CTI, 10 pixels after the serial prescan.
"""
regions = [(10, 20, 0, 1)]

"""
The normalization of every `Line`, this list determines how many `Line`'s are simulated.
"""
normalizations = [100, 500, 1000, 5000, 10000, 25000, 50000, 84700]

"""
The `Clocker` models the `Line` read-out, including CTI. 
"""
clocker = ac.Clocker(parallel_express=2)

"""
The CTI model used by arCTIc to add CTI to the input `Line`, which contains: 

 - 2 `Trap` species.
 - A simple CCD volume beta parametrization.
"""
trap_0 = ac.TrapInstantCapture(density=0.13, release_timescale=1.25)
trap_1 = ac.TrapInstantCapture(density=0.25, release_timescale=4.4)
ccd = ac.CCD(well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700)

"""
Use the `Line` normalizations and regions to create `LinePattern` of every image we'll simulate.
"""
line_pattern_list = [
    ac.ci.CIPatternUniform(normalization=normalization, regions=regions)
    for normalization in normalizations
]

"""
To simulate charge injection imaging, we pass theline pattern to a `SimulatorLine`, which adds CTI via arCTIc and 
read-noise to the data.

This creates instances of the `Line` class, which include the images, noise-maps and line_pre_cti images.
"""
simulator = ac.ci.SimulatorCIImaging(
    shape_native=shape_native, read_noise=0.001, scans=scans, pixel_scales=0.1
)

"""
We now pass each line pattern to the simulator. This generate the image of each line before passing them to arCTIc 
to add CTI performs the following:
"""
line_dataset_list = [
    simulator.from_ci_pattern(
        clocker=clocker,
        ci_pattern=ci_pattern,
        parallel_traps=[trap_0, trap_1],
        parallel_ccd=ccd,
    )
    for ci_pattern in line_pattern_list
]

ci_imaging_plotter = aplt.CIImagingPlotter(imaging=line_dataset_list[0])
ci_imaging_plotter.subplot_ci_imaging()

"""
Finally output the `Line`, noise-map and pre cti image of the charge injection dataset to .fits files.
"""
[
    line_dataset.output_to_fits(
        image_path=path.join(
            dataset_path, f"image_{line_dataset.ci_pattern.normalization}.fits"
        ),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{line_dataset.ci_pattern.normalization}.fits"
        ),
        ci_pre_cti_path=path.join(
            dataset_path, f"line_pre_cti_{line_dataset.ci_pattern.normalization}.fits"
        ),
    )
    for line_dataset in line_dataset_list
]
