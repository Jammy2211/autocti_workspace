"""
Simulator: Line With x2 Trap Continuum
======================================

This script simulates a `Line` with CTI, where:

 - CTI is added to the image using a 1 `Trap` species model where the lifetimes of traps following a continuum.
 - The volume filling behaviour uses the `CCD` class.
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

The 'dataset_label' describes the type of data being simulated (in this case, imaging data) and 'dataset_name' 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to '/autocti_workspace/dataset/dataset_name/image.fits'.
 - The noise-map will be output to '/autocti_workspace/dataset/dataset_name/noise_map.fits'.
 - The pre_cti_line will be output to '/autocti_workspace/dataset/dataset_name/pre_cti_line.fits'.
"""
dataset_type = "line"
dataset_name = "species_x2_continuum_0"
dataset_path = path.join("dataset", dataset_type, dataset_name)

"""
The 1D shape of the `Line`.
"""
shape_native = (200, 1)

"""
The locations of the serial prescan and serial overscan on the `Line`.
"""
layout_list = ac.Scans(
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
normalization_list = [100, 500, 1000, 5000, 10000, 25000, 50000, 84700]

"""
The `Clocker` models the `Line` read-out, including CTI. 
"""
clocker = ac.Clocker(parallel_express=2)

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the input `Line`, which contains: 

 - 2 `Trap` species.
 - A simple CCD volume beta parametrization.
"""
# dataset_name = "species_x2_continuum_0"
# trap_0 = ac.TrapLogNormalLifetimeContinuum(
#     density=1.0, release_timescale_mu=1.0, release_timescale_sigma=0.1
# )
# trap_1 = ac.TrapLogNormalLifetimeContinuum(
#     density=1.0, release_timescale_mu=5.0, release_timescale_sigma=0.1
# )

# dataset_name = "species_x2_continuum_1"
# trap_0 = ac.TrapLogNormalLifetimeContinuum(
#     density=1.0, release_timescale_mu=5.0, release_timescale_sigma=0.1
# )
# trap_1 = ac.TrapLogNormalLifetimeContinuum(
#     density=1.0, release_timescale_mu=20.0, release_timescale_sigma=0.1
# )

dataset_name = "species_x2_continuum_2"
trap_0 = ac.TrapLogNormalLifetimeContinuum(
    density=1.0, release_timescale_mu=1.0, release_timescale_sigma=0.1
)
trap_1 = ac.TrapLogNormalLifetimeContinuum(
    density=1.0, release_timescale_mu=20.0, release_timescale_sigma=0.1
)

dataset_path = path.join("dataset", dataset_type, dataset_name)

ccd = ac.CCDPhase(well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700.0)

"""
Use the `Line` normalization_list and regions to create `LinePattern` of every image we'll simulate.
"""
line_pattern_list = [
    ac.ci.PatternCIUniform(normalization=normalization, regions=regions)
    for normalization in normalization_list
]

"""
__Simulate__

To simulate charge injection imaging, we pass theline pattern to a `SimulatorLine`, which adds CTI via arCTIc and 
read-noise to the data.

This creates instances of the `Line` class, which include the images, noise-maps and pre_cti_line images.
"""
simulator = ac.ci.SimulatorImagingCI(
    shape_native=shape_native,
    read_noise=0.001,
    layout_list=layout_list,
    pixel_scales=0.1,
)

"""
We now pass each line pattern to the simulator. This generate the image of each line before passing them to arCTIc 
to add CTI performs the following:
"""
line_dataset_list = [
    simulator.from_pattern_ci(
        clocker=clocker,
        pattern_ci=pattern_ci,
        parallel_traps=[trap_0, trap_1],
        parallel_ccd=ccd,
    )
    for pattern_ci in line_pattern_list
]

imaging_ci_plotter = aplt.ImagingCIPlotter(imaging=line_dataset_list[0])
imaging_ci_plotter.subplot_imaging_ci()

"""
__Output__

Output the `Line`, noise-map and pre cti image of the charge injection dataset to .fits files.
"""
[
    line_dataset.output_to_fits(
        image_path=path.join(
            dataset_path, f"image_{line_dataset.pattern_ci.normalization}.fits"
        ),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{line_dataset.pattern_ci.normalization}.fits"
        ),
        pre_cti_image_path=path.join(
            dataset_path, f"pre_cti_line_{line_dataset.pattern_ci.normalization}.fits"
        ),
    )
    for line_dataset in line_dataset_list
]
