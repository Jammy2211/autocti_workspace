"""
__Transdimensional Pipelines__

This transdimensional pipeline runner fits `CIImaging` to calibrate serial CTI where:

 - The CTI model consists of an input number of serial `Trap` species.
 - The `CCD` volume fill parameterization is an input. .
 - Effects in the `CIImaging` such as cosmic rays, non-uniform injections can be includded or omitted.

"""

"""
Load the charge injection dataset 'ci_imaging/uniform/serial_x2' 'from .fits files, which is the dataset we will
use to perform CTI modeling.

This is the same dataset we fitted in the 'autocti/intro/fitting.py' example.
"""

import autofit as af
import autocti as ac
import autocti.plot as aplt

dataset_name = "serial_x2"
dataset_path = f"dataset/ci_imaging/uniform/{dataset_name}"

"""The locations of the overscans on the image, which is used to visualize the `CIImaging` during the model-fit."""

scans = ac.Scans(
    parallel_overscan=ac.Region((1980, 2000, 5, 95)),
    serial_prescan=ac.Region((0, 2000, 0, 5)),
    serial_overscan=ac.Region((0, 1980, 95, 100)),
)

"""Specify the charge injection regions on the CCD, which in this case is 3 equally spaced rectangular blocks."""

ci_regions = [
    (0, 450, scans.serial_prescan[3], scans.serial_overscan[2]),
    (650, 1100, scans.serial_prescan[3], scans.serial_overscan[2]),
    (1300, 1750, scans.serial_prescan[3], scans.serial_overscan[2]),
]

"""
We require the normalization of every charge injection image, as the names of the files are tagged with the charge
injection normalization level.
"""

normalizations = [100.0, 5000.0, 25000.0, 84700.0]

"""Use the charge injection normalizations and regions to create `CIPatternUniform` of every image we'll fit. The
`CIPattern` is used for visualizing the model-fit."""

ci_patterns = [
    ac.ci.CIPatternUniform(normalization=normalization, regions=ci_regions)
    for normalization in normalizations
]

"""
We can now load every image, noise-map and pre-CTI charge injection image as instances of the `CIImaging` object.
"""

imagings = [
    ac.ci.CIImaging.from_fits(
        image_path=f"{dataset_path}/image_{int(pattern.normalization)}.fits",
        noise_map_path=f"{dataset_path}/noise_map_{int(pattern.normalization)}.fits",
        ci_pre_cti_path=f"{dataset_path}/ci_pre_cti_{int(pattern.normalization)}.fits",
        pixel_scales=0.1,
        ci_pattern=pattern,
        roe_corner=(1, 0),
    )
    for pattern in ci_patterns
]

"""Lets plot the first `CIImaging` with its `Mask2D`"""

aplt.Imaging.subplot_imaging(imaging=imagings[0])

"""
The `Clocker` models the CCD read-out, including CTI. 

For serial clocking, we use 'charge injection mode' which transfers the charge of every pixel over the full CCD.
"""

clocker = ac.Clocker(serial_express=2, serial_charge_injection_mode=True)

"""
__Settings__

Next, we specify the *SettingsPhaseCIImaging*, which describes how the model is fitted to the data in the 
log likelihood function. Below, we specify:

Different `SettingsPhase` are used in different example model scripts and a full description of all `SettingsPhase`
can be found in the example script 'autoccti_workspace/examples/model/customize/settings.py' and the following 
link -> <link>
"""

settings = ac.SettingsPhaseCIImaging()

"""
__Pipeline_Setup__:

Pipelines use `Setup` objects to customize how different aspects of the model are fitted. 

First, we create a `Setup`, which customizes:

 - The number of serial `Trap` species and how they are modeled.
 - The `CCD` volumen filling parameterization.
"""

# serial_trap_0 = af.PriorModel(ac.TrapInstantCapture)
# serial_trap_1 = af.PriorModel(ac.TrapInstantCapture)
# serial_traps = [serial_trap_0, serial_trap_1]
total_serial_traps = 2
serial_ccd = af.PriorModel(ac.CCD)
serial_ccd.well_notch_depth = 0.0
serial_ccd.full_well_depth = 84700

"""
_Pipeline Tagging_

The `Setup` objects are input into a `SetupPipeline` object, which is passed into the pipeline and used to customize
the analysis depending on the setup. This includes tagging the output path of a pipeline. For example, if 
`no_serial_traps` is True, the pipeline`s output paths are `tagged` with the string `serial_traps_x2`.

This means you can run the same pipeline on the same data twice (e.g. with different numbers of serial traps) and the 
results will go to different output folders and thus not clash with one another!

The `path_prefix` below specifies the path the pipeline results are written to, which is:

 `autocti_workspace/output/transdimensional/dataset_name` 
 `autocti_workspace/output/transdimensional/serial_x2`
"""

"""
__Pipeline Creation__

To create a pipeline we import it from the pipelines folder and run its `make_pipeline` function, inputting the 
`Setup` and `SettingsPhase` above.
"""

from pipelines import serial

pipeline = serial.make_pipeline(
    setup=None,
    settings=settings,
    total_serial_traps=total_serial_traps,
    serial_ccd=serial_ccd,
)

"""
__Pipeline Run__

Running a pipeline is the same as running a phase, we simply pass it our lens dataset and mask to its run function.
"""

pipeline.run(datasets=imagings, clocker=clocker)
