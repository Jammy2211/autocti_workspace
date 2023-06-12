"""
Extract
-------

**PyAutoCTI** has numerous methods for extracting subsets of data from a charge injection dataset.

This script illustrates all the available options.

This script assumes familiaring with the **PyAutoCTI** API, if anything is unclear check out other scripts throughout
the workspace.
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
__Dataset + Layout__

We begin by loading a charge injection image which has parallel and serial CTI added, which we will use to illustrate
the extraction methods.

We also set up its corresponding `Layout2DCI` object, which is used to perform extractions.

You should be familiar with the **PyAutoCTI** API below, if not check out other scripts throughout the workspace.
"""
dataset_name = "parallel_x2__serial_x2"
dataset_name = "simple"
dataset_path = path.join("dataset", "imaging_ci", dataset_name)

shape_native = (2000, 100)

"""
The overscans input here are used to perform extractions.
"""
parallel_overscan = ac.Region2D((1980, 2000, 5, 95))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, 95, 100))

"""
These charge injection regions are also used to perform extractions.
"""
region_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

norm = 100

"""
The layout object contains all the extraction functionality used in this example.
"""
layout = ac.Layout2DCI(
    shape_2d=shape_native,
    region_list=region_list,
    parallel_overscan=parallel_overscan,
    serial_prescan=serial_prescan,
    serial_overscan=serial_overscan,
)

"""
We will demonstrate extraction on a charge injection image, but the functionality demonstrated here can be used
on any 2D array (e.g. the noise-map, a CTI corrected image, a residual-map of a fit).
"""
data = ac.Array2D.from_fits(
    file_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
    pixel_scales=0.1,
)

"""
__Parallel EPERs__

The 2D region of every set of parallel EPERs on the charge injection data is contained in the layout's extract
object.

To compute the parallel EPER regions, we specify the number of rows we want each the EPERs to span, using the
`pixels` input. 

For example, by inputting `pixels=(0, 30)` each parallel EPER region contains 30 pixel rows.

The returned shape of each region is (90, 30), where the 90 is the number of columns the parallel EPER spans (as 
defined by the  extent `serial_prescan[3]:serial_overscan[2]` in the `region_list` above).
"""
region_2d_parallel_eper_list = layout.extract.parallel_eper.region_list_from(
    settings=ac.SettingsExtract(pixels=(0, 30))
)
print(region_2d_parallel_eper_list)

"""
The parallel EPER regions above are used to extract the parallel EPERs from the charge injection image. 

The function below returns a list of 2D arrays containing each EPER.

We again specify the number of pixel rows each 2D parallel EPER that is extracted spans. 
"""
data_parallel_eper_list = layout.extract.parallel_eper.array_2d_list_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_parallel_eper_list[0])
array_2d_plotter.figure_2d()

"""
A stacked 2D array of the list of 2D arrays returned above can be extracted.

The stacking process reduces noise in the data, making the EPERs higher signal to noise.
"""
data_parallel_eper = layout.extract.parallel_eper.stacked_array_2d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_parallel_eper)
array_2d_plotter.figure_2d()

"""
A binned 1D array can be extracted. 

This first computes the stacked 2D array above and then bins the data over the serial direction in a single high
signal to noise parallel EPER.
"""
image_1d_parallel_eper = layout.extract.parallel_eper.binned_array_1d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_1d_plotter = aplt.Array1DPlotter(y=image_1d_parallel_eper)
array_1d_plotter.figure_1d()

"""
Negative inputs to the `pixels` tuple are supported in all of the above methods, which extract additional rows in 
front of the parallel EPERs.
"""
image_1d_parallel_eper = layout.extract.parallel_eper.binned_array_1d_from(
    array=data, settings=ac.SettingsExtract(pixels=(-10, 10))
)

array_1d_plotter = aplt.Array1DPlotter(y=image_1d_parallel_eper)
array_1d_plotter.figure_1d()


"""
__Parallel FPRs__

All of the above methods can be used extract, stack and bin the parallel FPR in an analogous fashion.
"""
region_2d_parallel_fpr_list = layout.extract.parallel_fpr.region_list_from(
    settings=ac.SettingsExtract(pixels=(0, 30))
)
print(region_2d_parallel_fpr_list)

data_parallel_fpr_list = layout.extract.parallel_fpr.array_2d_list_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_parallel_fpr_list[0])
array_2d_plotter.figure_2d()

data_parallel_fpr = layout.extract.parallel_fpr.stacked_array_2d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_parallel_fpr)
array_2d_plotter.figure_2d()

data_parallel_fpr = layout.extract.parallel_fpr.stacked_array_2d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_parallel_fpr)
array_2d_plotter.figure_2d()

image_1d_parallel_fpr = layout.extract.parallel_fpr.binned_array_1d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_1d_plotter = aplt.Array1DPlotter(y=image_1d_parallel_fpr)
array_1d_plotter.figure_1d()

"""
__Serial EPERs__

Extract methods for the serial EPERs are available, which again behave analogous to those above.

The only differences are:

- The `pixels` input now corresponds to the number of columns over which the EPERs are extracted. 
- When binning data to 1D, this is performed over the rows of the data in order to create a single serial EPER.
"""
region_2d_serial_eper_list = layout.extract.serial_eper.region_list_from(
    settings=ac.SettingsExtract(pixels=(0, 30))
)
print(region_2d_serial_eper_list)

data_serial_eper_list = layout.extract.serial_eper.array_2d_list_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_serial_eper_list[0])
array_2d_plotter.figure_2d()

data_serial_eper = layout.extract.serial_eper.stacked_array_2d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_serial_eper)
array_2d_plotter.figure_2d()

data_serial_eper = layout.extract.serial_eper.stacked_array_2d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_serial_eper)
array_2d_plotter.figure_2d()

image_1d_serial_eper = layout.extract.serial_eper.binned_array_1d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_1d_plotter = aplt.Array1DPlotter(y=image_1d_serial_eper)
array_1d_plotter.figure_1d()

"""
__Serial FPR__

Serial FPRs can also be extracted, with the behaviour hopefully self explanatory by now.
"""
region_2d_serial_fpr_list = layout.extract.serial_fpr.region_list_from(
    settings=ac.SettingsExtract(pixels=(0, 30))
)
print(region_2d_serial_fpr_list)

data_serial_fpr_list = layout.extract.serial_fpr.array_2d_list_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_serial_fpr_list[0])
array_2d_plotter.figure_2d()

data_serial_fpr = layout.extract.serial_fpr.stacked_array_2d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_serial_fpr)
array_2d_plotter.figure_2d()

data_serial_fpr = layout.extract.serial_fpr.stacked_array_2d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_serial_fpr)
array_2d_plotter.figure_2d()

image_1d_serial_fpr = layout.extract.serial_fpr.binned_array_1d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_1d_plotter = aplt.Array1DPlotter(y=image_1d_serial_fpr)
array_1d_plotter.figure_1d()


"""
__Parallel and Serial Overscans__

There are analogous extract methods for the parallel and serial overscans, which can be used for:

 - Extracting EPERs in science imaging data, where electrons are trailed into the overscan from the background sky
 of the observation. This data can be used to validate the CTI correction.
 
 - Extracting EPERs in flat field data, which again have electrons trailed into the overscan at the edge of the CCD.
 
The overscans only contain one region, however we mimick the API of the methods above. This means that the functions:

 - `array_2d_list_from` always returns a list with a single entry, corresponding to the full overscan region.
 - `stacked_array_2d_from` always returns the array in the list above, with no stacking process taken.
"""
region_2d_parallel_overscan_list = layout.extract.parallel_overscan.region_list_from(
    settings=ac.SettingsExtract(pixels=(0, 30))
)
print(region_2d_parallel_overscan_list)

data_parallel_overscan_list = layout.extract.parallel_overscan.array_2d_list_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_parallel_overscan_list[0])
array_2d_plotter.figure_2d()

data_parallel_overscan = layout.extract.parallel_overscan.stacked_array_2d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_parallel_overscan)
array_2d_plotter.figure_2d()

data_parallel_overscan = layout.extract.parallel_overscan.stacked_array_2d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_parallel_overscan)
array_2d_plotter.figure_2d()

image_1d_parallel_overscan = layout.extract.parallel_overscan.binned_array_1d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_1d_plotter = aplt.Array1DPlotter(y=image_1d_parallel_overscan)
array_1d_plotter.figure_1d()

"""
Equivalent methods are available for serial overscans.
"""
region_2d_serial_overscan_list = layout.extract.serial_overscan.region_list_from(
    settings=ac.SettingsExtract(pixels=(0, 30))
)
print(region_2d_serial_overscan_list)

data_serial_overscan_list = layout.extract.serial_overscan.array_2d_list_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_serial_overscan_list[0])
array_2d_plotter.figure_2d()

data_serial_overscan = layout.extract.serial_overscan.stacked_array_2d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_serial_overscan)
array_2d_plotter.figure_2d()

data_serial_overscan = layout.extract.serial_overscan.stacked_array_2d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_2d_plotter = aplt.Array2DPlotter(array=data_serial_overscan)
array_2d_plotter.figure_2d()

image_1d_serial_overscan = layout.extract.serial_overscan.binned_array_1d_from(
    array=data, settings=ac.SettingsExtract(pixels=(0, 30))
)

array_1d_plotter = aplt.Array1DPlotter(y=image_1d_serial_overscan)
array_1d_plotter.figure_1d()

"""
__Dataset 1D__

We can extract a `Dataset1D` object using any of the above `extract` objects.

This dataset contains the binned up 1D data, noise-map and pre-cti attributes of the 2D dataset (in this case,
an `ImagingCI` object used to create it. These are created using the `binned_array_1d_from` function.

We can therefore quickly convert a 2D CTI calibration dataset to 1D via stacking and binning, enabling us to fit a CTI
model in a signficant speed up. 

Below, we extract the parallel EPER's to create the 1D dataset. To ensure that the 1D pre-cti data contains flux
before the EPER's (e.g. the FPR) for accurate clocking with arctic we also include the 10 pixels in front of
the EPERs by specifying a negative `pixels` tuple.
"""
dataset = ac.ImagingCI.from_fits(
    data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
    noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
    pre_cti_data_path=path.join(dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"),
    layout=layout,
    pixel_scales=0.1,
)

dataset_1d = layout.extract.serial_overscan.dataset_1d_from(
    dataset_2d=dataset, settings=ac.SettingsExtract(pixels=(-10, 30))
)

dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
Finish.
"""
