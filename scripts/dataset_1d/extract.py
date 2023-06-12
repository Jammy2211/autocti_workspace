"""
Extract
-------

**PyAutoCTI** has numerous methods for extracting subsets of data from a 1D CTI dataset.

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

We begin by loading 1D CTI data which has CTI added, which we will use to illustrate the extraction methods.

We also set up its corresponding `Layout1DCI` object, which is used to perform extractions.

You should be familiar with the **PyAutoCTI** API below, if not check out other scripts throughout the workspace.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "dataset_1d", dataset_name)

shape_native = (200,)

"""
The overscans input here are used to perform extractions.
"""
prescan = ac.Region1D((0, 10))
overscan = ac.Region1D((190, 200))

"""
These charge regions are also used to perform extractions.
"""
region_list = [(10, 20)]

norm = 100

"""
The layout object contains all the extraction functionality used in this example.
"""
layout_1d = ac.Layout1D(
    shape_1d=shape_native,
    region_list=region_list,
    prescan=prescan,
    overscan=overscan,
)

"""
We will demonstrate extraction on a 1D CTI image, but the functionality demonstrated here can be used
on any 1D array (e.g. the noise-map, a CTI corrected data, a residual-map of a fit).
"""
data_1d = ac.Array1D.from_fits(
    file_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
    pixel_scales=0.1,
)

"""
__EPERs__

The 1D region of every set of EPERs on the 1D CTI data is contained in the layout's extract object.

To compute the EPER regions, we specify the number of rows we want each EPERs to span, using the `pixels` input. 

For example, by inputting `pixels=(0, 10)` each EPER region contains 30 pixels.
"""
region_1d_eper_list = layout_1d.extract.eper.region_list_from(
    settings=ac.SettingsExtract(pixels=(0, 10))
)
print(region_1d_eper_list)

"""
The EPER regions above are used to extract the EPERs from the 1D CTI data. 

The function below returns a list of 1D arrays containing each EPER.

We again specify the number of pixel each 1D EPER that is extracted spans. 
"""
data_1d_eper_list = layout_1d.extract.eper.array_1d_list_from(
    array=data_1d, settings=ac.SettingsExtract(pixels=(0, 10))
)

array_1d_plotter = aplt.Array1DPlotter(y=data_1d_eper_list[0])
array_1d_plotter.figure_1d()

"""
A stacked 2D array of the list of 2D arrays returned above can be extracted.

The stacking process reduces noise in the data, making the EPERs higher signal to noise.
"""
data_1d_eper = layout_1d.extract.eper.stacked_array_1d_from(
    array=data_1d, settings=ac.SettingsExtract(pixels=(0, 10))
)

array_1d_plotter = aplt.Array1DPlotter(y=data_1d_eper)
array_1d_plotter.figure_1d()

"""
Negative inputs to the `pixels` tuple are supported in all of the above methods, which extract additional pixels in 
front of the EPERs.
"""
data_1d_eper = layout_1d.extract.eper.stacked_array_1d_from(
    array=data_1d, settings=ac.SettingsExtract(pixels=(-10, 10))
)

array_1d_plotter = aplt.Array1DPlotter(y=data_1d_eper)
array_1d_plotter.figure_1d()

"""
__FPRs__

All of the above methods can be used extract and stack the FPR in an analogous fashion.
"""
region_1d_fpr_list = layout_1d.extract.fpr.region_list_from(
    settings=ac.SettingsExtract(pixels=(0, 10))
)
print(region_1d_fpr_list)

data_1d_fpr_list = layout_1d.extract.fpr.array_1d_list_from(
    array=data_1d, settings=ac.SettingsExtract(pixels=(0, 10))
)

array_1d_plotter = aplt.Array1DPlotter(y=data_1d_fpr_list[0])
array_1d_plotter.figure_1d()

data_1d_fpr = layout_1d.extract.fpr.stacked_array_1d_from(
    array=data_1d, settings=ac.SettingsExtract(pixels=(0, 10))
)

array_1d_plotter = aplt.Array1DPlotter(y=data_1d_fpr)
array_1d_plotter.figure_1d()

data_1d_fpr = layout_1d.extract.fpr.stacked_array_1d_from(
    array=data_1d, settings=ac.SettingsExtract(pixels=(0, 10))
)

array_1d_plotter = aplt.Array1DPlotter(y=data_1d_fpr)
array_1d_plotter.figure_1d()

"""
__Overscans__

There are analogous extract methods for the overscan.

In 1D, there is not an obvious reason for extracting the overscan. However, in 2D there are many (see the `extract.py`
script in the `imaging_ci` package of the workspace). Nevertheless, the API for doing this is included for completeness.
"""
region_1d_overscan_list = layout_1d.extract.overscan.region_list_from(
    settings=ac.SettingsExtract(pixels=(0, 10))
)
print(region_1d_overscan_list)

data_1d_overscan_list = layout_1d.extract.overscan.array_1d_list_from(
    array=data_1d, settings=ac.SettingsExtract(pixels=(0, 10))
)

array_1d_plotter = aplt.Array1DPlotter(y=data_1d_overscan_list[0])
array_1d_plotter.figure_1d()

data_1d_overscan = layout_1d.extract.overscan.stacked_array_1d_from(
    array=data_1d, settings=ac.SettingsExtract(pixels=(0, 10))
)

array_1d_plotter = aplt.Array1DPlotter(y=data_1d_overscan)
array_1d_plotter.figure_1d()

data_1d_overscan = layout_1d.extract.overscan.stacked_array_1d_from(
    array=data_1d, settings=ac.SettingsExtract(pixels=(0, 10))
)

array_1d_plotter = aplt.Array1DPlotter(y=data_1d_overscan)
array_1d_plotter.figure_1d()

"""
Finish.
"""
