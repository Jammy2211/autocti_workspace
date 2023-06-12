"""
Plots: MultiFigurePlotter
=========================

This example illustrates how to plot figures from different plotters on the same subplot, assuming that the same
type of `Plotter` and figure is being plotted.

An example of when to use this plotter would be when two different datasets (e.g. two charge injection datasets) are 
loaded and visualized, and the images of each dataset are plotted on the same subplot side-by-side. This is the example 
we will use in this example script.

This uses a `MultiFigurePlotter` object, which requires only a list of imaging datasets and `ImagingPlotter` objects
to be passed to it. The `MultiFigurePlotter` object then plots the same figure from each `ImagingPlotter` on the same
subplot.

The script `MultiSubplot.py` illustrates a similar example, but a more general use-case where different figures
from different plotters are plotted on the same subplot. This script offers a more concise way of plotting the same
figures on the same subplot, but is less general.
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
__Dataset__

Load the charge injection dataset 'imaging_ci/uniform/parallel_x2' from .fits files, which is the dataset we will
use to illustrate plotting charge injection data.
"""

shape_native = (2000, 100)

dataset_name = "simple"
dataset_path = path.join("dataset", "imaging_ci", dataset_name)

parallel_overscan = ac.Region2D((1980, 2000, 5, 95))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, 95, 100))

regions_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

norm_list = [100, 5000, 25000, 200000]

total_datasets = len(norm_list)

layout_list = [
    ac.Layout2DCI(
        shape_2d=shape_native,
        region_list=regions_list,
        parallel_overscan=parallel_overscan,
        serial_prescan=serial_prescan,
        serial_overscan=serial_overscan,
    )
    for i in range(total_datasets)
]

dataset_list = [
    ac.ImagingCI.from_fits(
        data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
        noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
        pre_cti_data_path=path.join(
            dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
        ),
        layout=layout,
        pixel_scales=0.1,
    )
    for layout, norm in zip(layout_list, norm_list)
]


"""
__Plot__

Plot the subhplot of each `ImagingCI` dataset individually using an `ImagingCIPlotter` object.
"""
for dataset in dataset_list:
    dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Multi Plot__

We now pass the list of `ImagingCIPlotter` objects to a `MultiFigurePlotter` object, which we use to plot the 
image of each dataset on the same subplot.

The `MultiFigurePlotter` object uses the `subplot_of_figure` method to plot the same figure from each `ImagingPlotter`,
with the inputs:

 - `func_name`: The name of the function used to plot the figure in the `ImagingPlotter` (e.g. `figures_2d`).
 - `figure_name`: The name of the figure plotted by the function (e.g. `image`).
"""
dataset_plotter_list = [
    aplt.ImagingCIPlotter(dataset=dataset) for dataset in dataset_list
]

multi_figure_plotter = aplt.MultiFigurePlotter(plotter_list=dataset_plotter_list)

multi_figure_plotter.subplot_of_figure(func_name="figures_2d", figure_name="data")

"""
__Wrap Up__

In the simple example above, we used a `MultiFigurePlotter` to plot the same figure from each `ImagingCIPlotter` on
the same `matplotlib` subplot. 

This can be used for any figure plotted by any `Plotter` object, as long as the figure is plotted using the same
function name and figure name.
"""
