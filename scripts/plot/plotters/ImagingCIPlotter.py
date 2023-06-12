"""
Plots: ImagingCIPlotter
=======================

This example illustrates how to plot a `ImagingCI` dataset using an `ImagingCIPlotter`.
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
__Plotting__

We now pass the first dataset in the imaging to a `ImagingCIPlotter` and call various `figure_*` methods to plot 
different attributes.
"""
dataset_plotter = aplt.ImagingCIPlotter(
    dataset=dataset_list[0],
)
dataset_plotter.figures_2d(
    data=True,
    noise_map=True,
    pre_cti_data=True,
)

"""
The `ImagingCIPlotter` may also plot a subplot of all of these attributes.
"""
dataset_plotter.subplot_dataset()

"""
__Regions__

We can also call `figures_1d_*` methods which create 1D plots of regions of the image binned over the parallel or
serial direction.

The regions available are:

 `parallel_fpr`: The charge injection region binned up over all columns (e.g. across serial).
 `parallel_eper`: The parallel CTI trails behind the charge injection region binned up over all columns (e.g. 
  across serial).
 `serial_front_edge`: The charge injection region binned up over all rows (e.g. across parallel).
 `serial_trails`: The serial CTI trails behind the charge injection region binned up over all rows (e.g. across serial).
"""
dataset_plotter.figures_1d(region="parallel_fpr", data=True, pre_cti_data=True)
dataset_plotter.figures_1d(region="parallel_eper", data=True, pre_cti_data=True)

"""
Region plots also include the data with error bars showing the noise map.
"""
dataset_plotter.figures_1d(region="parallel_fpr", data=True)
dataset_plotter.figures_1d(region="parallel_eper", data=True)

"""
The above plots can also be created with a logarithmic y axis.
"""
dataset_plotter.figures_1d(region="parallel_fpr", data_logy=True)
dataset_plotter.figures_1d(region="parallel_eper", data_logy=True)

"""
There is also a subplot of these 1D plots.
"""
dataset_plotter.subplot_1d(region="parallel_fpr")

"""`
Imaging` contains the following attributes which can be plotted automatically via the `Include2D` object.

(By default, an `Array2D` does not contain a `Mask2D`, we therefore manually created an `Array2D` with a mask to 
illustrate the plotted of a mask and its border below).
"""
include = aplt.Include2D(
    parallel_overscan=True, serial_prescan=True, serial_overscan=True
)
dataset_plotter = aplt.ImagingCIPlotter(dataset=dataset_list[0], include_2d=include)
dataset_plotter.figures_2d(data=True)

"""
__Multiple Images__

Our `ImagingCI` dataset consists of many images taken at different charge injection levels. We may wish to plot
all images on the same subplot, which can be performed using the method `subplot_of_figure`.
"""
dataset_plotter_list = [
    aplt.ImagingCIPlotter(dataset=dataset) for dataset in dataset_list
]
multi_plotter = aplt.MultiFigurePlotter(plotter_list=dataset_plotter_list)
multi_plotter.subplot_of_figure(func_name="figures_2d", figure_name="data")
multi_plotter.subplot_of_figure(func_name="figures_2d", figure_name="pre_cti_data")

"""
This method can also plot all of the 1D figures that we plotted above.
"""
multi_plotter.subplot_of_figure(
    func_name="figures_1d", figure_name="data", region="parallel_fpr"
)

"""
__Settings Dictionary__

The `settings_dict` of each dataset has entries corresponding to the settings used to create the data. 

For example, this might be the voltages of the charge injections.

This will display on 1D figures when they are plotted, so that when we create a subplot of many datasets we can
see the settings of each dataset.
"""
dataset_list = [
    ac.ImagingCI.from_fits(
        data_path=path.join(dataset_path, f"norm_{int(norm)}", "data.fits"),
        noise_map_path=path.join(dataset_path, f"norm_{int(norm)}", "noise_map.fits"),
        pre_cti_data_path=path.join(
            dataset_path, f"norm_{int(norm)}", "pre_cti_data.fits"
        ),
        layout=layout,
        pixel_scales=0.1,
        settings_dict={"voltage_0": "3V", "voltage_2": "6V"},
    )
    for layout, norm in zip(layout_list, norm_list)
]

dataset_plotter_list = [
    aplt.ImagingCIPlotter(dataset=dataset) for dataset in dataset_list
]
multi_plotter = aplt.MultiFigurePlotter(plotter_list=dataset_plotter_list)

multi_plotter.subplot_of_figure(
    func_name="figures_1d", figure_name="data", region="parallel_fpr"
)


"""
Finish.
"""
