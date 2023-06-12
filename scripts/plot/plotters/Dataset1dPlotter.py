"""
Plots: Dataset1DPlotter
=======================

This example illustrates how to plot a `Dataset1D` dataset using an `Dataset1DPlotter`.
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
__dataset__

Load the dataset 'dataset_1d/species_x1' from .fits files, which is the dataset we will use to illustrate plotting 
the dataset.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "dataset_1d", dataset_name)

shape_native = (200, 1)

prescan = ac.Region1D((0, 10))
overscan = ac.Region1D((190, 200))

region_list = [(10, 20)]

norm_list = [100, 5000, 25000, 200000]

layout_list = [
    ac.Layout1D(
        shape_1d=shape_native,
        region_list=region_list,
        prescan=prescan,
        overscan=overscan,
    )
    for norm in norm_list
]

dataset_list = [
    ac.Dataset1D.from_fits(
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

We now pass the first dataset in the imaging to a `Dataset1DPlotter` and call various `figure_*` methods to plot 
different attributes.
"""

output = aplt.Output(path=path.join("."), format="png")

dataset_plotter = aplt.Dataset1DPlotter(
    dataset=dataset_list[0], mat_plot_1d=aplt.MatPlot1D(output=output)
)
dataset_plotter.figures_1d(
    data_logy=True,
    noise_map=True,
    signal_to_noise_map=True,
    pre_cti_data=True,
)

"""
The `Dataset1DPlotter` may also plot a subplot of all of these attributes.
"""
dataset_plotter.subplot_dataset()

"""`
Imaging` contains the following attributes which can be plotted automatically via the `Include2D` object.

(By default, an `Array2D` does not contain a `Mask2D`, we therefore manually created an `Array2D` with a mask to 
illustrate the plotted of a mask and its border below).
"""
include = aplt.Include1D()
dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0], include_1d=include)
dataset_plotter.figures_1d(data=True)

"""
__Regions__

Specific regions of the data can be extracted and plotted, for example the EPER or FPR.
"""
dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0])
dataset_plotter.figures_1d(region="fpr", data=True)
dataset_plotter.figures_1d(region="eper", data=True)

"""
Region plots also include the data with error bars showing the noise map.
"""
dataset_plotter.figures_1d(region="fpr", data=True)
dataset_plotter.figures_1d(region="eper", data=True)

"""
The above plots can also be created with a logarithmic y axis.
"""
dataset_plotter.figures_1d(region="fpr", data_logy=True)
dataset_plotter.figures_1d(region="eper", data_logy=True)

"""
__Multiple Images__

Our `Dataset1D` dataset consists of many images taken at different charge injection levels. We may wish to plot
all images on the same subplot, which can be performed using the method `subplot_of_figure`.
"""
dataset_plotter_list = [
    aplt.Dataset1DPlotter(dataset=dataset)
    for dataset, norm in zip(dataset_list, norm_list)
]
multi_plotter = aplt.MultiFigurePlotter(plotter_list=dataset_plotter_list)
multi_plotter.subplot_of_figure(func_name="figures_1d", figure_name="data")
multi_plotter.subplot_of_figure(func_name="figures_1d", figure_name="pre_cti_data")

"""
__Settings Dictionary__

The `settings_dict` of each dataset has entries corresponding to the settings used to create the data. 

For example, this might be the voltages of the charge injections.

This will display on 1D figures when they are plotted, so that when we create a subplot of many datasets we can
see the settings of each dataset.
"""
dataset_list = [
    ac.Dataset1D.from_fits(
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
    aplt.Dataset1DPlotter(dataset=dataset)
    for dataset, norm in zip(dataset_list, norm_list)
]
multi_plotter = aplt.MultiFigurePlotter(plotter_list=dataset_plotter_list)
multi_plotter.subplot_of_figure(func_name="figures_1d", figure_name="data")

"""
Finish.
"""
