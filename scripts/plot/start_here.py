"""
Plots: Start Here
=================

This example illustrates the basic API for plotting.
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

First, lets load an example image of a CTI `Dataset1D`.

You should be familiar with how we load datasets in this way, if not checkout the `overview` 
and `modeling/start_here.py` examples.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "dataset_1d", dataset_name)
shape_native = (200,)

prescan = ac.Region1D(region=(0, 10))
overscan = ac.Region1D(region=(190, 200))

region_list = [(10, 20)]

norm_list = [100]

total_datasets = len(norm_list)

layout_list = [
    ac.Layout1D(
        shape_1d=shape_native,
        region_list=region_list,
        prescan=prescan,
        overscan=overscan,
    )
    for i in range(total_datasets)
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
__Plot Customization via MatPlot__

You can customize a number of matplotlib setup options using a `MatPlot` object, which 
wraps the `matplotlib` methods used to display the image.

(For example, the `Figure` class wraps the `matplotlib` method `plt.figure()`, whereas the `YTicks` class wraps
`plt.yticks`).

The `autocti.workspace.*.plot.mat_wrap` illustrates every `MatPlot` object, for 
example `Figure`, `YTicks`, etc.
"""
mat_plot = aplt.MatPlot1D(
    figure=aplt.Figure(figsize=(7, 7)),
    yticks=aplt.YTicks(fontsize=8),
    xticks=aplt.XTicks(fontsize=8),
    title=aplt.Title(fontsize=12),
    ylabel=aplt.YLabel(fontsize=6),
    xlabel=aplt.XLabel(fontsize=6),
)

dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0], mat_plot_1d=mat_plot)
dataset_plotter.figures_1d(data=True)

"""
__Configs__

All matplotlib options can be customized via the config files, such that those values are used every time.

Checkout the `mat_wrap.yaml`, `mat_wrap_1d.yaml` and `mat_wrap_2d.yaml` files 
in `autocti_workspace/config/visualize/mat_wrap`.

All default matplotlib values are here. There are a lot of entries, so lets focus on whats important for displaying 
figures:

 - mat_wrap.yaml -> Figure -> figure: -> figsize
 - mat_wrap.yaml -> YLabel -> figure: -> fontsize
 - mat_wrap.yaml -> XLabel -> figure: -> fontsize
 - mat_wrap.yaml -> TickParams -> figure: -> labelsize
 - mat_wrap.yaml -> YTicks -> figure: -> labelsize
 - mat_wrap.yaml -> XTicks -> figure: -> labelsize

__Subplots__

In addition to plotting individual `figures`, we also plot `subplots` which are again customized via
the `mat_plot` objects.

__Visuals__

Visuals can be added to any figure, using standard quantities.

For example, we can plot a mask on the image above using a `Visuals2D` object.

The `autocti.workspace.*.plot.visuals_2d` illustrates every `Visuals` object, for 
example `MaskScatter`, `LightProfileCentreScatter`, etc.
"""
visuals = aplt.Visuals1D(vertical_line=5.0)

dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0], visuals_1d=visuals)
dataset_plotter.figures_1d(data=True)

"""
__Searches__

Model-fits using a non-linear search (e.g. Nautilus, Emcee) produce search-specific visualization.

The `autocti.workspace.*.plot.search` illustrates how to perform this visualization for every search (e.g.
`NautilusPlotter`, `EmceePlotter`.

__Adding Plotter Objects Together__

The `MatPlot` objects can be added together. 

This is useful when we want to perform multiple visualizations which share the same base settings, but have
individually tailored settings:
"""
mat_plot_base = aplt.MatPlot1D(
    yticks=aplt.YTicks(fontsize=18),
    xticks=aplt.XTicks(fontsize=18),
    ylabel=aplt.YLabel(ylabel=""),
    xlabel=aplt.XLabel(xlabel=""),
)

mat_plot = aplt.MatPlot1D(
    title=aplt.Title(label="Example Figure 1"),
)

mat_plot = mat_plot + mat_plot_base

dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0], mat_plot_1d=mat_plot)
dataset_plotter.figures_1d(data=True)

mat_plot = aplt.MatPlot1D(
    title=aplt.Title(label="Example Figure 2"),
)

mat_plot = mat_plot + mat_plot_base

dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0], mat_plot_1d=mat_plot)
dataset_plotter.figures_1d(data=True)

mat_plot = mat_plot + mat_plot_base


"""
The `Visuals` objects can also be added together.
"""
visuals_1d_0 = aplt.Visuals1D(vertical_line=5.0)
visuals_1d_1 = aplt.Visuals1D(shaded_region=[6.0, 7.0])

visuals = visuals_1d_0 + visuals_1d_1

dataset_plotter = aplt.Dataset1DPlotter(
    dataset=dataset_list[0], visuals_1d=visuals, mat_plot_1d=aplt.MatPlot1D()
)
dataset_plotter.figures_1d(data=True)

"""
Finish.
"""
