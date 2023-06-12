"""
Plots: FitDataset1DPlotter
===================

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
The `Clocker` models the read-out of the data, including CTI. 
"""
clocker = ac.Clocker1D(express=5)

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the input line in the parallel direction, which contains: 

 - 2 `Trap` species in the parallel direction.
 - A simple CCDPhase volume filling parametrization.
 
This is the true CTI model used to simulate the dataset.
"""
trap_0 = ac.TrapInstantCapture(density=0.13, release_timescale=1.25)
trap_1 = ac.TrapInstantCapture(density=0.25, release_timescale=4.4)
trap_list = [trap_0, trap_1]

ccd = ac.CCDPhase(well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0)

cti = ac.CTI1D(
    trap_list=trap_list,
    ccd=ccd,
)

"""
Make a post-1D CTI Dataset from the pre-cti datas in our `Dataset1D` dataset, using the `Clocker`.
"""
post_cti_data_list = [
    clocker.add_cti(data=dataset.pre_cti_data, cti=cti)
    for dataset, norm in zip(dataset_list, norm_list)
]

"""
We now perform the fit.
"""
fit_1d_list = [
    ac.FitDataset1D(dataset=dataset, post_cti_data=post_cti_data)
    for dataset, post_cti_data in zip(dataset_list, post_cti_data_list)
]

"""
We now pass the `FitDataset1D` and call various `figure_*` methods to plot different attributes.
"""
fit_1d_plotter = aplt.FitDataset1DPlotter(fit=fit_1d_list[0])
fit_1d_plotter.figures_1d(
    data=True,
    noise_map=True,
    pre_cti_data=True,
    residual_map=True,
    normalized_residual_map=True,
    chi_squared_map=True,
)

"""
The `FitDataset1DPlotter` may also plot a subplot of these attributes.
"""
fit_1d_plotter.subplot_fit()

"""
Our `FitDataset1D` is performed over multiple lines taken at different charge injection levels. We may wish to plot
the results of the fit on each line on the same subplot, which can be performed using the 
method `subplot_of_figure`.
"""
fit_1d_plotter_list = [aplt.FitDataset1DPlotter(fit=fit_1d) for fit_1d in fit_1d_list]
multi_plotter = aplt.MultiFigurePlotter(plotter_list=fit_1d_plotter_list)

multi_plotter.subplot_of_figure(func_name="figures_1d", figure_name="data")
multi_plotter.subplot_of_figure(func_name="figures_1d", figure_name="residual_map")

"""
Finish.
"""
