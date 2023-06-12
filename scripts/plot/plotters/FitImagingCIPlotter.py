"""
Plots: FitImagingCIPlotter
==========================

This example illustrates how to plot a `FitImagingCI` dataset using an `FitImagingCIPlotter`.
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
The `Clocker` models the line read-out, including CTI. 
"""
clocker = ac.Clocker2D(
    parallel_express=5, parallel_roe=ac.ROEChargeInjection(), parallel_fast_mode=True
)

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

 - 2 `Trap` species in the parallel direction.
 - A simple CCDPhase volume filling parametrization.
 
This is the true CTI model used to simulate the dataset.
"""
parallel_trap_0 = ac.TrapInstantCapture(density=0.13, release_timescale=1.25)
parallel_trap_1 = ac.TrapInstantCapture(density=0.25, release_timescale=4.4)

parallel_trap_list = [parallel_trap_0, parallel_trap_1]

parallel_ccd = ac.CCDPhase(
    well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
)

cti = ac.CTI2D(
    parallel_trap_list=parallel_trap_list,
    parallel_ccd=parallel_ccd,
)

"""
Make a post-CTI image from the pre-CTI images in our `ImagingCI` dataset, using the `Clocker`.
"""
post_cti_data_list = [
    clocker.add_cti(data=dataset.pre_cti_data, cti=cti) for dataset in dataset_list
]

"""
We now perform the fit.
"""
fit_ci_list = [
    ac.FitImagingCI(dataset=dataset, post_cti_data=post_cti_data)
    for dataset, post_cti_data in zip(dataset_list, post_cti_data_list)
]

"""
We now pass the `FitImagingCI` and call various `figure_*` methods to plot different attributes.
"""
fit_plotter = aplt.FitImagingCIPlotter(fit=fit_ci_list[0])
fit_plotter.figures_2d(
    data=True,
    noise_map=True,
    pre_cti_data=True,
    residual_map=True,
    normalized_residual_map=True,
    chi_squared_map=True,
)

"""
The `FitImagingCIPlotter` may also plot a subplot of these attributes.
"""
fit_plotter.subplot_fit()

"""
__Regions__

We can also call `figures_1d_*` methods which create 1D plots of regions of the fit binned over the parallel or
serial direction.

The regions available are:

 `parallel_fpr`: The charge injection region binned up over all columns (e.g. across serial).
 `parallel_eper`: The parallel CTI trails behind the charge injection region binned up over all columns (e.g. 
  across serial).
 `serial_front_edge`: The charge injection region binned up over all rows (e.g. across parallel).
 `serial_trails`: The serial CTI trails behind the charge injection region binned up over all rows (e.g. across serial).
"""
fit_plotter.figures_1d(region="parallel_fpr", data=True, residual_map=True)
fit_plotter.figures_1d(region="parallel_eper", data=True, residual_map=True)

"""
Region plots also include the data with error bars showing the noise map.
"""
fit_plotter.figures_1d(region="parallel_fpr", data=True)
fit_plotter.figures_1d(region="parallel_eper", data=True)

"""
The above plots can also be created with a logarithmic y axis.
"""
fit_plotter.figures_1d(region="parallel_fpr", data_logy=True)
fit_plotter.figures_1d(region="parallel_eper", data_logy=True)

"""
There is also a subplot of these 1D plots.
"""
fit_plotter.subplot_1d(region="parallel_fpr")

"""
__Multiple Images on the Same Plot__

Our `FitImagingCI` is performed over multiple images taken at different charge injection levels. We may wish to plot
the results of the fit on each image on the same subplot, which can be performed using the 
method `subplot_of_figure`.
"""
fit_plotter_list = [aplt.FitImagingCIPlotter(fit=fit_ci) for fit_ci in fit_ci_list]
multi_plotter = aplt.MultiFigurePlotter(plotter_list=fit_plotter_list)

multi_plotter.subplot_of_figure(func_name="figures_2d", figure_name="data")
multi_plotter.subplot_of_figure(func_name="figures_2d", figure_name="residual_map")

"""
This method can also plot all of the 1D figures that we plotted above.
"""
fit_plotter_list = [aplt.FitImagingCIPlotter(fit=fit_ci) for fit_ci in fit_ci_list]

multi_plotter = aplt.MultiFigurePlotter(plotter_list=fit_plotter_list)

multi_plotter.subplot_of_figure(
    func_name="figures_1d", figure_name="residual_map", region="parallel_fpr"
)

"""
Finish.
"""
