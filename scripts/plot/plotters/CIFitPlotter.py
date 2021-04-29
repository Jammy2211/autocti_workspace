"""
Plots: FitImagingCIPlotter
===================

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
Load the charge injection dataset 'imaging_ci/uniform/parallel_x2' 'from .fits files, which is the dataset we will
use to illustrate plotting a fit to charge injection line data.
"""
dataset_name = "parallel_x2"
dataset_path = path.join("dataset", "imaging_ci", "uniform", dataset_name)

"""
The locations (using NumPy array indexes) of the parallel overscan, serial prescan and serial overscan on the image.
"""
parallel_overscan = ac.Region2D((1980, 2000, 5, 95))
serial_prescan = ac.Region2D((0, 2000, 0, 5))
serial_overscan = ac.Region2D((0, 1980, 95, 100))

"""
The normalization of every charge injection image, which names the files are with their charge injection normalization 
level.
"""
normalization_list = [100, 5000, 25000, 84700]

"""
Specify the charge injection regions on the CCD, which in this case is 5 equally spaced rectangular blocks.
"""
regions_list = [
    (0, 200, serial_prescan[3], serial_overscan[2]),
    (400, 600, serial_prescan[3], serial_overscan[2]),
    (800, 1000, serial_prescan[3], serial_overscan[2]),
    (1200, 1400, serial_prescan[3], serial_overscan[2]),
    (1600, 1800, serial_prescan[3], serial_overscan[2]),
]

"""
Create the layout of the charge injection pattern for every charge injection normalization.
"""
layout_list = [
    ac.ci.Layout2DCIUniform(
        shape_2d=(2000, 100),
        region_list=regions_list,
        normalization=normalization,
        parallel_overscan=parallel_overscan,
        serial_prescan=serial_prescan,
        serial_overscan=serial_overscan,
    )
    for normalization in normalization_list
]

"""
We can now load every image, noise-map and pre-CTI charge injection image as instances of the `ImagingCI` object.
"""
imaging_ci_list = [
    ac.ci.ImagingCI.from_fits(
        image_path=path.join(dataset_path, f"image_{layout.normalization}.fits"),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{layout.normalization}.fits"
        ),
        pre_cti_image_path=path.join(
            dataset_path, f"pre_cti_image_{layout.normalization}.fits"
        ),
        pixel_scales=0.1,
        layout=layout,
    )
    for layout in layout_list
]

"""
The `Clocker` models the `Line` read-out, including CTI. 
"""
clocker = ac.Clocker(parallel_express=2, parallel_charge_injection_mode=True)

"""
The CTI model used by arCTIc to add CTI to the input image in the parallel direction, which contains: 

 - 2 `Trap` species in the parallel direction.
 - A simple CCD volume beta parametrization.
 
This is the true CTI model used to simulate the dataset.
"""
parallel_trap_0 = ac.TrapInstantCapture(density=0.13, release_timescale=1.25)
parallel_trap_1 = ac.TrapInstantCapture(density=0.25, release_timescale=4.4)
parallel_ccd = ac.CCD(well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700)

"""
Make a post-CTI image from the pre-CTI images in our `ImagingCI` dataset, using the `Clocker`.
"""
post_cti_image_list = [
    clocker.add_cti(
        image=imaging_ci.pre_cti_image,
        parallel_traps=[parallel_trap_0, parallel_trap_1],
        parallel_ccd=parallel_ccd,
    )
    for imaging_ci in imaging_ci_list
]

"""
We now perform the fit.
"""
fit_ci_list = [
    ac.ci.FitImagingCI(imaging=imaging_ci, post_cti_image=post_cti_image)
    for imaging_ci, post_cti_image in zip(imaging_ci_list, post_cti_image_list)
]

"""
We now pass the `FitImagingCI` and call various `figure_*` methods to plot different attributes.
"""
fit_ci_plotter = aplt.FitImagingCIPlotter(fit=fit_ci_list[0])
fit_ci_plotter.figures_2d(
    image=True,
    noise_map=True,
    pre_cti_image=True,
    residual_map=True,
    normalized_residual_map=True,
    chi_squared_map=True,
)

"""
The `FitImagingCIPlotter` may also plot a subplot of these attributes.
"""
fit_ci_plotter.subplot_fit_ci()

"""
We can also call `figures_1d_*` methods which create 1D plots of regions of the fit binned over the parallel or
serial direction.

The regions available are:

 `parallel_front_edge`: The charge injection region binned up over all columns (e.g. across serial).
 `parallel_trails`: The parallel CTI trails behind the charge injection region binned up over all columns (e.g. 
  across serial).
 `serial_front_edge`: The charge injection region binned up over all rows (e.g. across parallel).
 `serial_trails`: The serial CTI trails behind the charge injection region binned up over all rows (e.g. across serial).
"""
fit_ci_plotter.figures_1d_ci_line_region(
    line_region="parallel_front_edge", image=True, residual_map=True
)
fit_ci_plotter.figures_1d_ci_line_region(
    line_region="parallel_trails", image=True, residual_map=True
)

"""
There is also a subplot of these 1D plots.
"""
fit_ci_plotter.subplot_1d_ci_line_region(line_region="parallel_front_edge")

"""
Our `FitImagingCI` is performed over multiple images taken at different charge injection levels. We may wish to plot
the results of the fit on each image on the same subplot, which can be performed using the 
method `subplot_of_figure`.
"""
fit_ci_plotter_list = [aplt.FitImagingCIPlotter(fit=fit_ci) for fit_ci in fit_ci_list]
multi_plotter = aplt.MultiFigurePlotter(plotter_list=fit_ci_plotter_list)

multi_plotter.subplot_of_figure(func_name="figures", figure_name="image")
multi_plotter.subplot_of_figure(func_name="figures", figure_name="residual_map")

"""
This method can also plot all of the 1D figures that we plotted above.
"""
multi_plotter.subplot_of_figure(
    func_name="figures_1d_ci_line_region",
    figure_name="residual_map",
    line_region="parallel_front_edge",
)

"""
Finish.
"""
