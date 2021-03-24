"""
Plots: CIFitPlotter
===================

This example illustrates how to plot a `CIImaging` dataset using an `CIImagingPlotter`.
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
Load the charge injection dataset 'ci_imaging/uniform/parallel_x2' 'from .fits files, which is the dataset we will
use to illustrate plotting a fit to charge injection line data.
"""
dataset_name = "parallel_x2"
dataset_path = path.join("dataset", "ci_imaging", "uniform", dataset_name)

"""
The locations of the prescans and overscans on the image, which is used to visualize the `CIImaging` during the 
model-fit.
"""
scans = ac.Scans(
    parallel_overscan=ac.Region2D((1980, 2000, 5, 95)),
    serial_prescan=ac.Region2D((0, 2000, 0, 5)),
    serial_overscan=ac.Region2D((0, 1980, 95, 100)),
)

"""
Specify the charge injection regions on the CCD, which in this case is 3 equally spaced rectangular blocks.
"""
ci_regions = [
    (0, 200, scans.serial_prescan[3], scans.serial_overscan[2]),
    (400, 600, scans.serial_prescan[3], scans.serial_overscan[2]),
    (800, 1000, scans.serial_prescan[3], scans.serial_overscan[2]),
    (1200, 1400, scans.serial_prescan[3], scans.serial_overscan[2]),
    (1600, 1800, scans.serial_prescan[3], scans.serial_overscan[2]),
]


"""
We require the normalization of every charge injection image, as the names of the files are tagged with the charge
injection normalization level.
"""
normalizations = [100, 5000, 25000, 84700]

"""
Use the charge injection normalizations and regions to create `CIPatternUniform` of every image we'll fit. The
`CIPattern` is used for visualizing the model-fit.
"""
ci_patterns = [
    ac.ci.CIPatternUniform(normalization=normalization, regions=ci_regions)
    for normalization in normalizations
]

"""
We can now load every image, noise-map and pre-CTI charge injection image as instances of the `CIImaging` object.
"""
ci_imaging_list = [
    ac.ci.CIImaging.from_fits(
        image_path=path.join(dataset_path, f"image_{pattern.normalization}.fits"),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{pattern.normalization}.fits"
        ),
        ci_pre_cti_path=path.join(
            dataset_path, f"ci_pre_cti_{pattern.normalization}.fits"
        ),
        pixel_scales=0.1,
        ci_pattern=pattern,
        roe_corner=(1, 0),
    )
    for pattern in ci_patterns
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
Make a post-CTI image from the pre-CTI images in our `CIImaging` dataset, using the `Clocker`.
"""
ci_post_cti_list = [
    clocker.add_cti(
        image=ci_imaging.ci_pre_cti,
        parallel_traps=[parallel_trap_0, parallel_trap_1],
        parallel_ccd=parallel_ccd,
    )
    for ci_imaging in ci_imaging_list
]

"""
We now perform the fit.
"""
ci_fit_list = [
    ac.ci.CIFitImaging(masked_ci_imaging=ci_imaging, ci_post_cti=ci_post_cti)
    for ci_imaging, ci_post_cti in zip(ci_imaging_list, ci_post_cti_list)
]

"""
We now pass the `CIFitImaging` and call various `figure_*` methods to plot different attributes.
"""
ci_fit_plotter = aplt.CIFitPlotter(fit=ci_fit_list[0])
ci_fit_plotter.figures(
    image=True,
    noise_map=True,
    ci_pre_cti=True,
    residual_map=True,
    normalized_residual_map=True,
    chi_squared_map=True,
)

"""
The `CIFitPlotter` may also plot a subplot of these attributes.
"""
ci_fit_plotter.subplot_ci_fit()

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
ci_fit_plotter.figures_1d_ci_line_region(
    line_region="parallel_front_edge", image=True, residual_map=True
)
ci_fit_plotter.figures_1d_ci_line_region(
    line_region="parallel_trails", image=True, residual_map=True
)

"""
There is also a subplot of these 1D plots.
"""
ci_fit_plotter.subplot_1d_ci_line_region(line_region="parallel_front_edge")

"""
Our `CIFitImaging` is performed over multiple images taken at different charge injection levels. We may wish to plot
the results of the fit on each image on the same subplot, which can be performed using the 
method `subplot_of_figure`.
"""
ci_fit_plotter_list = [aplt.CIFitPlotter(fit=ci_fit) for ci_fit in ci_fit_list]
multi_plotter = aplt.MultiPlotter(plotter_list=ci_fit_plotter_list)

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
