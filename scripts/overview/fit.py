"""
__Example: Fitting__

**PyAutoCTI** uses a `Clocker` object to clock an image without CTI through a simulation of the CCD readout process.
Now, we`re going use these objects to fit CTI cabliration `Imaging` data to infer a CTI model

The autocti_workspace comes distributed with simulated `Line` datasets of CTI trails of strong lenses (an example of
how a simulation is made can be found in the `simulate.py` example, with all simulator scripts located in
`autocti_workspac/simulators`.
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
Load the CTI `Line` dataset `species_x3` `from .fits files, which is the dataset we will fit with a CTI model.
"""
dataset_name = "species_x3"
dataset_path = path.join("dataset", "line", dataset_name)

"""
Specify the regions on the `Line` where charge was present before CTI, 10 pixels after the serial prescan.
"""
regions = [(10, 20, 0, 1)]

"""
The normalization of every `Line`, this list determines how many `Line`'s are simulated.
"""
normalization_list = [100, 5000, 25000, 84700]

"""
Use the `Line` normalization_list and regions to create `LinePattern` of every `Line` we fit.
"""
line_pattern_list = [
    ac.ci.PatternCIUniform(normalization=normalization, regions=regions)
    for normalization in normalization_list
]

"""
We can now load every `Line`, noise-map and pre-CTI line as instances of the `LineDataset` object.
"""
line_list = [
    ac.ci.ImagingCI.from_fits(
        image_path=path.join(dataset_path, f"image_{pattern.normalization}.fits"),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{pattern.normalization}.fits"
        ),
        pre_cti_image_path=path.join(
            dataset_path, f"pre_cti_line_{pattern.normalization}.fits"
        ),
        pixel_scales=0.1,
        pattern_ci=pattern,
        roe_corner=(1, 0),
    )
    for pattern in line_pattern_list
]

"""
Lets plot the first `LineDataset` with its `Mask1D`
"""
imaging_ci_plotter = aplt.ImagingCIPlotter(imaging=line_list[0])
imaging_ci_plotter.subplot_imaging_ci()

"""
We now need to mask the data, so that regions where there is no signal (e.g. the edges) are omitted from the fit.
"""
mask_list = [
    ac.Mask2D.unmasked(shape_native=line.shape_native, pixel_scales=line.pixel_scales)
    for line in line_list
]

"""
The MaskedImaging object combines the dataset with the mask.
"""
masked_line_list = [
    ac.ci.ImagingCI(imaging_ci=line, mask=mask)
    for line, mask in zip(line_list, mask_list)
]

"""
Here is what our image looks like with the mask applied.
"""
imaging_ci_plotter = aplt.ImagingCIPlotter(imaging=line_list[0])
imaging_ci_plotter.figures_2d(image=True)

"""
The `Clocker` models the `Line` read-out, including CTI. 
"""
clocker = ac.Clocker(parallel_express=2)

"""
__CTI Model__

The CTI model used by arCTIc to add CTI to the input `Line`, which contains: 

 - 2 `Trap` species.
 - A simple CCD volume beta parametrization.
"""
trap_0 = ac.TrapInstantCapture(density=0.0442, release_timescale=0.8)
trap_1 = ac.TrapInstantCapture(density=0.1326, release_timescale=4.0)
trap_2 = ac.TrapInstantCapture(density=3.9782, release_timescale=20.0)
traps = [trap_0, trap_1, trap_2]
ccd = ac.CCDPhase(well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700.0)

"""
Following the clocking.py example, we can make a post-CTI image from the pre-CTI image in our calibration dataset,
using the `Clocker`.
"""
post_cti_line_list = [
    clocker.add_cti(
        image_pre_cti=masked_line.pre_cti_image, parallel_traps=traps, parallel_ccd=ccd
    )
    for masked_line in masked_line_list
]

"""
We can now quantify how well this CTI model fits the CTI `Line` in our dataset, by performing a `FitLine`.
"""
fit_list = [
    ac.ci.FitImagingCI(imaging=masked_line, post_cti_image=post_cti_line)
    for masked_line, post_cti_line in zip(masked_line_list, post_cti_line_list)
]

"""
The fit creates the following:

 - The residual-map: The model-image subtracted from the observed dataset`s image.
 - The normalized residual-map: The residual-map divided by the noise-map.
 - The chi-squared-map: The normalized residual-map squared.

we'll plot all 3 of these, alongside a subplot containing them all.

For a good lens model where the model image and tracer are representative of the strong lens system the
residuals, normalized residuals and chi-squareds are minimized:
"""
fit_ci_plotter = aplt.FitImagingCIPlotter(fit=fit_list[0])
fit_ci_plotter.figures_2d(
    residual_map=True, normalized_residual_map=True, chi_squared_map=True
)

print(sum([fit.log_likelihood for fit in fit_list]))

"""
In contrast, a bad CTI model will show features in the residual-map and chi-squareds.

We can produce such an image by creating a CTI model with different `Traps` and / or `CCD` volumen filling.. In the 
example below, we divide the density of each trap of by 2, which leads to residuals appearing in the fit.
"""
trap_0 = ac.TrapInstantCapture(density=0.0442 / 2.0, release_timescale=0.8)
trap_1 = ac.TrapInstantCapture(density=0.1326 / 2.0, release_timescale=4.0)
trap_2 = ac.TrapInstantCapture(density=3.9782 / 2.0, release_timescale=20.0)
traps = [trap_0, trap_1, trap_2]

post_cti_line_list = [
    clocker.add_cti(
        image_pre_cti=masked_line.pre_cti_image, parallel_traps=traps, parallel_ccd=ccd
    )
    for masked_line in masked_line_list
]

"""
Lets create a new fit using this tracer and replot its residuals, normalized residuals and chi-squareds.
"""
fit_list = [
    ac.ci.FitImagingCI(imaging=masked_line, post_cti_image=post_cti_line)
    for masked_line, post_cti_line in zip(masked_line_list, post_cti_line_list)
]

fit_ci_plotter = aplt.FitImagingCIPlotter(fit=fit_list[0])
fit_ci_plotter.figures_2d(
    residual_map=True, normalized_residual_map=True, chi_squared_map=True
)

print(sum([fit.log_likelihood for fit in fit_list]))
