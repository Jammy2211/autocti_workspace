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
# dataset_name = "species_x1_continuum_0"
# trap_0 = ac.TrapLogNormalLifetimeContinuum(
#     density=1.0, release_timescale_mu=1.0, release_timescale_sigma=0.1
# )

# dataset_name = "species_x1_continuum_1"
# trap_0 = ac.TrapLogNormalLifetimeContinuum(
#     density=1.0, release_timescale_mu=5.0, release_timescale_sigma=0.1
# )

# dataset_name = "species_x1_continuum_2"
# trap_0 = ac.TrapLogNormalLifetimeContinuum(
#     density=1.0, release_timescale_mu=20.0, release_timescale_sigma=0.1
# )

# dataset_name = "species_x2_continuum_0"
# trap_0 = ac.TrapLogNormalLifetimeContinuum(
#     density=1.0, release_timescale_mu=1.0, release_timescale_sigma=0.1
# )
# trap_1 = ac.TrapLogNormalLifetimeContinuum(
#     density=1.0, release_timescale_mu=5.0, release_timescale_sigma=0.1
# )

dataset_name = "species_x2_continuum_1"
trap_0 = ac.TrapLogNormalLifetimeContinuum(
    density=1.0, release_timescale_mu=5.0, release_timescale_sigma=0.1
)
trap_1 = ac.TrapLogNormalLifetimeContinuum(
    density=1.0, release_timescale_mu=20.0, release_timescale_sigma=0.1
)

# dataset_name = "species_x2_continuum_2"
# trap_0 = ac.TrapLogNormalLifetimeContinuum(
#     density=1.0, release_timescale_mu=1.0, release_timescale_sigma=0.1
# )
# trap_1 = ac.TrapLogNormalLifetimeContinuum(
#     density=1.0, release_timescale_mu=20.0, release_timescale_sigma=0.1
# )

dataset_path = path.join("dataset", "line", dataset_name)

"""
The CTI model used by arCTIc to add CTI to the input `Line`, which contains: 

 - 1 `Trap` species.
 - A simple CCD volume beta parametrization.
 
This is the true model and thus will give us the log likelihood the best-fit model gives.
"""
traps = [trap_0, trap_1]
ccd = ac.CCD(well_fill_power=0.8, well_notch_depth=0.0, full_well_depth=84700)

"""
Specify the regions on the `Line` where charge was present before CTI, 10 pixels after the serial prescan.
"""
regions = [(10, 20, 0, 1)]

"""
The normalization of every `Line`, this list determines how many `Line`'s are simulated.
"""
normalizations = [100, 500, 1000, 5000, 10000, 25000, 50000, 84700]

"""
Use the `Line` normalizations and regions to create `LinePattern` of every `Line` we fit.
"""
line_pattern_list = [
    ac.ci.CIPatternUniform(normalization=normalization, regions=regions)
    for normalization in normalizations
]

"""
We can now load every `Line`, noise-map and pre-CTI line as instances of the `LineDataset` object.
"""
line_list = [
    ac.ci.CIImaging.from_fits(
        image_path=path.join(dataset_path, f"image_{pattern.normalization}.fits"),
        noise_map_path=path.join(
            dataset_path, f"noise_map_{pattern.normalization}.fits"
        ),
        ci_pre_cti_path=path.join(
            dataset_path, f"line_pre_cti_{pattern.normalization}.fits"
        ),
        pixel_scales=0.1,
        ci_pattern=pattern,
        roe_corner=(1, 0),
    )
    for pattern in line_pattern_list
]

"""
Lets plot the first `LineDataset` with its `Mask1D`
"""
ci_imaging_plotter = aplt.CIImagingPlotter(imaging=line_list[0])
ci_imaging_plotter.subplot_ci_imaging()

"""
We now need to mask the data, so that regions where there is no signal (e.g. the edges) are omitted from the fit.
"""
mask_list = [
    ac.ci.CIMask2D.unmasked(
        shape_native=line.shape_native, pixel_scales=line.pixel_scales
    )
    for line in line_list
]

mask_list = [
    ac.ci.CIMask2D.masked_front_edges_and_trails_from_ci_frame(
        mask=mask, ci_frame=line_list[0].image, settings=ac.ci.SettingsCIMask()
    )
    for mask in mask_list
]

"""
The MaskedImaging object combines the dataset with the mask.
"""
masked_line_list = [
    ac.ci.MaskedCIImaging(ci_imaging=line, mask=mask)
    for line, mask in zip(line_list, mask_list)
]

"""
Here is what our image looks like with the mask applied.
"""
ci_imaging_plotter = aplt.CIImagingPlotter(imaging=line_list[0])
ci_imaging_plotter.figures(image=True)

"""
The `Clocker` models the `Line` read-out, including CTI. 
"""
clocker = ac.Clocker(parallel_express=2)

"""
Following the clocking.py example, we can make a post-CTI image from the pre-CTI image in our calibration dataset,
using the `Clocker`.
"""
line_post_cti_list = [
    clocker.add_cti(
        image=masked_line.ci_pre_cti, parallel_traps=traps, parallel_ccd=ccd
    )
    for masked_line in masked_line_list
]

"""
We can now quantify how well this CTI model fits the CTI `Line` in our dataset, by performing a `FitLine`.
"""
fit_list = [
    ac.ci.CIFitImaging(masked_ci_imaging=masked_line, ci_post_cti=line_post_cti)
    for masked_line, line_post_cti in zip(masked_line_list, line_post_cti_list)
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
ci_fit_plotter = aplt.CIFitPlotter(fit=fit_list[0])
ci_fit_plotter.figures(
    residual_map=True, normalized_residual_map=True, chi_squared_map=True
)

print(sum([fit.log_likelihood for fit in fit_list]))
