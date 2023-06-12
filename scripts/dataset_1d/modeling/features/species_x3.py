"""
Modeling: Dataset 1D x3 Species
===============================

__Model__

In this script, we will fit a 1D CTI Dataset to calibrate a CTI model, where:

 - The CTI model consists of multiple parallel `TrapInstantCapture` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.

__Start Here Notebook__

If any code in this script is unclear, refer to the modeling `start_here.ipynb` notebook for more detailed comments.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autocti as ac
import autocti.plot as aplt

"""
__Dataset__

Load the CTI dataset 'dataset_1d/species_x3' 'from .fits files, which is the dataset we will use to perform CTI 
modeling.
"""
dataset_name = "species_x3"
dataset_path = path.join("dataset", "dataset_1d", dataset_name)

"""
__Shape__

The 1D shape of each 1D dataset.
"""
shape_native = (2000,)

"""
__Regions__

The locations of the prescan and overscan on the 1D data, which is used to visualize the 1D CTI dataset during the 
model-fit and customize aspects of the model-fit.
"""
prescan = ac.Region1D((0, 10))
overscan = ac.Region1D((1990, 2000))

"""
__FPR / EPER__

Specify the charge regions on the 1D CTI Dataset, corresponding to where a signal is contained that has its electrons 
captured and trailed by CTI.
"""
region_list = [
    (10, 40),
    (310, 340),
    (610, 640),
    (910, 940),
    (1210, 1240),
    (1510, 1540),
]

"""
__Normalizations__

We require the normalization of the charge in every CTI dataset, as the names of the files are tagged with this.
"""
norm_list = [100, 5000, 25000, 200000]

"""
__Layout__

We use the regions and norm_list above to create the `Layout1D` of every 1D CTI dataset we fit. This is used 
for visualizing the model-fit.
"""
layout_list = [
    ac.Layout1D(
        shape_1d=shape_native,
        region_list=region_list,
        prescan=prescan,
        overscan=overscan,
    )
    for norm in norm_list
]


"""
__Dataset__

We now load every cti-dataset, including a noise-map and pre-cti data containing the data before read-out and
therefore without CTI.
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
    )
    for layout, norm in zip(layout_list, norm_list)
]

"""
__Mask__

We now mask every 1D dataset, removing the FPR of each dataset so we use only the EPER to calibrate the CTI model.
"""
mask = ac.Mask1D.all_false(
    shape_slim=dataset_list[0].shape_slim,
    pixel_scales=dataset_list[0].pixel_scales,
)

mask = ac.Mask1D.masked_fpr_and_eper_from(
    mask=mask,
    layout=dataset_list[0].layout,
    settings=ac.SettingsMask1D(fpr_pixels=(0, 30)),
    pixel_scales=dataset_list[0].pixel_scales,
)

dataset_list = [dataset.apply_mask(mask=mask) for dataset in dataset_list]

"""
Plot the first dataset.
"""
dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0])
dataset_plotter.subplot_dataset()

"""
__Clocking__

The `Clocker` models the CCD read-out, including CTI. 
"""
clocker = ac.Clocker1D(express=5)

"""
__Model__

We now compose our CTI model, which represents the trap species and CCD volume filling behaviour used to fit the CTI 
1D data. In this example we fit a CTI model with:

 - Two `TrapInstantCapture`'s which capture electrons during clocking instantly in the parallel direction
 [4 parameters].
 
 - A simple `CCD` volume filling parametrization with fixed notch depth and capacity [1 parameter].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=5.

__Model Cookbook__

A full description of model composition, including CTI model customization, is provided by the model cookbook: 

https://pyautocti.readthedocs.io/en/latest/general/model_cookbook.html
"""
trap_0 = af.Model(ac.TrapInstantCapture)
trap_1 = af.Model(ac.TrapInstantCapture)
trap_2 = af.Model(ac.TrapInstantCapture)

trap_0.add_assertion(trap_0.release_timescale < trap_1.release_timescale)
trap_1.add_assertion(trap_1.release_timescale < trap_2.release_timescale)

trap_list = [trap_0, trap_1, trap_2]

ccd = af.Model(ac.CCDPhase)
ccd.well_notch_depth = 0.0
ccd.full_well_depth = 200000.0

model = af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))

"""
The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).

This confirms that the model is composed of three trap species.
"""
print(model.info)

"""
__Search__

The model is fitted to the data using the nested sampling algorithm Dynesty (https://dynesty.readthedocs.io/en/latest/).

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autocti_workspace/output/dataset_1d/species[x2]`.
"""
search = af.DynestyStatic(
    path_prefix=path.join("dataset_1d", dataset_name),
    name="species[x3]",
    nlive=50,
    walks=10,
)

"""
__Analysis__

The `AnalysisDataset1D` object defines the `log_likelihood_function` used by the non-linear search to fit the model 
to the `Dataset1D`dataset.

We sum the list to create an overall `Analysis` object, which we can use to fit the CTI model.
"""
analysis_list = [
    ac.AnalysisDataset1D(dataset=dataset, clocker=clocker) for dataset in dataset_list
]

analysis = sum(analysis_list)

analysis.n_cores = 1

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the folder `autocti_workspace/output/dataset_1d/species[x2]` for live outputs of the results of the fit, 
including on-the-fly visualization of the best fit model!
"""
result_list = search.fit(model=model, analysis=analysis)

"""
__Result__

The `info` attribute shows the model in a readable format (if this does not display clearly on your screen refer to
`start_here.ipynb` for a description of how to fix this).
"""
print(result_list.info)

"""
The result object also contains the fit corresponding to the maximum log likelihood solution in parameter space,
which can be used to visualizing the results.
"""
print(result_list[0].max_log_likelihood_instance.cti.trap_list[0].density)
print(result_list[0].max_log_likelihood_instance.cti.ccd.well_fill_power)

"""
Checkout `autocti_workspace/*/dataset_1d/modeling/results.py` for a full description of the result object.
"""
