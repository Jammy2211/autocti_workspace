"""
Modeling: Start Here
====================

This script is the starting point for modeling of 1D CTI datasets and it provides an overview 
of the modeling API.

After reading this script, the `features`, `customize` and `searches` folders provide example for performing CTI
modeling in different ways and customizing the analysis.

__Model__

In this script, we will fit a 1D CTI Dataset to calibrate a CTI model, where:

 - The CTI model consists of multiple parallel `TrapInstantCapture` species.
 - The `CCD` volume filling is a simple parameterization with just a `well_fill_power` parameter.
 
 __Plotters__

To produce images of the data `Plotter` objects are used, which are high-level wrappers of matplotlib
code which produce high quality visualization of strong CTIes.

The `PLotter` API is described in the script `autoCTI_workspace/*/plot/start_here.py`.

__Simulation__

This script fits a simulated `Imaging` dataset of a strong CTI, which is produced in the
script `autoCTI_workspace/*/imaging/simulators/start_here.py`
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

We begin by loading the CTI dataset `species_x2` via .fits files, which is a data format used by astronomers to 
store images.

A number of steps are performed below which prepare us to load the dataset beforehand. 
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "dataset_1d", dataset_name)

"""
__Shape__

The 1D shape of each 1D dataset, where the 1D dataset we will load is 200 pixels long.
"""
shape_native = (200,)

"""
__Regions__

Use `Region1D` objects to define the locations of the prescan and overscan on the 1D data. 

1D regions are defined as a tuple of the form (x0, x1) = (left-pixel, right-pixel), where the integer values of this
tuple are used to perform NumPy array indexing of the 1D data.

For example, if the overscan of 1D data is between pixels 40 and 50, its region is `region=(40, 50)`.

These are used to visualize these regions of the 1D CTI dataset during the model-fit and customize aspects of the 
model-fit.
"""
prescan = ac.Region1D(region=(0, 10))
overscan = ac.Region1D(region=(190, 200))

"""
__FPR / EPER__

Specify the charge regions on the 1D CTI Dataset, corresponding to where an injected signal is present that has its 
electrons captured and trailed by CTI.

This is referred to as a the "First Pixel Response" (FPR), with the trails of electrons which appear after it 
referred to as the "Extended Pixel Edge Response" (EPER).

This dataset has only one charge region (which is between pixels 10 and 20), but more regions can be specified if
required.
"""
region_list = [(10, 20)]

"""
__Normalizations__

Specify the normalization of the charge in every individual 1D CTI dataset. 

This is not used internally by **PyAutoCTI**, and only required for loading the dataset because the dataset file
names use the normalizations.
"""
norm_list = [100, 5000, 25000, 200000]

"""
The total number of 1D CTI datasets that are fitted.
"""
total_datasets = len(norm_list)

"""
__Layout__

We now create a `Layout1D` object for every 1D dataset fitted in this script.

This object contains all functionality associated with the layout of the data (e.g. where the FPR is, where the
EPERs are, where the overscans are, etc.). 

This is used for performing tasks like extracting a small region of the data for visualization.
"""
layout_list = [
    ac.Layout1D(
        shape_1d=shape_native,
        region_list=region_list,
        prescan=prescan,
        overscan=overscan,
    )
    for i in range(total_datasets)
]


"""
__Dataset__

We now use a `Dataset1D` object to load every 1D CTI dataset, including a noise-map and pre-cti data containing the d
ata before read-out and therefore without CTI. 

The `pixel_scales` define the arc-second to pixel conversion factor of the image, which for the dataset we are using 
is 0.1" / pixel. This is used for visualization only, specifically to convert axis labels from pixels to arc-seconds.
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
Use a `Dataset1DPlotter` to the plot the data, including: 

 - `data`: The 1D CTI data.
 - `noise_map`: The noise-map of the data, which quantifies the noise in every pixel as their RMS values.
 - `pre_cti_data`: The data before CTI, which has CTI added to it for every CTI model, which is compared to the data. 
 - `signal_to_noise_map`: Quantifies the signal-to-noise in every pixel.
"""
dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0])
dataset_plotter.subplot_dataset()

"""
__Mask__

We apply a `Mask1D` to the dataset, which defines the regions of the data we fit the CTI model to the data. 

We mask the FPR of each dataset, such that this fit will only the EPER to calibrate the CTI model.
"""
mask = ac.Mask1D.all_false(
    shape_slim=dataset_list[0].shape_slim,
    pixel_scales=dataset_list[0].pixel_scales,
)

mask = ac.Mask1D.masked_fpr_and_eper_from(
    mask=mask,
    layout=dataset_list[0].layout,
    settings=ac.SettingsMask1D(fpr_pixels=(0, 10)),
    pixel_scales=dataset_list[0].pixel_scales,
)

dataset_list = [dataset.apply_mask(mask=mask) for dataset in dataset_list]

"""
By plotting the masked data, the mask removes the FPR of the data and now shows only the EPER trails.
"""
dataset_plotter = aplt.Dataset1DPlotter(dataset=dataset_list[0])
dataset_plotter.subplot_dataset()

"""
__Clocker / arCTIc__

To model the CCD clocking process, including CTI, we use  arCTIc, or the "algorithm for Charge Transfer Inefficiency 
clocking".

arCTIc is written in c++ can be used standalone outside of **PyAutoCTI** as described on its GitHub 
page (https://github.com/jkeger/arctic). **PyAutoCTI** uses arCTIc's built-in Python wrapper.

In **PyAutoCTI** we call arCTIc via a `Clocker` object, which is a Python class that wraps arCTIc. This class has 
many optional inputs that customize how clocking is performed, but we'll omit these for now to keep things simple.

For this example, we only input the `express` parameter, which determines how many electrons are clocked per cycle
and trades off speed for accuracy. For this example we use `express=5`, which is a good balance.
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

__Model Composition__

The API below for composing a CTI model uses the `Model` and `Collection` objects, which are imported from 
**PyAutoCTI**'s parent project **PyAutoFit** 

The API is fairly self explanatory and is straight forward to extend, for example adding more light profiles
to the CTI and source or using a different mass profile.

__Model Cookbook__

A full description of model composition, including CTI model customization, is provided by the model cookbook: 

https://pyautocti.readthedocs.io/en/latest/general/model_cookbook.html
"""
trap_0 = af.Model(ac.TrapInstantCapture)
trap_1 = af.Model(ac.TrapInstantCapture)

trap_0.add_assertion(trap_0.release_timescale < trap_1.release_timescale)

trap_list = [trap_0, trap_1]

ccd = af.Model(ac.CCDPhase)
ccd.well_notch_depth = 0.0
ccd.full_well_depth = 200000.0

model = af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))

"""
The `info` attribute shows the model in a readable format.

The `info` below may not display optimally on your computer screen, for example the whitespace between parameter
names on the left and parameter priors on the right may lead them to appear across multiple lines. This is a
common issue in Jupyter notebooks.

The`info_whitespace_length` parameter in the file `config/general.yaml` in the [output] section can be changed to 
increase or decrease the amount of whitespace (The Jupyter notebook kernel will need to be reset for this change to 
appear in a notebook).
"""
print(model.info)

"""
__Search__

The CTI model is fitted to the data using a non-linear search. 

All examples in the autoCTI workspace use the nested sampling algorithm 
Nautilus (https://nautilus.readthedocs.io/en/latest/), which extensive testing has revealed gives the most accurate
and efficient CTI modeling results.

We make the following changes to the Nautilus settings:

 - Increase the number of live points, `nlive`, from the default value of 50 to 100. 
 - Increase the number of random walks per live point, `walks` from the default value of 5 to 10. 

These are the two main Nautilus parameters that trade-off slower run time for a more reliable and accurate fit.
Increasing both of these parameter produces a more reliable fit at the expense of longer run-times.

__Customization__

The folders `autoCTI_workspace/*/imaging/modeling/searches` gives an overview of alternative non-linear searches,
other than Nautilus, that can be used to fit CTI models. They also provide details on how to customize the
model-fit, for example the priors.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autoCTI_workspace/output/imaging/modeling/simple/light[bulge_disk]_mass[sie]_source[bulge]/unique_identifier`.

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.
 
An identical combination of model and search generates the same identifier, meaning that rerunning the script will use 
the existing results to resume the model-fit. In contrast, if you change the model or search, a new unique identifier 
will be generated, ensuring that the model-fit results are output into a separate folder.

We additionally want the unique identifier to be specific to the dataset fitted, so that if we fit different datasets
with the same model and search results are output to a different folder. We achieve this below by passing 
the `dataset_name` to the search's `unique_tag`.

__Number Of Cores__

We include an input `number_of_cores`, which when above 1 means that Nautilus uses parallel processing to sample multiple 
CTI models at once on your CPU. When `number_of_cores=2` the search will run roughly two times as
fast, for `number_of_cores=3` three times as fast, and so on. The downside is more cores on your CPU will be in-use
which may hurt the general performance of your computer.

You should experiment to figure out the highest value which does not give a noticeable loss in performance of your 
computer. If you know that your processor is a quad-core processor you should be able to use `number_of_cores=4`. 

Above `number_of_cores=4` the speed-up from parallelization diminishes greatly. We therefore recommend you do not
use a value above this.

For users on a Windows Operating system, using `number_of_cores>1` may lead to an error, in which case it should be 
reduced back to 1 to fix it.
"""
search = af.Nautilus(
    path_prefix=path.join("dataset_1d", dataset_name), name="species[x2]", n_live=100
)

"""
__Analysis__

We next create an `AnalysisImaging` object, which can be given many inputs customizing how the CTI model is 
fitted to the data (in this example they are omitted for simplicity).

Internally, this object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `Imaging` dataset. 
"""
analysis_list = [
    ac.AnalysisDataset1D(dataset=dataset, clocker=clocker) for dataset in dataset_list
]

"""
By summing this list of analysis objects, we create an overall `Analysis` which we can use to fit the CTI model, where:

 - The log likelihood function of this summed analysis class is the sum of the log likelihood functions of each 
 individual analysis object.

 - The summing process ensures that tasks such as outputting results to hard-disk, visualization, etc use a 
 structure that separates each analysis.
"""
analysis = sum(analysis_list)

"""
We can parallelize the likelihood function of these analysis classes, whereby each evaluation will be performed on a 
different CPU.
"""
analysis.n_cores = 1

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs the 
nautilus non-linear search in order to find which models fit the data with the highest likelihood.

__Output Folder__

Now this is running you should checkout the `autoCTI_workspace/output` folder.

This is where the results of the search are written to your hard-disk (in the `tutorial_1_non_linear_search` folder). 
When its completed, images, results and information about the fit appear in this folder, meaning that you don't need 
to keep running Python code to see the result.

__On The Fly Outputs__

Even when the search is running, information about the highest likelihood model inferred by the search so far 
is output to this folder on-the-fly. 

If you navigate to the folder: 

 `output/dataset_1d/simple` 
 
Even before the search has finished, you will see:

 1) The `images` folder, where images of the highest likelihood CTI model are output on-the-fly. This includes the
 `FitImaging` subplot we plotted in the previous chapter, which therefore gives a real sense of 'how good' the model
 fit is.
 
 2) The `samples` folder, which contains a `.csv` table of every sample of the non-linear search as well as other 
 information. 
 
 3) The `model.info` file, which lists the CTI model, its parameters and their priors (discussed in the next tutorial).
 
 4) The `model.results` file, which lists the highest likelihood CTI model and the most probable CTI model with 
 errors (this outputs on-the-fly).
 
 5) The `search.summary` file, which provides a summary of the non-linear search settings and statistics on how well
 it is performing.
"""
result_list = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format.

[Above, we discussed that the `info_whitespace_length` parameter in the config files could b changed to make 
the `model.info` attribute display optimally on your computer. This attribute also controls the whitespace of the
`result.info` attribute.]
"""
print(result_list.info)

"""
The `Result` object also contains:

 - The model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `CTI1D` and `FitDataset1D` objects.
 - Information on the posterior as estimated by the `Nautilus` non-linear search. 
"""
print(result_list[0].max_log_likelihood_instance.cti.trap_list[0].density)
print(result_list[0].max_log_likelihood_instance.cti.ccd.well_fill_power)

for result in result_list:
    fit_plotter = aplt.FitDataset1DPlotter(fit=result.max_log_likelihood_fit)
    fit_plotter.subplot_fit()

"""
Checkout `autocti_workspace/*/dataset_1d/modeling/results.py` for a full description of the result object.
"""
