"""
__Example: Advanced Camera for Surveys (ACS) CTI Correction__

In this example, we'll use **PyAutoCTI** and **arcticpy* to correct CTI in examplle imaging from the Hubble Space
Telescope (HST) ACS instrument.
"""


"""
Load the ACS example dataset 'j8xi61meq_raw.fits' 'from .fits files, which is the dataset we will correct for CTI.
"""


import autocti as ac
import autocti.plot as aplt

dataset_type = "examples"
dataset_label = "acs"
dataset_name = "j8xi61meq_raw"
dataset_path = f"dataset/{dataset_type}/{dataset_label}/{dataset_name}"


"""
Imaging data observed on the ACS consists of four quadrants ('A', 'B', 'C', 'D') which have the following layout:

       <--------S-----------   ---------S----------->
    [] [========= 2 =========] [========= 3 =========] []          /\
    /    [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  /        |
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | Direction arctic
    P   [xxxxxxxxx B/C xxxxxxx] [xxxxxxxxx A/D xxxxxxx]  P         | clocks an image
    |   [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  |         | without any rotation
    \/  [xxxxxxxxxxxxxxxxxxxxx] [xxxxxxxxxxxxxxxxxxxxx]  \/        | (e.g. towards row 0
                                                                   | of the ndarrays)

For a ACS .fits file:

 - The images contained in hdu 1 correspond to quadrants B (left) and A (right).
 - The images contained in hdu 4 correspond to quadrants C (left) and D (right).

The `ImageACS` contains all the functionality we need to automatically load ACS data from the above .fits structure, 
without requiring us to manually set up the dataset for CTI correction. This includes:

 - Loading the input quadrant from the .fits file, from the correct .fits HDU extension.
 - Extracting the quadrant from this array.
 - Rotating this extracted quadrant such that for our CTI correction **arcticpy** clocks the image is the appropriate
   direction.
 - Loading and storing exposure information (e.g. date of observation, exposure time).
 - Using this information to convert the units of the image to counts, which are the appropriate unit for **arcticpy**.

"""


frame = ac.acs.ImageACS.from_fits(file_path=f"{dataset_path}.fits", quadrant_letter="A")


"""
We can use the **PyAutoCTI** visualization libary to plot this `Frame`. 

The image wouldn't not display clearly using the default color scale (which is the default color scale of matplotlib)
so we've updated it below.
"""


plotter = aplt.Plotter(cmap=aplt.ColorMap(norm="linear", norm_max=2800.0))
aplt.Frame(frame=frame, plotter=plotter)


"""
To correct the ACS data, we need a CTI model describing:
 
 - The properties of the `Trap`'s on the CCD that are responsible for CTI by capturing and releasing electrons during 
   read-out.
 - The volume-filling behaviour of electrons in CCD pixels, as this effects whether `Trap`'s are able to captrue them
   or not!
   
**PyAutoCTI** again has in-built tools for automatically loading the CTI model appropriate for correcting ACS data, 
using the date of observation loaded in the `ACSFrame` object to choose a model based on the date of observation.
"""

# traps = ac.acs.Traps.from_exposure_info(exposure_info=exposure_info)
# ccd = ac.acs.CCD()

traps = [ac.TrapInstantCapture(density=0.1, release_timescale=1.0)]
ccd = ac.CCD(full_well_depth=1000, well_fill_power=0.8, well_notch_depth=100.0)


"""
We next create a `Clocker`, which determines how we 'clock' the image to mimic the effects of CTI and in turn use this
clocked image to perform the CTI correction.

For simplicity, we'll use all default values except using 5 iterations, which means when correcting the image we clock 
it to reproduce CTI and use this image to correct the image, and repeat that process 5 times.
"""
clocker = ac.Clocker(iterations=5, serial_express=20)


"""
We can now pass the image and CTI model to the `Clocker` to remove CTI.
"""

frame_corrected = clocker.remove_cti(image=frame, serial_traps=traps, serial_ccd=ccd)
aplt.Frame(frame=frame_corrected, plotter=plotter)
