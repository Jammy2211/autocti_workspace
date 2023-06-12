The ``data_preparation`` folder contains example scripts for data preparation charge injection data before  CTI calibration.

Files (Beginner)
----------------

- ``pre_cti``: Use the FPRs to estimate the charge injected into each column.
- ``serial_cti``: Use a previously estimated serial CTI model to correct the data before estimating the pre CTI image.
- ``cosmic_ray_flagging``: Flag cosmic rays in CTI calibration data.
- ``bias_subtraction``: Perform bias subtraction on the data, if not already performed
- ``pre_cti_and_cosmics``: Flag cosmic rays in CTI calibration and estimate the pre CTI image simultaneously.