The ``data_preparation`` folder contains example scripts for preprocessing charge injection data before  CTI calibration.

Files (Beginner)
----------------

- ``preprocess_1_pre_cti``: Use the FPRs to estimate the charge injected into each column.
- ``preprocess_2_serial_cti``: Use a previously estimated serial CTI model to correct the data before estimating the pre CTI image.
- ``preprocess_3_cosmic_ray_flagging``: Flag cosmic rays in CTI calibration data.
- ``preprocess_optional_bias_subtraction``: Perform bias subtraction on the data, if not already performed
- ``preprocess_optional_pre_cti_and_cosmics``: Flag cosmic rays in CTI calibration and estimate the pre CTI image simultaneously.