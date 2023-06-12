The ``modeling/non_uniform`` folder contains example scripts showing how to fit a model to non-uniform charge injection
CTI calibration data.

Non-uniform charge injection data assumes the injected charge in every column are drawn from a common parent Gaussian
distribution, with the scatter in this Gaussian leading different columns to have different levels of injected charge.

Files (Beginner)
----------------

- ``parallel_x2.py``: Fits a parallel CTI model with two trap species.
- ``serial_x2.py``: Fits a serial CTI model with two trap species.