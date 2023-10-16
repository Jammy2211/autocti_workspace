PyAutoCTI Workspace
====================

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Jammy2211/autocti_workspace/HEAD

|binder|

`Installation Guide <https://pyautocti.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautocti.readthedocs.io/en/latest/index.html>`_ |

Welcome to the **PyAutoCTI** Workspace. You can get started right away by going to the `autocti workspace
Binder <https://mybinder.org/v2/gh/Jammy2211/autocti_workspace/HEAD>`_.
Alternatively, you can get set up by following the installation guide on our `readthedocs <https://pyautocti.readthedocs.io/>`_.

Getting Started
---------------

We recommend new users begin by looking at the following notebooks: 

- ``notebooks/overview``: Examples giving an overview of **PyAutoCTI**'s core features.

Installation
------------

If you haven't already, install `PyAutoCTI via pip or conda <https://pyautocti.readthedocs.io/en/latest/installation/overview.html>`_.

Next, clone the ``autocti workspace`` (the line ``--depth 1`` clones only the most recent branch on
the ``autocti_workspace``, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autocti_workspace
   git clone https://github.com/Jammy2211/autocti_workspace --depth 1
   cd autocti_workspace

Run the ``welcome.py`` script to get started!

.. code-block:: bash

   python3 welcome.py

Workspace Structure
-------------------

The workspace includes the following main directories:

- ``notebooks``: **PyAutoCTI** examples written as Jupyter notebooks.
- ``scripts``: **PyAutoCTI** examples written as Python scripts.
- ``config``: Configuration files which customize **PyAutoCTI**'s behaviour.
- ``dataset``: Where data is stored, including example datasets distributed.
- ``output``: Where the **PyAutoCTI** analysis and visualization are output.

The examples in the ``notebooks`` and ``scripts`` folders are structured as follows:

- ``overview``: Examples giving an overview of **PyAutoCTI**'s core features.

- ``dataset_1d``: Examples for analysing and simulating 1D CTI datasets (e.g. warm pixels).
- ``imaging_ci``: Examples for analysing and simulating CCD charge injection imaging data (e.g. Euclid).

- ``results``: Examples using the results of a model-fit, including database tools which loads results from hard-disk.
- ``plot``: An API reference guide for **PyAutoCTI**'s plotting tools.
- ``misc``: Miscellaneous scripts for specific cti analysis.

Inside these packages are packages titled ``advanced`` which only users familiar **PyAutoCTI** should check out.

In the ``dataset_1d`` and ``imaging_ci`` folders you'll find the following packages:

- ``correction``: Examples of how to correct data with a CTI model.
- ``modeling``: Examples of how to fit a CTI model to data via a non-linear search.
- ``simulators``: Scripts for simulating realistic CTI calibration datasets.
- ``data_preparation``: Tools to preprocess CTI calibration data before an analysis (e.g. cosmic ray flagging).

- ``advanced/chaining``: Advanced modeling scripts which chain together multiple non-linear searches.

The files ``README.rst`` distributed throughout the workspace describe what is in each folder.

Getting Started
---------------

We recommend new users begin with the example notebooks / scripts in the *overview* folder and the **HowToCTI**
tutorials.

Workspace Version
-----------------

This version of the workspace is built and tested for using **PyAutoCTI v2023.9.18.4**.

Contribution
------------
To make changes in the tutorial notebooks, please make changes in the corresponding python files(.py) present in the
``scripts`` folder of each chapter. Please note that  marker ``# %%`` alternates between code cells and markdown cells.


Support
-------

Support for installation issues, help with cti modeling and using **PyAutoCTI** is available by
`raising an issue on the autocti_workspace GitHub page <https://github.com/Jammy2211/autocti_workspace/issues>`_. or
joining the **PyAutoCTI** `Slack channel <https://pyautocti.slack.com/>`_, where we also provide the latest updates on
**PyAutoCTI**.

Slack is invitation-only, so if you'd like to join send an `email <https://github.com/Jammy2211>`_ requesting an
invite.
