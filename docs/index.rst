NiftyML: PET-MR with Machine Learning
=====================================

|Version| |Py-Versions| |DOI| |Licence| |Tests| |Coverage|

Installation
------------

.. code:: sh

    pip install tensorflow-gpu  # or CPU version: `pip install tensorflow`
    pip install niftyml


Usage
-----

.. code:: python

    from niftypet import ml
    model = ml.dcl2021(...)
    model.fit(...)

See `examples <./examples/>`_ for more.


Acknowledgements
----------------

This work was supported in part by the King's College London and Imperial College London EPSRC Centre for Doctoral Training in Medical Imaging under Grant [EP/L015226/1], in part by the Wellcome EPSRC Centre for Medical Engineering at King's College London under Grant [WT 203148/Z/16/Z], in part by EPSRC under Grant [EP/M020142/1], in part by the National Institute for Health Research (NIHR) Biomedical Research Centre Award to Guy's and St Thomas' NHS Foundation Trust in partnership with King's College London, and in part by the NIHR Healthcare Technology Co-operative for Cardiovascular Disease at Guy's and St Thomas' NHS Foundation Trust. The views expressed are those of the author(s) and not necessarily those of the NHS, the NIHR or the Department of Health.

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4654096.svg
   :target: https://doi.org/10.5281/zenodo.4654096
.. |Licence| image:: https://img.shields.io/pypi/l/niftyml.svg?label=licence
   :target: https://github.com/NiftyPET/NiftyML/blob/master/LICENCE
.. |Tests| image:: https://img.shields.io/github/workflow/status/NiftyPET/NiftyML/Test?logo=GitHub
   :target: https://github.com/NiftyPET/NiftyML/actions
.. |Coverage| image:: https://codecov.io/gh/NiftyPET/NiftyML/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/NiftyPET/NiftyML
.. |Version| image:: https://img.shields.io/pypi/v/niftyml.svg?logo=python&logoColor=white
   :target: https://github.com/NiftyPET/NiftyML/releases
.. |Py-Versions| image:: https://img.shields.io/pypi/pyversions/niftyml.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/niftyml
