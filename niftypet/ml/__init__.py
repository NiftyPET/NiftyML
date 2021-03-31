"""
Neuro-image Machine Learning.
"""
__author__ = "Casper da Costa-Luis <imaging@cdcl.ml>"
__date__ = "2021"
# version detector. Precedence: installed dist, git, 'UNKNOWN'
try:
    from ._dist_ver import __version__
except ImportError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="../..", relative_to=__file__)
    except (ImportError, LookupError):
        __version__ = "UNKNOWN"

from .layers import *  # NOQA, yapf: disable
from .models import *  # NOQA, yapf: disable
