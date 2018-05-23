"""This package contains functions and classes which help develop,
train and deploy machine learning models based on Tensorflow
that classify scanner and webscraped data to the CPI product classification
"""

from .core import *
from . import train
from . import prediction
from . import layers

__author__ = "Ross Beck-MacNeil"
__email__ = "ross.beck-macneil@canada.ca"
__division__ = "CPD/DPC"
__version__ = "0.11"