# cpiclasser:
**AUTHOR**: Ross Beck-MacNeil (CPD)
**EMAIL**: ross.beck-macneil@canada.ca
This Python package contains code for classifying scanner and webscraped data to the CPI classsification system.

## HOW TO USE:
If downloading directly from Gitlab, you can add the directory to your Python path. You can then import the module
and use its functions and classes to help develop machine learning models. 

If you are accessing the package from the CPD shared drive, there should be a pretrained model in the "\models" folder.
In this case, the imported functionality will allow you make predictions using this pretrained model. Alternatively,
invoke the module from command line on the folder like so:

```python
python cpiclasser in_path out_path
```

Here *in_path* is the .csv file that contains products that need to be classified, while *outpath* is path
of the file that will be created. Additional details to follow. 

## REQUIRED (OR SUGGESTED?) PACKAGES/PYTHON VERSION:
Anaconda 4.4.0 (64 bit) for Python version 3.6 
* scikit-learn 0.19.1 
* numpy 1.14.0
* scipy 1.0.0
* keras 2.12
* tensorflow 1.3.0