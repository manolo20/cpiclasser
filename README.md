# cpiclasser:
**AUTHOR**: Ross Beck-MacNeil (CPD) - ross.beck-macneil@canada.ca

This Python package contains code for classifying scanner and webscraped data to the CPI classsification system.

## HOW TO USE:
If downloading directly from Gitlab, you can add the directory to your Python path. You can then import the module
and use its functions and classes to help develop machine learning models. 

Alternatively, you can access the version of the package that is stored on CPD's
shared drive at [P:\Research\Code Library (CoLi)]. This version of the package
is almost guaranteed to not be broken and should contain the latest pretrained
model as well as some notebooks that how to use the package to train models and
make predictions. Alternatively, you invoke the module from command line on the folder like so:

```
python cpiclasser in_path out_path
```

Here *in_path* is the .csv file that contains products that need to be classified, while *outpath* is path
of the file that will be created. The input file can contain any number of text variables such as product descriptions
or categories. It should not contain any other variables unless the *--index* option is specified, it should not contain any other variables.
The *--index* option allows the inclusion of a unique identifier such as UPC or SKU. For example:

```
python cpiclasser /my/home/dir/products_to_classify.csv /my/home/dir/classified_products.csv --index UPC
```

**N.B**: If accessing the package from the P drive, please make certain not
to update or delete any files. It is probably best to make a copy of the notebooks
to somewhere else.

## Dependcies :
This package has been developed with the following setup:
* Anaconda 4.4.0 (64 bit) for Python version 3.6 
* scikit-learn 0.19.1 
* numpy 1.14.0
* scipy 1.0.0
* keras 2.12
* tensorflow 1.3.0