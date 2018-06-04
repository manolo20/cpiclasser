# cpiclasser:
**AUTHOR**: Ross Beck-MacNeil (CPD) - ross.beck-macneil@canada.ca

This Python package contains code that helps with classifying scanner and webscraped data to the CPI classsification system. It is also general enough to work for other text classificaton problems.

## HOW TO USE:
You can access a preliminary, pretrained, model through the Github release functionality.
This model will classify products to the published food classes. You invoke the module from command line
on the folder like so:

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

Please note that the input file must be encoded as utf-8. 


## Dependencies :
This package has been developed with the following setup:
* Anaconda 4.4.0 (64 bit) for Python version 3.6 
* scikit-learn 0.19.1 
* numpy 1.14.0
* scipy 1.0.0
* keras 2.12
* tensorflow 1.3.0
