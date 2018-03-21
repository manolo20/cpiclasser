# cpiclasser

**AUTHOR**: Ross Beck-MacNeil (CPD) ross.beck-macneil@canada.ca

**README LAST MODIFIED**: March 21, 2018

## DESCRIPTION:
This Python program classifies products to the CPI classification
based on text descriptions of the product.

## HOW TO USE:
If there is a pretrained model, you can call python on the folder like so:
```
python cpiclasser in_path out_path -index
```
Otherwise, you can import it a module:
```python
python cpiclasser in_path out_path -index
```

## FOR HELP (and explanation of arguments):
python cpiclasser -h

## REQUIRED (OR SUGGESTED?) PACKAGES /PYTHON VERSION:
Anaconda 4.4.0 (64 bit) for Python version 3.6 
scikit-learn 0.19.1 
numpy 1.14.0
scipy 1.0.0
keras 2.12
tensorflow 1.3.0