"""
This Python script classifies products to the CPI classification
based on text descriptions of the product.
"""
import sys
import os
import argparse
import pandas as pd
#add grand-parent folder of file to python path, so can import prediction
PARENT_DIR = os.path.abspath(os.path.join(__file__, os.pardir))
GRANDPARENT_DIR = os.path.join(PARENT_DIR, os.pardir)
sys.path.insert(0, GRANDPARENT_DIR)
import cpiclasser


def main(in_path, out_path, index, n_preds):
    df = pd.read_csv(in_path, index_col=index, dtype=str, keep_default_na=False)
    print("\nRead in data. It has {} rows and {} columns\n".format(df.shape[0], df.shape[1]))
    #now load model
    model = cpiclasser.load_classer((os.path.join(PARENT_DIR, "model")))
    print("\nLoaded model")
    #Transform from dataframe into a list of tuples
    texts = [tuple(row) for row in df.values]
    top_class = model.predict_top_n(texts, n=n_preds)
    top_class[df.index.name] = df.index
    top_class.set_index(df.index.name, inplace=True)
    return top_class

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("in_path",
        help = "Path to the csv dataset that contains the products that need to be classified."
               " All columns except for the index column(s) will be used as features.")
    parser.add_argument("out_path", help = "Path to output file. It will be a csv file.")

    parser.add_argument("--index", nargs='*',
                        help = "Variable(s) that identify the product."
                        " Will be returned on the output dataset."
                        " All other variables will be used as features."
                        " If not provided, an index from 0 to number of"
                        " products will be generated.", default = None)
    parser.add_argument("--n_preds", type=int,
                        help = """Number of predicted classes to output.
                                  Columns are suffixed with the index, starting
                                  with 0 and going to n-1, in decreasing order of
                                  probability """, default = 1)
    
    args = parser.parse_args()

    in_path = args.in_path
    out_path = args.out_path
    index = args.index
    n_preds = args.n_preds
    top_class = main(in_path, out_path, index, n_preds)

    top_class.to_csv(out_path)


