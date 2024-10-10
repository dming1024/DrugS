
# DrugS Model

A DNN model offers a novel approach for drug screening and mechanism research in cancer therapy from a genomic perspective and demonstrates the potential applications of the DrugS model in personalized therapy and resistance mechanism elucidation.

Here are `notebook.ipynb` and example data in `dataset/` for model building and testing. You need to rebuild the autoencode model which we couldn't share here because of size limitation.

The source code are all available in `notebook.ipynb`

## Example 

using this model under `example/`, but you have to replace the `autoencoder_model` with `autoencoder_model.hd5` and move the `dataset/dnn_model_basedon_encoder30_gdsc1_gdsc2` into `example/models/dnn_model_basedon_encoder30_gdsc1_gdsc2`
 
```python

python test_v1.py --expression tmp.csv

```

+ tmp.csv, an example of gene expression data 
+ output, prediction_results.csv