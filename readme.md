# MILL
The code implementation of paper MILL: Mutual Verification with Large Language Models for Zero-Shot Query Expansion

## Prepare the Environment

```bash
conda create -n terrier python=3.8
conda activate terrier
pip install python-terrier openai retrying torch transformers pandarallel
```

The java home maybe need to be set when you use pyterrier

```bash
export JAVA_HOME= your_java_home
```

## Before Run the Code

please set the `api_key` and `base_url` in both `run.py` and `utils.py`

## Run the code

```bash
cd terrier
python run.py --dataset={} --qe={} --pos_lis={} --N={} --n={} --K={} --k={}
```

The description for the parameters:

* dataset: the dataset you want use. Candidate values: refer to [pyterrier doc](https://pyterrier.readthedocs.io/en/latest/datasets.html)
* qe: the query expansion method name.
* pos_lis: the (start,end) queries that you want to use in the dataset
* N: the number of generated candidate documents $N$
* n: the number of selected generated documents $n$
* K: the number of PRF candidate documents $K$
* k: the number of selected PRF documents $k$
