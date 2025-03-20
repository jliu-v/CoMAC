# CoMAC: Conversational Agent for Multi-Source Auxiliary Context with Sparse and Symmetric Latent Interactions

This repository has the source codes for paper "CoMAC: Conversational Agent for Multi-Source Auxiliary Context with
Sparse and Symmetric Latent Interactions".

The data processing part of this implementation is based on the paper *
*[Call for Customized Conversation: Customized Conversation Grounding Persona and Knowledge](https://arxiv.org/abs/2112.08619)
**
and its **[source code](https://github.com/pkchat-focus/FoCus)**.

## Environment Setup

We use CUDA version `12.4`, and `python==3.9` and `torch==2.6.0` in our environment.
Users can create the virtual python environment with Pyenv and install the required packages with Poetry by using
the `pyproject.toml` file.

1. Install [Pyenv](https://github.com/pyenv/pyenv) and [Poetry](https://python-poetry.org/docs/) (or your own choice of other virtual environment management tools)

2. Make a virtual Python environment with `Pyenv`

```    
pyenv install 3.9.21
pyenv local 3.9.21
```

3. Activate virtual environment

```
poetry install --no-root
```   

Note: if you use a different version of CUDA driver, you will need to re-install `torch` that matches your CUDA version.
In most cases, Poetry should automatically match a distribution of torch that matches your CUDA version by the following
commands.

```
poetry remove torch
poetry add torch
```

## Downloading Dataset

In this repo, we provide a tiny dataset in `data/` for development purpose. We also included the pre-computed IDF
weights used by the `CoMAC` method.
The full dataset is not included in this repository due to the large file size.
Please refer to the original [FoCus Dataset paper](https://arxiv.org/abs/2112.08619)
to download the dataset and put into correct file path.

## Run Program

### Create IDF Weights
**Create IDF weights offline prior to training based on BART model vocabulary**

```commandline
python get_comac_tfidf.py \
    --model_name BART \
    --train_dataset_path data/train_focus_tiny.json \
    --output_idf_file_path data/term_idf_BART.csv
```

### Training

**Train with BART model**

```commandline
python train.py \
    --kp_method comac \
    --model_name BART \
    --train_dataset_path data/train_focus_tiny.json \
    --dev_dataset_path data/valid_focus_tiny.json \
    --n_epochs 2 \
    --lm_coef 10 \
    --kn_coef 1 \
    --ps_coef 1 \
    --sncolbert_sample_rate 0.35 \
    --flag comac_bart \
    --incontext \
    --train_batch_size 2 \
    --model_dir test_model_dir \
    --idf_file data/term_idf_BART.csv \
    --lr 1e-04 \
	--pg_label_weight 0.9 \
	--pg_loss_sample_p 0.2 > train_log_comac_bart.log 2>&1
```

### Testing

**Test with BART model**

```commandline
# Evaluation metrics except perplexity
python evaluate_test.py \
    --kp_method comac \
    --model_name BART \
    --model_checkpoint test_model_dir/comac_bart \
    --test_dataset_path data/valid_focus_tiny.json \
    --idf_file data/term_idf_BART.csv > test_log_comac_bart_MTL.log 2>&1

# Perplexity
python evaluate_test_ppl.py \
    --kp_method comac \
    --model_name BART \
    --model_checkpoint test_model_dir/comac_bart \
    --test_dataset_path data/valid_focus_tiny.json \
    --idf_file data/term_idf_BART.csv > test_log_comac_bart_MTL_ppl.log 2>&1
```


## References

This paper has been accepted to The 29th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD2025).
If you found this method or code helpful, please cite the paper "CoMAC: Conversational Agent for Multi-Source Auxiliary Context with
Sparse and Symmetric Latent Interactions" 

## License

See the LICENSE file in the root repo folder for more details.