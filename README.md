# Bigram-based Hybrid Attack (BHA)

This repository contains Keras implementations of the paper: BHA: A Bigram-based Hybrid Attack for Crafting Natural
Language Adversarial Samples.



## Requirements
* tensorflow == 1.15.2
* Keras == 2.2.4
* spacy == 2.1.4
* nltk == 3.4.5
* pandas == 0.23.4
* OpenHowNet == 0.0.1a8
* numpy == 1.15.4
* scikit_learn == 0.21.2
* If you did not download WordNet before, use `nltk.download('wordnet')` to do so.(Cancel the code comment on line 9 in `BHA_paraphrase. py`) 


## Usage

* Download [IMDB](https://drive.google.com/file/d/193BhcxN0fxClJl9xZyNaLhg5COc4lN4R/view?usp=sharing), [AG's News](https://drive.google.com/file/d/1cySABH3juxFB-YVRe-EK10yjnDS2Nl4F/view?usp=sharing) and [Yahoo! Answer](https://drive.google.com/file/d/1qvMfiB5vUSwR7lcAoPzXaEHtrIO9oaV1/view?usp=sharing) datasets and place them in `/data_set`.
* Download `glove.6B.100d.txt`from [google drive](https://drive.google.com/file/d/1eUV5XW-B0CKRAyHsnp89cHc-s0psRot-/view?usp=sharing) and place the file in `/`.
* Use our pretrained model stored in `/runs` or train models by running `training.py`.
* Run `bigram.py` to generate bigram candidates or use the prelearnd bigram data in `/bigram`.
* To ensure the quick reproducibility, we provide HowNet candidate in [google drive](https://drive.google.com/drive/folders/18b_opVai9igJMze4h_Ip0wewuW2czuRi?usp=sharing). To recalculate the HowNet candidate set, run `build_embeddings.py`, `gen_pos_tag.py`, `lemma.py` and `gen_candidates.py` under the `/hownet_candidates` for each dataset.
* Run `BHA_fool.py` to generate adversarial examples using BHA.
* If you want to train or fool different models, reset the argument in `training.py`and`fool.py`.
