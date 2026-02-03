# Swedish BERT for multi-label emotion classification 

This project shows how to fine-tune and use a BERT-model for multi-label emotion classification of Swedish texts.

## Fine-tuning

See `train.py` for how the model was trained. 

## Dataset

The model is trained on the Swedish subset of the [Brigther](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories) dataset. Each sentence can be assigned one or multiple emotion labels. More information about the dataset can be found here: <br>
[BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages](https://aclanthology.org/2025.acl-long.436/) (Muhammad et al., ACL 2025) 

## Model 

The resulting model is based on [KB-BERT](https://huggingface.co/KB/bert-base-swedish-cased) and can 
be found [here](https://huggingface.co/sbx/KB-bert-base-swedish-cased_emotions_brighter). 

The model classifies six emotion labels:
- Anger
- Disgust
- Fear
- Joy
- Sadness
- Surprise

Example: 'Den här produkten lever inte alls upp till mina förväntningar, skäms!' Labels: |anger,0.97|disgust,0.97|surprise,0.35|

## Usage

See `demo.ipynb` for a complete example of loading and using the model for inference.

The model is also available as a plugin the [Sparv](https://spraakbanken.gu.se/sparv/) tool, which can be found [here](https://github.com/spraakbanken/sparv-sbx-emotions-brighter/) 

## Performance

The model achieves:
- F1 micro: 0.77
- F1 macro: 0.67
- Accuracy: 0.68

See `demo.ipynb` for detailed per-label results and dataset information.


