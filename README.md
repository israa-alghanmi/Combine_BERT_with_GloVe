# BERT + GloVe
This repo contains an implementation example of the model proposed in our paper:  Combining BERT with Static Word Embeddings for Categorizing Social Media 
http://noisy-text.github.io/2020/pdf/2020.d200-1.5.pdf


# Usage:
Installing the required packages
``` 
pip install requirements.txt
``` 

Optional Arguments:
Arguments available for providing paths to the dataset, GloVe and the fine-tuned BERT ( NOTE: fine-tuning BERT separately and saving a checkpoint are needed before running this code):
``` 
python main.py --help

usage: main.py [-h] [--traint TRAINING_PATH_TEXT]
               [--trainl TRAINING_PATH_LABELS] [--validt VALIDATION_PATH_TEXT]
               [--validl VALIDATION_PATH_LABELS] [--textt TESTING_PATH_TEXT]
               [--testl TESTING_PATH_LABELS] [--glovepath GLOVE_FILE_PATH]
               [--finetunedbert FINETUNED_BERT_PATH] 
               

optional arguments:
  -h, --help            show this help message and exit
  --traint TRAINING_PATH_TEXT
                        The path of training data text
  --trainl TRAINING_PATH_LABELS
                        The path of training data labels
  --validt VALIDATION_PATH_TEXT
                        The path of validation data text
  --validl VALIDATION_PATH_LABELS
                        The path of validation data labels
  --textt TESTING_PATH_TEXT
                        The path of testing data text
  --testl TESTING_PATH_LABELS
                        The path of testing data labels
  --glovepath GLOVE_FILE_PATH
                        The path of Glove vectors file
  --finetunedbert FINETUNED_BERT_PATH
                        The path of finetuned BERT checkpoint
``` 

If you find this useful, please cite the paper:

``` 
@inproceedings{alghanmi2020combining,
  title={Combining BERT with Static Word Embeddings for Categorizing Social Media},
  author={Alghanmi, Israa and Anke, Luis Espinosa and Schockaert, Steven},
  booktitle={Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)},
  pages={28--33},
  year={2020}
}

```
