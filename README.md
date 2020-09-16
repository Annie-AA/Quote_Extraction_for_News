# Quote_Extraction_for_News
Using a neural network model to extract quotes from news feeds
## Train a spaCy sequence labelling model using PARC
#### Extract quotes
The ```parc_spacy_model.py``` is to train a spaCy sequence labelling model on PARC data (focused on articles in news domain) first and then we can extract quotes from each article's text by asking the model to annotate the role of each word in a sentence for us. The articles in the news genre are listed in ```news_genre_file_num.txt```, according to [the news section](http://www.let.rug.nl/~bplank/metadata/genre_files_updated.html). The PARC corpus was obtained by contacting directly the author of this paper (Pareti, 2016). Run the ```parc_spacy_model.py``` along with the path to the PARC dataset to generate annotated datasets.

#### Optional arguments
```
python3 parc_spacy_model.py --domain news_genre_file_num.txt --parc_dataset /Users/anniean/PARC3_complete --raw_input ../data/rawtext --output ../validation/V4.0 --epoch 1

usage: parc_spacy_model.py [-h] [--domain] [--parc_dataset] [--raw_input] [--epoch] [--output]

optional arguments:
  -h, --help        Show this help message and exit
  --domain          Domain of articles to use
  --parc_dataset    Path to PARC article data
  --raw_input       Path to raw news article data
  --epoch           Number of training procedures implemented
  --output          Path to raw news article data
  
```  
  
This results in added named-entity fields in the output JSON as follows:
```    
   {
        "speaker": "Kim",
        "verb": "told",
        "quote": "\"Honestly, it feels like we're living our worst nightmare right now,\"",
        "speaker_index": "(596,599)",
        "verb_index": "(600,604)",
        "quote_index": "(526,595)"
    }
```  

#### Validate the extracted quotes vs. human-annotated quotes
This outputs the quote extraction's validation results for two thresholds as shown below.
```
Quote Extraction - 0.3, Precision: 82.7%
Quote Extraction - 0.3, Recall: 75.4%
Quote Extraction - 0.3, F1 Score: 78.9%
Speaker Match - 0.3, Accuracy: 79.8%
Verb Match - 0.3, Accuracy: 86.1%
------------------------
Quote Extraction - 0.8, Precision: 80.0%
Quote Extraction - 0.8, Recall: 72.9%
Quote Extraction - 0.8, F1 Score: 76.3%
Speaker Match - 0.8, Accuracy: 80.7%
Verb Match - 0.8, Accuracy: 87.1%
------------------------
```

#### Workflow of the system
![Workflow](/img/workflow.JPG)

#### Reference
Pareti, S. (2016). PARC 3.0: A Corpus of Attribution Relations. In *Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16)*. pages 3914-3920. Portorož, Slovenia.
