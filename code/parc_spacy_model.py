import os
import json
import nltk
import spacy
import argparse
import numpy as np
from copy import deepcopy
from bs4 import BeautifulSoup
from spacy.util import minibatch
from random import shuffle, seed
from collections import defaultdict



# ----- Formatting Functions
def join_punctuation(seq, 
                     suffix={'.',',',';',':','?','!',"'s", "%","'s", "'re","n't"}, 
                     quote_char = ["'",'"'], 
                     other_prefix_char = ["$"]):
    """Yields strings with the punctuation & affix already joined on.
    
    :param lst seq: A list of words
    :param lst suffix: A set of elements placed at the end of a word
    :param lst quote_char: A list of quotation marks
    :param lst other_prefix_char: A list of elements placed at the beginning of a word
    :return: Yields a list of strings where the punctuation and affix already joined on
    """
    characters = set(suffix)
    seq = iter(seq)
    current = next(seq)
    # Initiate quote_flag. quote_flag = 1 if have seen a quotation mark, reset to 0 when it sees the second quotation mark
    quote_flag = 0 
    for nxt in seq:
        if nxt in suffix:
            current += nxt
        elif quote_flag == 1 and nxt in quote_char:
            current += nxt
            quote_flag = 0
        elif current in quote_char and quote_flag == 0: 
            current += nxt
            quote_flag = 1
        elif current in other_prefix_char:
            current += nxt
        else:
            yield current
            current = nxt
    yield current


def preprocess_text(txt):
    """Reformat text

    :param str txt: A text 
    :return: A reformatted text
    """
    txt = txt.replace(u'\xa0', u' ')
    txt = txt.replace(u'\xe9', u' ')
    # To fix the problem of not breaking at \n
    txt = txt.replace("\n", ".\n ")
    # To remove potential duplicate dots
    txt = txt.replace("..\n ", ".\n ")
    txt = txt.replace(". .\n ", ".\n ")
    txt = txt.replace("  ", " ")
    # Replace double quotes
    txt = txt.replace("”", '"')
    txt = txt.replace("“", '"')
    txt = txt.replace("〝", '"')
    txt = txt.replace("〞", '"')
    txt = txt.replace("’’", '"') 
    return txt


# ========== Prepare PARC data to train spaCy sequence labelling model ==========
def get_spacy_formatted_data(directory, domain):
    """Return all files in the given domain in spaCy format from the directory. 'domain' is the filename list of a given domain/genre.
    
    :param str directory: Path to PARC 
    :param lst domain: List of article filenames of a domain
    :return: A list of tuples, each tuple includes a sentence, and a dictionary with key = "entities", value = list of entities tuples (start_index, end_index, entity_name) for the sentence
    """
    spacy_sent = []
    characters=['.',',',';',':','?','!', "%"]
    special = ["'s", "'re","n't"]
    quote_mark = ["'",'"']
    prefix = ["$"]
    for foldername in sorted(os.listdir(directory)):
        if not foldername.startswith("."):
            for filename in sorted(os.listdir(directory+foldername)):
                if filename.split('.')[0] in domain:
                    file_path = directory+foldername + '/'+filename
                    file = open(file_path).read()
                    soup = BeautifulSoup(file, "lxml-xml")
                    for sent_tag in soup.find_all("SENTENCE"):
                        all_words = []
                        first_cue_flag = True
                        first_source_flag = True
                        first_content_flag = True
                        spacy_attribution_dict = defaultdict(list)
                        attribution_dict = defaultdict(list)
                        index_pointer = 0
                        first_quote_flag = 0
                        for word_tag in sent_tag.find_all("WORD"):
                            tagged_flag = 0
                            # Ensure the quotation marks are correct as '"'                           
                            word = '"' if word_tag['text'] in ["``", "''"] else word_tag['text']
                            if len(word_tag.contents)>0:
                                for attribution in word_tag.contents:
                                    # Exclude the Nested Annotation
                                    if "Attribution_relation" in attribution["id"] or "PDTB" in attribution["id"]: 
                                        if word in characters:
                                            # Remove the extra space before the punctuations in characters = ['.',',',';',':','?','!', "%"]
                                            index_pointer = index_pointer - 1 
                                            start_idx = index_pointer
                                            end_idx = index_pointer + len(word)
                                        # For the second quotation mark of a quotation marks pair in a sentence, remove the extra space before it
                                        elif word in quote_mark and first_quote_flag == 1: 
                                            index_pointer = index_pointer - 1 
                                            start_idx = index_pointer
                                            end_idx = index_pointer + len(word)
                                            first_quote_flag ==0
                                        elif word in special:
                                            index_pointer = index_pointer - 1
                                            start_idx = index_pointer
                                            end_idx = index_pointer + len(word)
                                            if attribution_dict[attribution.attributionRole["roleValue"].upper()] != []:
                                                attribution_dict[attribution.attributionRole["roleValue"].upper()][-1][-1] = end_idx
                                            else:
                                                attribution_dict[attribution.attributionRole["roleValue"].upper()].append([start_idx,end_idx]) 
                                        else:
                                            start_idx = index_pointer
                                            end_idx = index_pointer + len(word)
                                        if attribution.attributionRole["roleValue"] == "cue" and word not in special: 
                                            attribution_dict["CUE"].append([start_idx,end_idx])
                                        elif attribution.attributionRole["roleValue"] == "source" and word not in special:
                                            attribution_dict["SOURCE"].append([start_idx,end_idx])
                                        elif attribution.attributionRole["roleValue"] == "content" and word not in special:
                                            attribution_dict["CONTENT"].append([start_idx,end_idx])
                                        tagged_flag = 1
                            if word in quote_mark and first_quote_flag == 0:
                                index_pointer += len(word)
                                first_quote_flag = 1
                            elif word in quote_mark and first_quote_flag == 1 and tagged_flag ==0:
                                index_pointer = index_pointer - 1 + len(word)
                            elif word in prefix:
                                index_pointer += len(word)
                            elif word in characters and tagged_flag == 0:
                                index_pointer = index_pointer - 1 + len(word) + 1
                            elif word in special and tagged_flag == 0:
                                index_pointer = index_pointer - 1 + len(word)
                            else:
                                index_pointer += len(word) + 1
                            all_words.append(word)
                        spacy_attribution_dict['entities'] = [sub_lst+[key] for key,value in attribution_dict.items() for sub_lst in value]
                        sentence = ' '.join(join_punctuation(all_words))
                        spacy_sent.append(tuple((sentence,spacy_attribution_dict)))
    return spacy_sent


def evaluate(sys_spacy_data,gold_spacy_data):
    """Compute the precision, reacall, and fscore of a spaCy sequence labelling model.

    :param lst sys_spacy_data: Dataset with annotations from spaCy model
    :param lst gold_spacy_data: Dataset with gold annotations 
    :return: The value of precision, reacall, and fscore
    """
    precision, recall, fscore = 0, 0, 0
    tp = 0
    fp = 0
    fn = 0
    for sys_ex, gold_ex in zip(sys_spacy_data, gold_spacy_data):
        if sys_ex[0] != gold_ex[0]:
            print(sys_ex,gold_ex)
        gold_annotations = set([tuple(e) for e in gold_ex[1]["entities"]])
        sys_annotations = set([tuple(e) for e in sys_ex[1]["entities"]])
        tp += len(sys_annotations.intersection(gold_annotations))
        fp += len(sys_annotations.difference(gold_annotations))
        fn += len(gold_annotations.difference(sys_annotations))
    if (tp != 0):
        recall = (tp/(tp+fn)) *100
        precision = (tp/(tp+fp)) *100
        fscore=2*recall*precision/(recall+precision)
    return precision, recall, fscore


def init_model(spacy_train_data, language):
    """Initialize a spaCy sequence labelling model and get optimizer.

    :param lst spacy_train_data: An annotated dataset in spaCy format
    :param str language: A language code (e.g. "en")
    :return: A model and a optimizer
    """
    model = spacy.blank(language)
    seed(0)
    np.random.seed(0)
    spacy.util.fix_random_seed(0)
    ner = model.create_pipe("ner")
    model.add_pipe(ner, last=True)
    for sent, annotations in spacy_train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    # Make sure we're only training the NER component of the pipeline
    pipe_exceptions = ["ner"]
    other_pipes = [pipe for pipe in model.pipe_names if pipe not in pipe_exceptions]
    model.disable_pipes(*other_pipes)
    optimizer = model.begin_training()
    return model, optimizer


def annotate(spacy_data, model):
    """Annotate a dataset using the spaCy sequence labelling model.

    :param lst spacy_data: A dataset in spaCy format
    :param model: A spaCy sequence labelling model
    :return: An annotated dataset in the same format
    """
    result = []
    for sentence, _ in spacy_data:
        result.append((sentence,{"entities":[]}))
        annotated = model(sentence)
        
        for entity in annotated.ents:
            result[-1][1]["entities"].append((entity.start_char, entity.end_char, entity.label_))
    return result


def train(spacy_train_data, spacy_dev_data, epochs,language):
    """Implement the training procedure for each epoch.

    :param lst spacy_train_data: An annotated training dataset in spaCy format
    :param lst spacy_dev_data: An annotated dev dataset in spaCy format
    :param int epochs: Number of epochs
    :param str language: A language code (e.g. "en")
    :return: The best model out of the epochs
    """
    # Initialize model and get optimizer
    model, optimizer = init_model(spacy_train_data,language)
    # Make sure original training data is not permuted
    spacy_train_data = deepcopy(spacy_train_data)
    best_f = 0
    best_model = None
    for itn in range(epochs):
        losses = {}
        shuffle(spacy_train_data)
        # Batch up the examples using spaCy's minibatch
        batches = list(minibatch(spacy_train_data, size=5))
        for i,batch in enumerate(batches):
            texts, annotations = zip(*batch)
            model.update(texts,  
                         annotations,  
                         losses=losses,
                         drop=0.1)
        # Evaluate the model
        print("Loss for epoch %u: %.4f" % (itn+1, losses["ner"]))
        spacy_dev_sys = annotate(spacy_dev_data, model)
        p, r, f = evaluate(spacy_dev_sys,spacy_dev_data)
        print("  PRECISION: %.2f%%, RECALL: %.2f%%, F-SCORE: %.2f%%" % (p,r,f))
        if f > best_f:
            best_f = f
            best_model = deepcopy(model)
    return best_model


# ========== Prepare spaCy model output data for validation.py==========
def convert_spacy_sent_output(tup_for_sent,context_index):
    """Reformat a sentence spacy output for validation, update the indices as in an article level instead of a sentence level.

    :param tuple tup_for_sent: A tuple contatining a sentence and a dictionary with key = "entities", value = (start_index,end_index,entity_name) for the sentence
    :param int context_index: Current index pointer in the article
    :return: tup_for_sent with updated indcies in an article level instead of a sentence level
    """
    sentence = tup_for_sent[0]
    ent_dic = tup_for_sent[1]
    seen_ent = set()
    new_sent_dic = {"speaker":"",
                       "verb":"",
                       "quote":"",
                       "speaker_index":"",
                       "verb_index":"",
                       "quote_index":""}
    reformatted_output = [new_sent_dic]
    prev_ent = ''
    for ent_tup in ent_dic['entities']:
        curr_ent = ent_tup[-1]
        if prev_ent != curr_ent:
            if ent_tup[-1] not in seen_ent:
                start = ent_tup[0]
                end = ent_tup[1]
                seen_ent.add(ent_tup[-1])
                reformatted_output[-1][convert_ent_name(ent_tup[-1])+'_index']=(start,end)
            else:
                start = ent_tup[0]
                end = ent_tup[1]
                seen_ent = set()
                seen_ent.add(ent_tup[-1])
                reformatted_output.append({"speaker":"",
                       "verb":"",
                       "quote":"",
                       "speaker_index":"",
                       "verb_index":"",
                       "quote_index":""})
                reformatted_output[-1][convert_ent_name(ent_tup[-1])+'_index']=(start,end)
        else:
            new_end =  ent_tup[1]
            reformatted_output[-1][convert_ent_name(ent_tup[-1])+'_index']=(start,new_end)
        prev_ent = curr_ent
    # Add the content of the quote, speaker, verb using the index pair
    for dic_idx in range(len(reformatted_output)):
        keys = deepcopy([key for key in reformatted_output[dic_idx].keys() if 'index' in key and len(reformatted_output[dic_idx][key])>0])
        for key in keys:
            indicies = reformatted_output[dic_idx][key]
            reformatted_output[dic_idx][key.split("_")[0]] = sentence[indicies[0]:indicies[1]]
    # Update indices to article level instead of sentence-level
    for dic_idx in range(len(reformatted_output)):
        index_keys = deepcopy([key for key in reformatted_output[dic_idx].keys() if 'index' in key and len(reformatted_output[dic_idx][key])>0])
        for key in index_keys:
            reformatted_output[dic_idx][key] = "({0},{1})".format(reformatted_output[dic_idx][key][0] + context_index,reformatted_output[dic_idx][key][1] + context_index)
    return reformatted_output


def get_filename(directory):
    """To get all the filenames in the given folder.

    :param str directory: Folder path 
    :return: A list of filenames in the given folder
    """
    # Target files' name should have a length 24
    filename_len = 24
    filename_lst = []
    for file in os.listdir(directory):
        filename = file.split('.')[0]
        if len(filename) == filename_len:
            filename_lst.append(filename)
    return filename_lst


def convert_ent_name(ent_name):
    """Convert the entity names used in PARC to our data, specifically, from 'CONTENT', 'CUE', 'SOURCE' to 'quote', 'verb' and 'speaker' correspondingly.

    :param str ent_name: Entity name used in PARC
    :return: Entity name used in our data
    """
    if ent_name == 'CONTENT':
        return 'quote'
    elif ent_name == 'CUE':
        return 'verb'
    elif ent_name == 'SOURCE':
        return 'speaker'


def main(rawtext_dir):
    """Annotate each article file in the given directory and save the annotated data.

    :param str rawtext_dir: Path to raw news articles
    """
    gap = len("\n .\n ")
    for filename in FILENAMES: 
        prep_raw_text = preprocess_text(open(rawtext_dir +'/' + filename + ".txt").read())
        spacy_input_sent = []
        first_quote = False
        for sent in nltk.sent_tokenize(prep_raw_text):
            # Check previous sentence
            if len(spacy_input_sent) >0:
                prev_sent = spacy_input_sent[-1][0]
            else:
                prev_sent = ' '
            # Rejoin the quotation parts if a quotation with multiple sentences is split previously using nltk.sent_tokenize
            if sent.count('"') %2 != 0 and first_quote == False:
                spacy_input_sent.append(tuple((sent,{'entities':[[]]})))
                first_quote = True
            elif sent.count('"') %2 == 0 and first_quote == True:
                extended_sent = spacy_input_sent[-1][0]+ ' ' + sent
                spacy_input_sent[-1] = tuple((extended_sent,{'entities':[[]]}))
            elif sent.count('"') %2 != 0 and first_quote == True:
                extended_sent = spacy_input_sent[-1][0]+ ' ' + sent
                spacy_input_sent[-1] = tuple((extended_sent,{'entities':[[]]}))
                first_quote = False
            else:
                if prev_sent[-1] == '.' and prev_sent != '.' and prev_sent[-2].isupper():
                    extended_sent = prev_sent+ ' ' + sent
                    spacy_input_sent[-1] = tuple((extended_sent,{'entities':[[]]}))
                else:
                    spacy_input_sent.append(tuple((sent,{'entities':[[]]})))
        # Annotate our data using the spaCy model
        output_spacy = annotate(spacy_input_sent, NER_model)
        pred_output_final = []
        context_index = 0
        prev_is_para_break = False
        # Keep tracking the current index (context_index) in an article
        for sent_tup in output_spacy:
            if context_index != 0:
                if sent_tup[0] == '.' and prev_is_para_break == False:
                    context_index = context_index + gap -1
                    prev_is_para_break = True
                elif sent_tup[0] == '.' and prev_is_para_break == True:
                    prev_is_para_break = False
                elif prev_is_para_break == True:
                    # do not need to add one extra space as the previous line is a paragraph break
                    prev_is_para_break = False
                else:
                    context_index += 1 
            # Exclude the sentences without entities and filter out the sent_tup if the length of predicted quotes <= 4 words
            if len(sent_tup[1]['entities']) >0 and sum([tup.count('CONTENT') for tup in sent_tup[1]['entities']])>4:
                for dic in convert_spacy_sent_output(sent_tup, context_index):
                    if len(dic['quote'].split(' ')) > 3: # exclude quote if it contains four or less words                  
                        pred_output_final.append(dic)
            context_index = context_index + len(sent_tup[0]) 
        # Save annotated data
        with open(os.path.join(OUTPUT_DIRECTORY, filename + ".json"), 'w') as f:
            json.dump(pred_output_final, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="news_genre_file_num.txt", help="Domain of articles to use")
    parser.add_argument("--parc_dataset", type=str, default="/Users/anniean/Documents/Capstone/COLX595_Informed_Opinions/data/PARC3_complete", help="Path to PARC article data")
    parser.add_argument("--raw_input", type=str, default="../data/rawtext", help="Path to raw news article data")
    parser.add_argument("--epoch", type=int, default=1, help="Number of training procedures implemented")
    parser.add_argument("--output", type=str, default="../validation/V4.0", help="Path to raw news article data")
    args = parser.parse_args()
    
    RAWTEXT_DIR = args.raw_input
    EPOCH = args.epoch
    news = open(args.domain)
    news_files = news.read().split(', ')
    train_data = get_spacy_formatted_data(args.parc_dataset + "/train/", news_files)
    dev_data = get_spacy_formatted_data(args.parc_dataset + "/dev/", news_files)
    test_data = get_spacy_formatted_data(args.parc_dataset + "/test/", news_files) 
    FILENAMES = sorted(get_filename(RAWTEXT_DIR))
    OUTPUT_DIRECTORY = args.output
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    # Train the spaCy sequence labelling model
    NER_model = train(train_data, dev_data, EPOCH, "en")
    print()
    print("Evaluating model on development set:")
    en_spacy_dev_sys = annotate(dev_data, NER_model)
    p, r, f = evaluate(en_spacy_dev_sys,dev_data)
    print("  PRECISION: %.2f%%, RECALL: %.2f%%, F-SCORE: %.2f%%" % (p,r,f))

    main(RAWTEXT_DIR)
