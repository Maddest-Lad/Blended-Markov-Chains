import re
import time
from os.path import join

import markovify
import spacy


class POSifiedText(markovify.Text):

    def word_split(self, sentence):
        return ['::'.join((word.orth_, word.pos_)) for word in nlp(sentence)]

    def word_join(self, words):
        sentence = ' '.join(word.split('::')[0] for word in words)
        return sentence


def text_cleaner(text):
    text = re.sub(r'--', ' ', text)
    text = re.sub('[\[].*?[\]]', '', text)
    text = re.sub(r'(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b', '', text)
    text = re.sub(r"http\S+", "", text)
    text = ' '.join(text.split())
    return text


# Loads Data From Text Files into List
def load_data(path, datasets):
    texts = []

    for dataset in datasets:
        filepath = join(path, dataset)

        with open(filepath, encoding='utf-8') as file:
            # Cleaning Steps
            text = file.read()
            text = text_cleaner(text)
            if NLP:
                text = nlp(text)
                text = ' '.join([sent.text.lower().capitalize() for sent in text.sents if len(sent.text) > 1])
            texts.append(text)

    return texts


# From a List of Models, Derive Weights So That Each Model's Weight Is Equal Relative to It's Length
def gen_equal_weights(models):
    weights = []
    for model in models:
        weights.append(len(model))
    weights = [float(i) / sum(weights) for i in weights]
    return weights


# Combine the Models Together to Form the Full Chain
def construct_model(models, weights, chain_length=2):
    if NLP:
        markov_models = [POSifiedText(text, chain_length, well_formed=False) for text in models]
    else:
        markov_models = [markovify.Text(text, chain_length, well_formed=False) for text in models]
    return markovify.combine(markov_models, weights)


# Mildly Important Parameters
DATA_PATH = "data/processed"
DATASETS = ["ENTER FILENAMES HERE"]
NLP = False

if __name__ == '__main__':

    # Start Time Tracking
    start = time.time()

    # Load Natural Language
    if NLP:
        nlp = spacy.load('en_core_web_sm')

    # Load Text, Gen Weights And Build Model
    texts = load_data(DATA_PATH, DATASETS)
    weights = gen_equal_weights(texts)
    print("weights: ", weights)
    generator = construct_model(texts, weights)
    generator.compile(inplace=True)

    # End Time Tracking
    end = time.time()
    print("Time Elapsed: {} Seconds".format(int(end - start)))

    # Generate Text
    input("Press Enter To Generate a New Sentence")
    while True:
        sentence = generator.make_short_sentence(250)
        print(sentence)
        input("...")
