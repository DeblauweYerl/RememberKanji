from keytotext import pipeline
from transformers.file_utils import MULTIPLE_CHOICE_DUMMY_INPUTS

# loading the pre-trained model
nlp = pipeline("k2t-base")

# decoding params
params = {"do_sample":True, "num_beams":4, "no_repeat_ngram_size":3, "early_stopping":True}

# keyword sets
kw_easy = ['cut', 'seven', 'sword']
kw_medium = ['floor', 'building', 'compare', 'white']
kw_hard = ['wear', 'horns', 'king', 'slide', 'eye']

# sentence generation
sen_easy = nlp(kw_easy, **params)
sen_medium = nlp(kw_medium, **params)
sen_hard = nlp(kw_hard, **params)

# print results
print(f"Sentence easy: {sen_easy}")
print(f"Sentence medium: {sen_medium}")
print(f"Sentence hard: {sen_hard}")
