import json
from keytotext import pipeline
from transformers.file_utils import MULTIPLE_CHOICE_DUMMY_INPUTS


# loading the pre-trained model
nlp = pipeline("k2t-base")

# decoding params
params = {"do_sample":True, "num_beams":4, "no_repeat_ngram_size":3, "early_stopping":True}


# gets set of keywords from kanji character
def get_keywords(character):
    
    # check if character is present in dataset
    if character in kanji_data.keys():
        char_properties = kanji_data[character]
        
        # check if radicals are available
        if char_properties['wk_radicals'] != None:
            
            # add meaning as keyword
            keywords = [char_properties['wk_meanings'][0].lower()]
            
            # add radicals as keywords
            [keywords.append(rad.lower()) for rad in char_properties['wk_radicals']]
            print(f"Keywords: {keywords}")
            return keywords
        else:
            print("Radicals not available for this character. Try another one.")
            return []
    else:
        print("Character not available. Try another one.")
        return []

# import kanji data
with open('data/kanji.json', encoding='utf8') as f:
    kanji_data = json.load(f)


# input and get keywords
keywords = get_keywords(input("Insert a character: "))


# generate sentence
if len(keywords) > 2:
    final_result = nlp(keywords, **params)
    print(f"Sentence: {final_result}")
else:
    print("Try another character.")