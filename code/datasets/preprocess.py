import pandas as pd
import os
from pathlib import Path
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
nltk.download('punkt_tab')
nltk.download('stopwords')

# cur_dir = Path(os.getcwd())
#
# train_dataframe = pd.read_csv(cur_dir.parent.parent.parent/'data/raw/train.csv')
# test_dataframe = pd.read_csv(cur_dir.parent.parent.parent/'data/raw/test.csv')

def preprocess_score_inplace(df):
    """
    Normalizes score to make it from 0 to 1.

    For now it is from 1.0 to 5.0, so natural choice
    is to normalize by (f - 1.0)/4.0
    """
    df['Score'] = (df['Score'] - 1.0) / 4.0
    return df


def preprocess_helpfulness_inplace(df):
    """
    Splits feature by '/' and normalize helpfulness to make it from 0 to 1

    The total number of assessments can be 0, so let's substitute it
    with 1. The resulting helpfulness still will be zero but we
    remove the possibility of division by zero exception.

    Return value should be float
    """
    helpf_df = df['Helpfulness'].str.split("/", expand=True).astype(int)
    helpf_df.columns = ["Helpful", "Total"]
    helpf_df["Total"] = helpf_df["Total"].replace(0, 1)

    df["Helpfulness"] = helpf_df["Helpful"] / helpf_df["Total"]

    return df


def concat_title_text_inplace(df):
    """
    Concatenates Title and Text columns together
    """
    df['Text'] = df['Title'] + " " + df['Text']
    df.drop('Title', axis=1, inplace=True)
    return df


# define categories indices
cat2idx = {
    'toys games': 0,
    'health personal care': 1,
    'beauty': 2,
    'baby products': 3,
    'pet supplies': 4,
    'grocery gourmet food': 5,
}
# define reverse mapping
idx2cat = {
    v:k for k,v in cat2idx.items()
}

def encode_categories(df):
    df['Category'] = df['Category'].apply(lambda x: cat2idx[x])
    return df

# train_copy = train_dataframe.head().copy()
#
# encode_categories(preprocess_score_inplace(preprocess_helpfulness_inplace(concat_title_text_inplace(train_copy))))


def lower_text(text: str):
    return text.lower()


def remove_numbers(text: str):
    text_nonum = re.sub(r'\d+', ' ', text)
    return text_nonum


def remove_punctuation(text: str):
    text_nopunct = re.sub(r'\W+', ' ', text)
    return text_nopunct


def remove_multiple_spaces(text: str):
    text_no_doublespace = re.sub(r'\s+', ' ', text)
    return text_no_doublespace

def tokenize_text(text: str) -> list[str]:
    return word_tokenize(text)

def remove_stop_words(tokenized_text: list[str]) -> list[str]:
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in tokenized_text if not w.lower() in stop_words]
    return filtered_sentence

def stem_words(tokenized_text: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokenized_text]

# sample_text = train_copy['Text'][4]
#
# _lowered = lower_text(sample_text)
# _without_numbers = remove_numbers(_lowered)
# _without_punct = remove_punctuation(_without_numbers)
# _single_spaced = remove_multiple_spaces(_without_punct)

# print(sample_text)
# print('-'*10)
# print(_lowered)
# print('-'*10)
# print(_without_numbers)
# print('-'*10)
# print(_without_punct)
# print('-'*10)
# print(_single_spaced)


def preprocessing_stage(text):
    _lowered = lower_text(text)
    _without_numbers = remove_numbers(_lowered)
    _without_punct = remove_punctuation(_without_numbers)
    _single_spaced = remove_multiple_spaces(_without_punct)
    _tokenized = tokenize_text(_single_spaced)
    _without_sw = remove_stop_words(_tokenized)
    _stemmed = stem_words(_without_sw)

    return _stemmed


def clean_text_inplace(df):
    df['Text'] = df['Text'].apply(preprocessing_stage)
    return df


def preprocess(df):
    df.fillna(" ", inplace=True)
    _preprocess_score = preprocess_score_inplace(df)
    _preprocess_helpfulness = preprocess_helpfulness_inplace(_preprocess_score)
    _concatted = concat_title_text_inplace(_preprocess_helpfulness)

    if 'Category' in df.columns:
        _encoded = encode_categories(_concatted)
        _cleaned = clean_text_inplace(_encoded)
    else:
        _cleaned = clean_text_inplace(_concatted)
    return _cleaned


# ratio = 0.2
# train, val = train_test_split(
#     train_dataframe, stratify=train_dataframe['Category'], test_size=0.2, random_state=420
# )
#
# train.to_csv(cur_dir.parent.parent.parent/'data/processed/train.csv', index=False)
# val.to_csv(cur_dir.parent.parent.parent/'data/processed/val.csv', index=False)
# test_dataframe.to_csv(cur_dir.parent.parent.parent/'data/processed/test.csv', index=False)