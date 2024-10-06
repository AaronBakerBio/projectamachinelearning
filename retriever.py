import numpy as np
import pandas as pd
import os
import sklearn.feature_extraction.text as handler
import re



def get_combined_x(full_x_array):
    # push both together to interpret
    combined_array = np.array([f"{source} {text}" for source, text in full_x_array])
    return combined_array


def split_x_in_halves(full_x_array):
    # separate it in halves (one half we call site)
    site_array = [entry[0] for entry in full_x_array]
    text_array = [entry[1] for entry in full_x_array]
    return site_array, text_array


def return_x_and_y(data_dir: str):
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    print(x_train_df.columns)
    # I chose to arbitrarily destroy all the numbers, I firmly believe they hold no meaning, feel free to put em back in
    #by deleting this line
    x_train_df['text'] = x_train_df['text'].apply(lambda text: re.sub(r'\d+', '', text))
    return x_train_df, y_train_df

def get_word_counts(vectorizer, x_array):
    """Returns the list of words, and the occurrences by document"""
    counts = vectorizer.fit_transform(x_array)
    return vectorizer.get_feature_names_out(), counts


def main():
    #y_train_df is already as it should be, so no real need to touch it much except typecasting
    x_train_df, y_train_df = return_x_and_y('data_reviews')
    # full x array contains [ [reviewer, review] ] by entry
    full_x_array = x_train_df.values.tolist()
    sites, reviews = split_x_in_halves(full_x_array)
    #I set lowercase = true so that all words are cast down to lowercase when analyzed.
    vectorizer = handler.CountVectorizer(lowercase=True)
    #words is an array of words (no numbers at all) and wordcounts is a count of each word by entry
    words, wordcounts = get_word_counts(vectorizer, reviews)

    #make an instance of the tfidf object to handle calculating our tfidf
    tfid_handler = handler.TfidfTransformer()
    tfid_values = tfid_handler.fit_transform(wordcounts)
    print(tfid_values.shape)



if __name__ == '__main__':
    main()
