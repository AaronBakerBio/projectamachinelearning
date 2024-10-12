import numpy as np
import pandas as pd
import os
import sklearn.feature_extraction.text as handler
import re
import nltk

stoppers = nltk.corpus.stopwords.words('english')
replacement_dict = {word: "good" for word in nltk.corpus.opinion_lexicon.positive()}
replacement_dict.update({word: "bad" for word in nltk.corpus.opinion_lexicon.negative()})

def get_combined_x(full_x_array):
    # push both together to interpret
    combined_array = np.array([f"{source} {text}" for source, text in full_x_array])
    return combined_array


def split_x_in_halves(full_x_array):
    """Split the input into an array of sites and reviews """
    # separate it in halves (one half we call site)
    site_array = [entry[0] for entry in full_x_array]
    text_array = [entry[1] for entry in full_x_array]
    return site_array, text_array

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stoppers]
    return ' '.join([word for word in text.split() if word.lower() not in stoppers])

def replace_simple_pos_neg(text):
    words = text.split()  # No need for lowercasing
    return ' '.join([replacement_dict.get(word, word) for word in words])

def return_x_and_y(data_dir: str):
    """ Gets training data from the directory and returns the data in numpy form"""
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    # print(x_train_df.columns)
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
    vectorizer = handler.CountVectorizer(lowercase=True, stop_words='english')
    #words is an array of words (no numbers at all) and wordcounts is a count of each word by entry
    words, wordcounts = get_word_counts(vectorizer, reviews)

    #make an instance of the tfidf object to handle calculating our tfidf
    tfid_handler = handler.TfidfTransformer()
    tfid_values = tfid_handler.fit_transform(wordcounts)
    print(tfid_values.shape)

def train_models_and_calc_scores_for_n_fold_cv(
        estimator, x_NF, y_N, n_folds=3, random_state=0):
    ''' 
    Stolen from hw1. Will use to train on 3 folds.
    '''
    train_error_per_fold = np.zeros(n_folds, dtype=np.float32)
    validation_error_per_fold = np.zeros(n_folds, dtype=np.float32)  # Changed test_error_per_fold to validation_error_per_fold
    train_ids_per_fold, validation_ids_per_fold = make_train_and_validation_row_ids_for_n_fold_cv(x_NF.shape[0], n_folds, random_state)  # Changed test_ids_per_fold to validation_ids_per_fold
    features = x_NF.shape[1]
    for f in range(n_folds):
        training_ind = train_ids_per_fold[f]
        validation_ind = validation_ids_per_fold[f]  # Changed test_ind to validation_ind
        trainingx = np.empty(shape=(len(training_ind), features))
        trainingy = np.empty(len(training_ind))
        validationx = np.empty(shape=(len(validation_ind), features))  # Changed testx to validationx
        validationy = np.empty(len(validation_ind))  # Changed testy to validationy
        for x in range(0, len(training_ind)):
            trainingx[x] = x_NF[training_ind[x]]
            trainingy[x] = y_N[training_ind[x]]
        for y in range(0, len(validation_ind)):
            validationx[y] = x_NF[validation_ind[y]]
            validationy[y] = y_N[validation_ind[y]]
        estimator.fit(trainingx, trainingy)
        training_pred = estimator.predict(trainingx)
        train_error_per_fold[f] = calc_root_mean_squared_error(trainingy, training_pred)
        validation_pred = estimator.predict(validationx)  # Changed test_pred to validation_pred
        validation_error_per_fold[f] = calc_root_mean_squared_error(validationy, validation_pred)  # Changed testy to validationy, and test_pred to validation_pred
    return train_error_per_fold, validation_error_per_fold  # Changed test_error_per_fold to validation_error_per_fold


def make_train_and_validation_row_ids_for_n_fold_cv(
        n_examples=0, n_folds=3, random_state=0):
    ''' 
    
    '''
    if hasattr(random_state, 'rand'):
        random_state = random_state
    else:
        random_state = np.random.RandomState(int(random_state))
    train_ids_per_fold = list()
    validation_ids_per_fold = list()  # Changed test_ids_per_fold to validation_ids_per_fold
    fold_sizes = np.full(n_folds, n_examples // n_folds, dtype=int)
    fold_sizes[:n_examples % n_folds] += 1  # remainder distribution
    folds = []
    for x in fold_sizes:
        folds.append([])
    to_append = np.arange(n_examples)
    for x in range(len(folds)):
        for y in range(fold_sizes[x]):
            index = random_state.choice(len(to_append))
            number = to_append[index]
            folds[x].append(number)
            to_append = np.delete(to_append, index)
    for f in range(n_folds):
        validation_ids = np.array(folds[f])  # Changed test_ids to validation_ids
        validation_ids_per_fold.append(validation_ids)  # Changed test_ids_per_fold to validation_ids_per_fold
        train_ids = np.hstack([folds[i] for i in range(n_folds) if i != f])
        train_ids_per_fold.append(train_ids)
    return train_ids_per_fold, validation_ids_per_fold  # Changed test_ids_per_fold to validation_ids_per_fold
