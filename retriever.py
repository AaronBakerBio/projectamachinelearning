import numpy as np
import pandas as pd
import os
import re
import nltk
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB



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


def calc_root_mean_squared_error(y_N, yhat_N):
    ''' Compute root mean squared error given true and predicted values

    Args
    ----
    y_N : 1D array, shape (N,)
        Each entry represents 'ground truth' numeric response for an example
    yhat_N : 1D array, shape (N,)
        Each entry represents predicted numeric response for an example

    Returns
    -------
    rmse : scalar float
        Root mean squared error performance metric
        .. math:
            rmse(y,\hat{y}) = \sqrt{\frac{1}{N} \sum_{n=1}^N (y_n - \hat{y}_n)^2}
    '''
    y_N = np.atleast_1d(y_N)
    yhat_N = np.atleast_1d(yhat_N)
    assert y_N.ndim == 1
    assert y_N.shape == yhat_N.shape
    return np.sqrt(np.mean(((yhat_N-y_N)**2)))



def print_results(model_name, values, n_folds, start, end):
    print(f"Model name: {model_name}")
    print(f"number folds: {n_folds}")
    print(f"validation time: {end - start:.9f} seconds")
    print(f"rmse for train: {values[0]:.4f}")
    print(f"rmse for val: {values[1]:.4f}")
    print(f"accuracy for train: {values[2]:.4f}")
    print(f"accuracy for val: {values[3]:.4f}")
    if len(values) > 4:  
        print(f"AUROC for val: {values[4]:.4f}\n")
    else:
        print("AUROC not available for this model.\n")

def print_results2(model_name, values, n_folds):
    print(f"Model name: {model_name}")
    print(f"number folds: {n_folds}")
    print(f"rmse for train: {values[0]:.4f}")
    print(f"rmse for val: {values[1]:.4f}")
    print(f"accuracy for train: {values[2]:.4f}")
    print(f"accuracy for val: {values[3]:.4f}")
    if len(values) > 4:  
        print(f"AUROC for val: {values[4]:.4f}\n")
    else:
        print("AUROC not available for this model.\n")

def train_models_and_calc_scores_with_bag_of_words(estimator, x_reviews, y_N, n_folds=3, random_state=0):
    """
    Train models and calculate scores using Bag-of-Words, with vectorization done within each cross-validation fold.
    """
    train_rmse_per_fold = np.zeros(n_folds, dtype=np.float32)
    validation_rmse_per_fold = np.zeros(n_folds, dtype=np.float32)
    train_accuracy_per_fold = np.zeros(n_folds, dtype=np.float32)
    validation_accuracy_per_fold = np.zeros(n_folds, dtype=np.float32)
    validation_auroc_per_fold = np.zeros(n_folds, dtype=np.float32)

    # Get train and validation indices for n-fold cross-validation
    train_ids_per_fold, validation_ids_per_fold = make_train_and_validation_row_ids_for_n_fold_cv(len(x_reviews), n_folds, random_state)
    
    auroc_available = hasattr(estimator, "predict_proba")  # Check if the model supports AUROC calculation

    for f in range(n_folds):
        training_ind = train_ids_per_fold[f]
        validation_ind = validation_ids_per_fold[f]
        
        # Split raw text into training and validation sets
        X_train_raw = np.array(x_reviews)[training_ind]
        X_val_raw = np.array(x_reviews)[validation_ind]
        y_train = y_N[training_ind]
        y_val = y_N[validation_ind]

        # Vectorize using CountVectorizer (fit only on training data)
        vectorizer = CountVectorizer(lowercase=True)
        X_train = vectorizer.fit_transform(X_train_raw)
        X_val = vectorizer.transform(X_val_raw)

        # Train the estimator on vectorized training data
        estimator.fit(X_train, y_train)
        
        # Generate predictions for training and validation sets
        training_pred = estimator.predict(X_train)
        validation_pred = estimator.predict(X_val)

        # Calculate RMSE and accuracy for training and validation sets
        train_rmse_per_fold[f] = calc_root_mean_squared_error(y_train, training_pred)
        train_accuracy_per_fold[f] = accuracy_score(y_train, training_pred)
        validation_rmse_per_fold[f] = calc_root_mean_squared_error(y_val, validation_pred)
        validation_accuracy_per_fold[f] = accuracy_score(y_val, validation_pred)

        # Calculate AUROC if available
        if auroc_available:
            validation_proba = estimator.predict_proba(X_val)[:, 1]
            validation_auroc_per_fold[f] = roc_auc_score(y_val, validation_proba)

    # Return average scores across all folds
    if auroc_available:
        return (np.mean(train_rmse_per_fold), np.mean(validation_rmse_per_fold), 
                np.mean(train_accuracy_per_fold), np.mean(validation_accuracy_per_fold),
                np.mean(validation_auroc_per_fold))
    else:
        return (np.mean(train_rmse_per_fold), np.mean(validation_rmse_per_fold), 
                np.mean(train_accuracy_per_fold), np.mean(validation_accuracy_per_fold))

def train_models_and_calc_scores_with_rmse_and_accuracy(estimator, x_reviews, y_N, n_folds=3, random_state=0):
    """
    Train models and calculate RMSE and accuracy with vectorization done inside each fold.
    """
    train_rmse_per_fold = np.zeros(n_folds, dtype=np.float32)
    validation_rmse_per_fold = np.zeros(n_folds, dtype=np.float32)
    train_accuracy_per_fold = np.zeros(n_folds, dtype=np.float32)
    validation_accuracy_per_fold = np.zeros(n_folds, dtype=np.float32)
    validation_auroc_per_fold = np.zeros(n_folds, dtype=np.float32)

    train_ids_per_fold, validation_ids_per_fold = make_train_and_validation_row_ids_for_n_fold_cv(
        len(x_reviews), n_folds, random_state
    )

    auroc_available = hasattr(estimator, "predict_proba")  # Check if the model supports AUROC calculation

    for f in range(n_folds):
        training_ind = train_ids_per_fold[f]
        validation_ind = validation_ids_per_fold[f]
        
        # Split raw text into training and validation sets
        X_train_raw = np.array(x_reviews)[training_ind]
        X_val_raw = np.array(x_reviews)[validation_ind]
        y_train = y_N[training_ind]
        y_val = y_N[validation_ind]

        # Vectorize using CountVectorizer (fit only on training data)
        vectorizer = CountVectorizer(lowercase=True)
        X_train = vectorizer.fit_transform(X_train_raw)
        X_val = vectorizer.transform(X_val_raw)

        # Train the estimator on vectorized training data
        estimator.fit(X_train, y_train)
        
        # Generate predictions for training and validation sets
        training_pred = estimator.predict(X_train)
        validation_pred = estimator.predict(X_val)

        # Calculate RMSE and accuracy for training and validation sets
        train_rmse_per_fold[f] = calc_root_mean_squared_error(y_train, training_pred)
        train_accuracy_per_fold[f] = accuracy_score(y_train, training_pred)
        validation_rmse_per_fold[f] = calc_root_mean_squared_error(y_val, validation_pred)
        validation_accuracy_per_fold[f] = accuracy_score(y_val, validation_pred)

        # Calculate AUROC if available
        if auroc_available:
            validation_proba = estimator.predict_proba(X_val)[:, 1]
            validation_auroc_per_fold[f] = roc_auc_score(y_val, validation_proba)

    # Return average scores across all folds
    return (np.mean(train_rmse_per_fold), np.mean(validation_rmse_per_fold), 
            np.mean(train_accuracy_per_fold), np.mean(validation_accuracy_per_fold),
            np.mean(validation_auroc_per_fold) if auroc_available else np.nan)



def train_with_ngram_and_tfidf_thresholding(estimator, x_reviews, y_N, ngram_ranges, use_tfidf=False, tfidf_percentile=90, n_folds=3, random_state=0):
    best_auroc = -1  # Initialize best AUROC
    best_values = None  # To store the best results
    best_ngram = None  # To store the best n-gram range
    
    # Get train and validation indices for n-fold cross-validation
    train_ids_per_fold, validation_ids_per_fold = make_train_and_validation_row_ids_for_n_fold_cv(len(x_reviews), n_folds, random_state)

    # Iterate over the n-gram ranges to test
    for ngram_range in ngram_ranges:
        print(f"\nTesting n-grams: {ngram_range}")

        validation_auroc_per_fold = np.zeros(n_folds, dtype=np.float32)

        # Cross-validation loop
        for f in range(n_folds):
            training_ind = train_ids_per_fold[f]
            validation_ind = validation_ids_per_fold[f]
            
            # Split into training and validation sets
            X_train = np.array(x_reviews)[training_ind]
            X_val = np.array(x_reviews)[validation_ind]
            y_train = y_N[training_ind]
            y_val = y_N[validation_ind]

            # Create a CountVectorizer with the specified n-gram range and fit it on the training data only
            vectorizer = CountVectorizer(ngram_range=ngram_range)
            X_train_counts = vectorizer.fit_transform(X_train)

            # Get the feature names (n-grams)
            feature_names = vectorizer.get_feature_names_out()

            # Optionally filter n-grams based on stop words
            words_to_keep = [word for word in feature_names if not (word.split()[0] in stoppers or word.split()[-1] in stoppers)]

            # Optionally filter by TF-IDF percentile if the flag is set
            if use_tfidf:
                # Transform count matrix to TF-IDF
                tfidf_transformer = TfidfTransformer()
                X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

                # Calculate the average TF-IDF score for each term
                avg_tfidf_scores = np.array(X_train_tfidf.mean(axis=0)).flatten()

                # Determine the threshold based on the percentile
                threshold_value = np.percentile(avg_tfidf_scores, tfidf_percentile)

                # Keep only terms that meet the TF-IDF threshold
                words_to_keep = [word for i, word in enumerate(feature_names) if avg_tfidf_scores[i] > threshold_value]
            
            # Create a new CountVectorizer with the filtered vocabulary
            filtered_vectorizer = CountVectorizer(vocabulary=words_to_keep, ngram_range=ngram_range)
            
            # Transform the training and validation sets using the filtered vocabulary
            X_train_filtered = filtered_vectorizer.transform(X_train)
            X_val_filtered = filtered_vectorizer.transform(X_val)

            # Convert to dense matrices if needed for some models
            X_train_filtered_dense = X_train_filtered.toarray()
            X_val_filtered_dense = X_val_filtered.toarray()

            # Train the model and calculate scores
            estimator.fit(X_train_filtered_dense, y_train)
            y_train_pred = estimator.predict(X_train_filtered_dense)
            y_val_pred = estimator.predict(X_val_filtered_dense)

            # Calculate RMSE and accuracy for training and validation sets
            train_rmse = calc_root_mean_squared_error(y_train, y_train_pred)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_rmse = calc_root_mean_squared_error(y_val, y_val_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred)

            # Calculate AUROC if the estimator supports probability prediction
            if hasattr(estimator, "predict_proba"):
                y_val_proba = estimator.predict_proba(X_val_filtered_dense)[:, 1]
                val_auroc = roc_auc_score(y_val, y_val_proba)
                validation_auroc_per_fold[f] = val_auroc

            # If no predict_proba, default AUROC to 0
            else:
                validation_auroc_per_fold[f] = 0

        # Calculate average AUROC for this n-gram range
        avg_auroc = np.mean(validation_auroc_per_fold)

        print(f"AUROC for n-gram range {ngram_range}: {avg_auroc:.4f}")

        # Check if this is the best AUROC
        if avg_auroc > best_auroc:
            best_auroc = avg_auroc
            best_values = (train_rmse, val_rmse, train_accuracy, val_accuracy, avg_auroc)
            best_ngram = ngram_range

    # Return the best results and n-gram range
    return best_ngram, best_values, best_auroc

def perform_error_analysis_with_one_fold(x_array, y_array, vectorizer_params, alpha=0.8):
    """
    Perform manual split into training and validation, train a CNB model, and evaluate errors.
    
    Args:
    x_array: Array of text data
    y_array: Corresponding labels
    vectorizer_params: Parameters for CountVectorizer (e.g., ngram_range)
    alpha: Regularization parameter for Complement Naive Bayes.
    
    Returns:
    - Confusion matrix and the false positives and false negatives.
    """
    # Manually split the dataset into 2/3 training and 1/3 validation
    X_train, X_val, y_train, y_val = train_test_split(x_array, y_array, test_size=0.33, random_state=43)

    # Initialize the CountVectorizer with given parameters (e.g., ngram_range)
    vectorizer = CountVectorizer(**vectorizer_params)
    
    # Fit on the training data
    X_train_vect = vectorizer.fit_transform(X_train)
    X_val_vect = vectorizer.transform(X_val)

    # Initialize Complement Naive Bayes with the given alpha
    cnb_model = ComplementNB(alpha=alpha)

    # Train the model
    cnb_model.fit(X_train_vect, y_train)

    # Predict on the validation set
    y_val_pred = cnb_model.predict(X_val_vect)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Extract false positives and false negatives
    false_positives = np.where((y_val == 0) & (y_val_pred == 1))[0]
    false_negatives = np.where((y_val == 1) & (y_val_pred == 0))[0]

    # Convert X_val to a pandas Series to align indexing properly
    X_val_series = pd.Series(X_val)

    # Ensure false_positives and false_negatives indices are valid
    valid_false_positives = [idx for idx in false_positives if idx < len(X_val_series)]
    valid_false_negatives = [idx for idx in false_negatives if idx < len(X_val_series)]

    # Display a few false positives and false negatives



    print("Confusion Matrix:")
    print(pd.DataFrame(conf_matrix, 
                       columns=["Predicted Negative", "Predicted Positive"], 
                       index=["True Negative", "True Positive"]))
    
    # Print false positives with line wrapping at 40 characters
    print("\nFalse Positives:")
    for idx in valid_false_positives:
        print(f"Index {idx}:")
        text = X_val_series.iloc[idx]
        wrapped_text = '\n'.join([text[i:i+40] for i in range(0, len(text), 40)])
        print(f"{wrapped_text}\n{'-'*80}")
    
    # Print false negatives with line wrapping at 40 characters
    print("\nFalse Negatives:")
    for idx in valid_false_negatives:
        print(f"Index {idx}:")
        text = X_val_series.iloc[idx]
        wrapped_text = '\n'.join([text[i:i+40] for i in range(0, len(text), 40)])
        print(f"{wrapped_text}\n{'-'*80}")


    return conf_matrix, valid_false_positives, valid_false_negatives

