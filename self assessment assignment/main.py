# Naive Bayes Classifier with text preprocessing(conversion to lower case, stop word removal, punctuation removal) and Laplacian Smoothing. Trained model achieves Testing accuracy of 82.172% on the aclImdb_v1 dataset.
import numpy as np
import os
import re
import string

DATA_DIR = "aclImdb_v1" # Modify this to point to your dataset path

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "aclImdb/train")
TEST_DATA_PATH = os.path.join(DATA_DIR, "aclImdb/test")

# Preprocessing Training Data:
# Read in Training data samples
pos_train_data_path = os.path.join(TRAIN_DATA_PATH, 'pos')
neg_train_data_path = os.path.join(TRAIN_DATA_PATH, 'neg')

# Read in pos/neg comment file names
pos_train_file_names = os.listdir(pos_train_data_path)
neg_train_file_names = os.listdir(neg_train_data_path)

# Preprocessing helper function
def get_preprocessed_comment(file_path):
    """
    Given file path to the .txt file, preprocessed the text inside it and returns list of preprocessed words (split words, lower case, remove punctuation, remove stop words)
    Input:
        file_path (os.Path)
    Output:
        stipped (list)
    """
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    # Read comment in the file
    with open(file_path, 'rt') as f:
        comment = f.read() # whole file is read as string
        # split into words by white space
        exp = r'\W+'
        words = re.split(exp, comment)
        # remove punctuation, convert to lower case, remove stop words
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table).lower() for w in words if w not in stopwords] # type=list
        return stripped


# Create Word Freq map
pos_word_freq_map = {}
neg_word_freq_map = {}

# Iterate through pos samples
for cur_file_name in pos_train_file_names:
    cur_file_path = os.path.join(pos_train_data_path, cur_file_name)
    preprocessed_word_list = get_preprocessed_comment(cur_file_path)
    # Update frequency list
    for w in preprocessed_word_list:
        pos_word_freq_map[w] = 1 + pos_word_freq_map.get(w, 0)


for cur_file_name in neg_train_file_names:
    cur_file_path = os.path.join(neg_train_data_path, cur_file_name)
    preprocessed_word_list = get_preprocessed_comment(cur_file_path)
    # Update frequency list
    for w in preprocessed_word_list:
        neg_word_freq_map[w] = 1 + neg_word_freq_map.get(w, 0)


print("Number of words in positive frequence map: ", len(pos_word_freq_map))
print("Number of words in negative frequency map: ", len(neg_word_freq_map))
print("-"*20)

# -------------------- 
# Implement Naive Bayes Classifier Model:
class NaiveBayesClassifier():
    def __init__(self, pos_word_freq_map, neg_word_freq_map, num_positive_comments, num_negative_comments):
        self.pos_word_freq_map = pos_word_freq_map
        self.neg_word_freq_map = neg_word_freq_map
        self.q_pos = num_positive_comments / (num_positive_comments + num_negative_comments)
        self.q_neg = num_negative_comments / (num_positive_comments + num_negative_comments)

        assert self.q_pos + self.q_neg == 1.0, "self.q_pos + self.q_neg = 1 must hold for a valid probability distribution."

        self.pos_word_freqs_sum = sum(list(self.pos_word_freq_map.values()))
        self.neg_word_freqs_sum = sum(list(self.neg_word_freq_map.values()))
        # Laplacian smoothing (i.e some words will have zero probability if never before observed with a class)
        self.smoothing_factor = 1


    def get_pos_log_prob(self, preprocessed_word_list):
        """
        Given preprocessed comment as input, returns the log probability of the given comment belonging to positive category
        Inputs:
            preprocessed_word_list (list): list of preprocessed words that came from the same comment.
        Outputs:
            pog_log_prob (np.float64): log probability of comment belonging to positive category
        """
        pos_log_prob = np.log(np.float64(self.q_pos))
        for cur_word in preprocessed_word_list:
            # check if word exists in word_freq_map
            if cur_word in self.pos_word_freq_map:
                cur_word_freq = np.float64(self.pos_word_freq_map[cur_word]) + self.smoothing_factor
                log_q_x = np.log(cur_word_freq / (self.pos_word_freqs_sum + len(self.pos_word_freq_map)))
                pos_log_prob += log_q_x
            elif cur_word in self.neg_word_freq_map:
                cur_word_freq = self.smoothing_factor
                log_q_x = np.log(cur_word_freq / (self.pos_word_freqs_sum + len(self.pos_word_freq_map)))
                pos_log_prob += log_q_x


        return pos_log_prob


    def get_neg_log_prob(self, preprocessed_word_list):
        """
        Given preprocessed comment as input, returns the log probability of the given comment belonging to positive category
        Inputs:
            preprocessed_word_list (list): list of preprocessed words that came from the same comment.
        Outputs:
            pog_log_prob (np.float64): log probability of comment belonging to positive category
        """
        neg_log_prob = np.log(np.float64(self.q_neg))
        for cur_word in preprocessed_word_list:
            # check if word exists in word_freq_map
            if cur_word in self.neg_word_freq_map:
                cur_word_freq = np.float64(self.neg_word_freq_map[cur_word]) + self.smoothing_factor
                log_q_x = np.log(cur_word_freq / (self.neg_word_freqs_sum + len(self.neg_word_freq_map)))
                neg_log_prob += log_q_x
            elif cur_word in self.pos_word_freq_map:
                cur_word_freq = self.smoothing_factor
                log_q_x = np.log(cur_word_freq / (self.neg_word_freqs_sum + len(self.neg_word_freq_map)))
                neg_log_prob += log_q_x

        return neg_log_prob







model = NaiveBayesClassifier(pos_word_freq_map, neg_word_freq_map, len(pos_train_file_names), len(neg_train_file_names))


# -------------------- 
# Testing Model:

# Read in Test Data samples
pos_test_data_path = os.path.join(TEST_DATA_PATH, 'pos')
neg_test_data_path = os.path.join(TEST_DATA_PATH, 'neg')

# Read in pos/neg comment file names
pos_test_file_names = os.listdir(pos_test_data_path)
neg_test_file_names = os.listdir(neg_test_data_path)

# Testing the model on the Positive Test Samples:
true_pos_predictions = 0
false_neg_predictions = 0
# Testing loop:
for i, cur_file_name in enumerate(pos_test_file_names):
    cur_file_path = os.path.join(pos_test_data_path, cur_file_name)
    # preprocess the txt file
    preprocessed_word_list = get_preprocessed_comment(cur_file_path) # type=list
    
    # Get Log prob of pos class using model
    log_pos_prob = model.get_pos_log_prob(preprocessed_word_list)
    # Get Log prob of neg class using model
    log_neg_prob = model.get_neg_log_prob(preprocessed_word_list)

    assert log_pos_prob <= 0 and log_neg_prob <= 0, "log probabilities predicted should be smaller than or equal to log(1) = 0. Check your model implementation."

    if log_pos_prob > log_neg_prob:
        true_pos_predictions += 1
    else:
        false_neg_predictions += 1


print("true_pos_predictions: ", true_pos_predictions)
print("false_neg_predictions: ", false_neg_predictions)
print("Testing positive class accuracy of the model is: ", (100 * (true_pos_predictions / (false_neg_predictions + true_pos_predictions))), "%")
print("-"*20)

# Testing the model on the Negative Test Samples:
true_neg_predictions = 0
false_pos_predictions = 0
# Testing loop:
for i, cur_file_name in enumerate(neg_test_file_names):
    cur_file_path = os.path.join(neg_test_data_path, cur_file_name)
    # preprocess the txt file
    preprocessed_word_list = get_preprocessed_comment(cur_file_path) # type=list
    
    # Get Log prob of pos class using model
    log_pos_prob = model.get_pos_log_prob(preprocessed_word_list)
    # Get Log prob of neg class using model
    log_neg_prob = model.get_neg_log_prob(preprocessed_word_list)

    assert log_pos_prob <= 0 and log_neg_prob <= 0, "log probabilities predicted should be smaller than or equal to log(1) = 0. Check your model implementation."

    if log_pos_prob > log_neg_prob:
        false_pos_predictions += 1
    else:
        true_neg_predictions += 1


print("true_neg_predictions: ", true_neg_predictions)
print("false_pos_predictions: ", false_pos_predictions)
print("Testing negative class accuracy of the model is: ", (100 * (true_neg_predictions / (false_pos_predictions + true_neg_predictions))), "%")
print("-"*20)

# Printing Overall Metrics
print("Overall Testing Accuracy: ", (100 * (true_neg_predictions + true_pos_predictions) / (false_neg_predictions + false_pos_predictions + true_neg_predictions + true_pos_predictions)), "%")

