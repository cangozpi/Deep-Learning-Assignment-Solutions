# models.py

import numpy as np
import collections

import torch
from torch import nn

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context: corresponds to a single sentence (str)
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(ConsonantVowelClassifier, nn.Module):
    def __init__(self, Indexer, word_embedding_dim, gru_hidden_dim):
        super().__init__()
        self.Indexer = Indexer
        self.vocab_size = len(Indexer)
        self.word_embedding_dim = word_embedding_dim
        self.gru_hidden_dim = gru_hidden_dim

        self.embedding_layer = torch.nn.Embedding(self.vocab_size, word_embedding_dim)
        self.gru_model = torch.nn.GRU(self.word_embedding_dim, gru_hidden_dim, batch_first=True)
        self.classification_layer = torch.nn.Linear(self.gru_hidden_dim, 2) # note that output_dim = 2 since there are two possible classes (namely, vowel and consonant)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, context):
        """
        context: corresponds to a single sentence (str)
        return: classification probabilities of shape [B, 2]
        """
        # convert characters in context to indices (int)
        word_indices_input = [self.Indexer.index_of(c) for c in context]

        # convert sequences of mapped indices to embedding vectors
        embed_input = self.embedding_layer(torch.LongTensor(word_indices_input)) # --> [seq_len, word_embedding_dim]
        embed_input = torch.unsqueeze(embed_input, 0) # add batch dimension of 1 --> [B=1, seq_len, word_embedding_dim]

        # pass embeddings through GRU
        out, _ = self.gru_model(embed_input) # --> [B=1, seq_len, gru_hidden_dim]
        out = out[:, -1, :] # extract the output for the final pass # --> [B=1, word_embedding_dim]

        # pass RNN output through classification head
        out = torch.relu(out)
        out = self.classification_layer(out) # --> [B=1, 2] logits
        return out

    def predict(self, context):
        """
        context: corresponds to a single sentence (str)
        return: 1 if vowel, 0 if consonant
        """
        with torch.no_grad():
            out = self.forward(context) # --> [B=1, 2] predicted classification probabilities
            out = self.softmax(out) # --> [B=1, 2]
            out = torch.squeeze(out, dim=0) # --> [2]

            # return prediction (0 for consonant, 1 for vowel)
            pred = torch.argmax(out) # --> [1]
            return pred


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def create_and_shuffle_dataset(vowel_exs, cons_exs):
    cons_labelled = [(cons_sentance, 0) for cons_sentance in cons_exs]
    vowels_labelled = [(vowels_sentance, 1) for vowels_sentance in vowel_exs]

    joined_dataset = cons_labelled + vowels_labelled 

    # shuffle dataset randomly
    shuffle_indices = np.random.choice(np.arange(len(joined_dataset)), len(joined_dataset))
    shuffled_dataset = list(map(joined_dataset.__getitem__, shuffle_indices))
    return shuffled_dataset


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


# Wrapper function that times how long the function execution took
def timer_wrapper(func_name):
    def func_wrapper(func):
        import time
        def wrapper(*args, **kwargs):
            start_time = time.time()
            return_val = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func_name} took {(end_time - start_time):.3f} seconds to execute.")
            return return_val

        return wrapper
    return func_wrapper
    


@timer_wrapper("train_rnn_classifier function")
def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    # set seed for reproducibility
    set_seed(42)

    # Hyperparameters ============================== 
    word_embedding_dim = 200
    gru_hidden_dim = 300
    lr = 5e-3
    epochs = 10
    model = RNNClassifier(vocab_index, word_embedding_dim, gru_hidden_dim)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    # ==============================================

    # Train the model
    print("========== Training RNN Model ===========")
    model.train()
    accuracy_hist = []
    for epoch in range(1, epochs+1):
        # Get shuffled and merged dataset
        shuffled_dataset = create_and_shuffle_dataset(train_vowel_exs, train_cons_exs)
        # metrics to track
        num_correct_preds = 0
        num_total_preds = len(shuffled_dataset)

        for sentance, label in shuffled_dataset:
            model.zero_grad()
            pred = model(sentance) # --> [B=1, 2] classification logits
            target = torch.zeros(pred.shape[0], dtype=torch.long) if label == 0 else torch.ones(pred.shape[0], dtype=torch.long)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            # record metrics
            if torch.argmax(torch.squeeze(pred, dim=0)) == label:
                num_correct_preds += 1
        
        # record epoch metrics
        accuracy_hist.append(num_correct_preds / num_total_preds)
        print(f"Epoch: {epoch}, accuracy: {accuracy_hist[-1]}")

    
    return model


#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel, nn.Module):
    def __init__(self, Indexer, word_embedding_dim, gru_hidden_dim):
        super().__init__()
        self.Indexer = Indexer
        self.vocab_size = len(Indexer)
        self.word_embedding_dim = word_embedding_dim
        self.gru_hidden_dim = gru_hidden_dim

        self.embedding_layer = torch.nn.Embedding(self.vocab_size, word_embedding_dim)
        self.gru_model = torch.nn.GRU(self.word_embedding_dim, gru_hidden_dim, batch_first=True)
        self.classification_layer = torch.nn.Linear(self.gru_hidden_dim, self.vocab_size) # note that output_dim = 27 since there are 27 characters in the vocabulary 
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, context):
        # convert characters in context to indices (int)
        word_indices_input = [self.Indexer.index_of(c) for c in context]

        # convert sequences of mapped indices to embedding vectors
        embed_input = self.embedding_layer(torch.LongTensor(word_indices_input)) # --> [seq_len, word_embedding_dim]
        embed_input = torch.unsqueeze(embed_input, 0) # add batch dimension of 1 --> [B=1, seq_len, word_embedding_dim]

        # pass embeddings through GRU
        out, _ = self.gru_model(embed_input) # --> [B=1, seq_len, gru_hidden_dim]

        # pass RNN output through classification head
        out = torch.relu(out) # --> [B=1, seq_len, gru_hidden_dim]
        out = self.classification_layer(out) # --> [B=1, seq_len, vocab_size=27] logits
        return out

    def get_next_char_log_probs(self, context):
        # Get model prediction over vocab
        with torch.no_grad():
            pred = self.forward(context) # --> [B=1, seq_len, vocab_size=27] logits
            pred = self.softmax(pred) # --> [B=1, seq_len, vocab_size=27 probs
            pred = pred[0, -1, :] # --> [vocab_size=27], i.e. prediction prob for the last char
        
        # Convert to log prob
        pred_log_prob = torch.log(pred) # --> [vocab_size=27] log probs
        return pred_log_prob.cpu().detach().numpy()


    def get_log_prob_sequence(self, next_chars, context):
        pred_prob = 0 # log prob of next_chars following the given context under lm
        for target_char in next_chars:
            # Get model prediction over vocab
            with torch.no_grad():
                pred = self.forward(context) # --> [B=1, seq_len, vocab_size=27] logits
                pred = self.softmax(pred) # --> [B=1, seq_len, vocab_size=27 probs
                pred = pred[0, -1, :] # --> [vocab_size=27], i.e. prediction prob for the last char
        
            # Find index of target char
            target_index = self.Indexer.index_of(target_char)

            # Get corresponding prediction prob
            pred = pred[target_index]

            # Convert to log prop
            pred_log_prob = torch.log(pred)

            # accumulate probs
            pred_prob += pred_log_prob

            # update context
            context += target_char
        
        return float(pred_prob.item())



def create_lm_dataset(data, chunk_size):
    # Create dataset
    dataset = [] # holds (sentance, label) pairs in ints indices
    for i in range(0, (len(data) - chunk_size), chunk_size):
        cur_sample = data[i : ( i + chunk_size)] # --> [chunk_size]
        cur_label = data[(i + 1) : ( i + chunk_size + 1)] # --> [chunk_size]
        dataset.append((cur_sample, cur_label))
    return dataset


@timer_wrapper("train_lm function")
def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    # set seed for reproducibility
    set_seed(42)

    # Hyperparameters ============================== 
    chunk_size = 20 # determines the sequence lengths of input samples used during training
    word_embedding_dim = 200
    gru_hidden_dim = 300
    lr = 5e-3
    epochs = 10
    model = RNNLanguageModel(vocab_index, word_embedding_dim, gru_hidden_dim)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    # ==============================================

    # Create dataset
    train_dataset = create_lm_dataset(train_text, chunk_size) 

    # Train the model
    print("========== Training RNN Model ===========")
    model.train()
    loss_hist = []
    perplexity_hist = []
    accuracy_hist = []
    for epoch in range(1, epochs+1):
        # Get shuffled and merged dataset
        # shuffled_dataset = create_and_shuffle_dataset(train_vowel_exs, train_cons_exs)
        # metrics to track
        it_loss = 0
        it_perplexity = 0
        num_correct_preds = 0
        num_total_preds = 0

        for sentance, target in train_dataset:
            model.zero_grad()
            pred = model(sentance) # --> [B=1, seq_len, vocab_size=27] logits
            pred = torch.squeeze(pred, dim=0) # --> [seq_len, vocab_size=27] logits
            # Convert target from characters to indices of type torch tensor
            target = [vocab_index.index_of(c) for c in target]
            target = torch.LongTensor(target)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            # Record metrics
            # calculate loss
            it_loss += loss.detach().cpu().item()
            # calculate perplexity
            with torch.no_grad():
                log_prob = model.get_log_prob_sequence(sentance, " ")
                perplexity = np.exp(-log_prob / len(sentance))
                it_perplexity += perplexity
            # calculate accuracy
            pred_indices = torch.argmax(pred, dim=-1) # --> [seq_len]
            num_correct_preds += sum(pred_indices == target)
            num_total_preds += len(pred_indices)

        
        # record epoch metrics
        loss_hist.append(it_loss / len(train_dataset))
        perplexity_hist.append(it_perplexity / len(train_dataset))
        accuracy_hist.append(num_correct_preds / num_total_preds)
        print(f"Epoch: {epoch}, loss: {loss_hist[-1]}, perplexity: {perplexity_hist[-1]}, accuracy: {accuracy_hist[-1]}")

    
    model.eval()
    return model
