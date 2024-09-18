import random
import string
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import List, Tuple, Dict
from itertools import compress
from collections import Counter, deque
from sklearn.base import BaseEstimator, TransformerMixin
from datasets import load_dataset, load_from_disk


def prepareData(words: List[str], vocab_size: int = 50000):
    """
    Prepares the data for word vectorization by converting words to indices and creating dictionaries for word-to-index and index-to-word mappings.

    Parameters:
    words:      The corpus of words to be processed.
    vocab_size: The maximum size of the vocabulary. Default is 50,000.

    Returns:
    Tuple[List[int], List[Tuple[str, int]], dict, dict]:
        - data:        The corpus converted to a list of word indices.
        - count:       A list of tuples where each tuple contains a word and its frequency, including the <unk> token for rare words.
        - dictionary:  A dictionary mapping words to their corresponding indices.
        - reverse_dictionary:  A dictionary mapping indices to their corresponding words.
    """
    ## Corpus pre-processing
    translator = str.maketrans('', '', string.punctuation)
    words = [word.lower().translate(translator) for word in words]
    words = [word for word in words if word.isalpha()]
    
    ## Get word counts for vocabulary with <unk> token to replace rare words
    count = [['<unk>', -1]]
    count.extend(Counter(words).most_common(vocab_size - 1))

    ## Build dictionaries for index to word mapping
    dictionary = {word: idx for idx, (word, _) in enumerate(count)}
    reverse_dictionary = {idx: word for word, idx in dictionary.items()}
    
    ## Convert corpus to list of indices
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    
    return data, count, dictionary, reverse_dictionary



def preparePickleData(pickle_file_path: str, vocab_size: int = 50000) -> Tuple[np.ndarray, List[Tuple[str, int]], Dict[str, int], Dict[int, str]]:
    """
    Prepares the data for word vectorization by loading directly from a pickle file, converting words to indices, and creating dictionaries for word-to-index and index-to-word mappings.

    Parameters:
    pickle_file_path: The path to the pickle file containing the corpus of words to be processed.
    vocab_size:       The maximum size of the vocabulary. Default is 50,000.

    Returns:
    Tuple[np.ndarray, List[Tuple[str, int]], dict, dict]:
        - data:        The corpus converted to a NumPy array of word indices.
        - count:       A list of tuples where each tuple contains a word and its frequency, including the <unk> token for rare words.
        - dictionary:  A dictionary mapping words to their corresponding indices.
        - reverse_dictionary:  A dictionary mapping indices to their corresponding words.
    """
    
    ## First pass: Preprocess data and count unique words
    word_counts = Counter()
    translator = str.maketrans('', '', string.punctuation)
    with open(pickle_file_path, "rb") as file:
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        with tqdm(total=file_size, desc="Counting words") as pbar:
            while True:
                try:
                    words = pickle.load(file)
                    words = [word.lower().translate(translator) for word in words]
                    words = [word for word in words if word.isalpha()]
                    word_counts.update(words)
                    pbar.update(file.tell() - pbar.n)
                except EOFError:
                    break

    ## Get word counts for vocabulary with <unk> token to replace rare words
    count = [['<unk>', -1]]
    count.extend(word_counts.most_common(vocab_size - 1))

    ## Build dictionaries for index to word mapping
    dictionary = {word: idx for idx, (word, _) in enumerate(count)}
    reverse_dictionary = {idx: word for word, idx in dictionary.items()}
    
    ## Second pass: Convert words to indices
    index = 0
    unk_count = 0
    total_words = sum(word_counts.values())
    data = np.zeros(total_words, dtype=np.int32)
    with open(pickle_file_path, "rb") as file:
        with tqdm(total=file_size, desc="Converting words to indices") as pbar:
            while True:
                try:
                    words = pickle.load(file)
                    words = [word.lower().translate(translator) for word in words]
                    words = [word for word in words if word.isalpha()]
                    for word in words:
                        if word in dictionary:
                            data[index] = dictionary[word]
                        else:
                            data[index] = 0
                            unk_count += 1
                        index += 1
                    pbar.update(file.tell() - pbar.n)
                except EOFError:
                    break
    count[0][1] = unk_count
    
    return data, count, dictionary, reverse_dictionary



def cbow(data: List[int], batch_size: int, num_skips: int, skip_window: int, data_index: int = 0):
    """
    Generate a batch of data for the CBOW model.

    Parameters:
    data:        List of word indices.
    batch_size:  Number of words in each batch.
    num_skips:   How many times to reuse an input to generate a label.
    skip_window: How many words to consider left and right.
    data_index:  Index to start with in the data list. Default is 0.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Batch of context words and corresponding labels.
    """    
    assert batch_size < len(data)
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    window_size = 2 * skip_window + 1
    
    # Create a buffer to store the data
    buffer = deque(maxlen=window_size)
    for _ in range(window_size):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # Generates the batch of context words and labels
    for i in range(batch_size):
        mask = [1] * window_size
        mask[skip_window] = 0
        batch[i] = list(compress(buffer, mask))
        labels[i, 0] = buffer[skip_window]

        # Move the window
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        
    return batch, labels


def skipgram (data: List[int], batch_size: int, num_skips: int, skip_window: int, data_index: int = 0):
    """
    Generate a batch of data for the skipgram model.

    Parameters:
    data:        List of word indices.
    batch_size:  Number of words in each batch.
    num_skips:   How many times to reuse an input to generate a label.
    skip_window: How many words to consider left and right.
    data_index:  Index to start with in the data list. Default is 0.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Batch of input words and corresponding labels.
    """
    assert batch_size < len(data)
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    window_size = 2 * skip_window + 1

    # Create a buffer to store the data
    buffer = deque(maxlen=window_size)
    for _ in range(window_size):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # Generates the batch of context words and labels
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]

        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, window_size - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        # Move the window
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels


class WordEmbedding:
    def __init__(self, vocab_size: int,
                 batch_size: int,
                 embedding_size: int,
                 n_steps: int,
                 architecture: str,
                 loss_type: str,
                 optimizer: str):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.n_steps = n_steps
        self.architecture = architecture
        self.loss_type = loss_type
        self.optimizer = optimizer
        self.final_embeddings = None

    def tokenMapping(self, words):
        if isinstance(words, list):
            data, count, dictionary, reverse_dictionary = prepareData(words, self.vocab_size)
        elif isinstance(words, str) and words.endswith('.pkl'):
            data, count, dictionary, reverse_dictionary = preparePickleData(words, self.vocab_size)
        else:
            raise ValueError("Input must either be a list of words or a path to a pickle file if the list is too large to efficiently hold in memory.")
        if len(count) < self.vocab_size:
            raise ValueError(f"Vocab_size set too high. Only {len(count)} unique words in the provided training data.")
        else:
            self.data = data
            self.count = count
            self.dictionary = dictionary
            self.reverse_dictionary = reverse_dictionary
        return data

    def get_embedding(self, word):
        if word in self.dictionary:
            return self.final_embeddings[self.dictionary[word]]
        else:
            raise ValueError(f"Word '{word}' not in dictionary")

    def similar_by_word(self, word, top_n=5):
        word_vector = self.get_embedding(word)
        similarities = np.dot(self.final_embeddings, word_vector) / (np.linalg.norm(self.final_embeddings, axis=1) * np.linalg.norm(word_vector))
        similar_indices = np.argsort(-similarities)[:top_n]
        
        print(f"Similar indices for '{word}': {similar_indices}")
        similar_words = []
        for idx in similar_indices:
            if idx in self.reverse_dictionary:
                similar_words.append(self.reverse_dictionary[idx])
            else:
                print(f"Index {idx} not found in reverse_dictionary")
                similar_words.append(f"Index {idx} not found")
        
        return similar_words

    def similar_by_vector(self, vector, topn=1):
        similarities = np.dot(self.final_embeddings, vector) / (
            np.linalg.norm(self.final_embeddings, axis=1) * np.linalg.norm(vector)
        )
        best_indices = np.argsort(similarities)[-topn:][::-1]
        return [(self.reverse_dictionary[idx], similarities[idx]) for idx in best_indices]
    
    def eval(self):
        try:
            # Check embedding and nearby words
            embedding_king = self.get_embedding('king')
            similar_words = self.similar_by_word('king')
            print(f"\nEmbedding for 'king': {embedding_king}")
            print(f"\nWords similar to 'king': {similar_words}")
            
            # Test embedding relationships by analogy
            embedding_man = self.get_embedding('man')
            embedding_woman = self.get_embedding('woman')
            result_vector = embedding_king - embedding_man + embedding_woman
            closest_word = self.similar_by_vector(result_vector, topn=1)[0][0]
            print(f"\nThe word closest to 'king - man + woman' is: {closest_word}")
        except ValueError as e:
            print(e)


class Word2Vec(WordEmbedding, BaseEstimator, TransformerMixin):
    def __init__(self, vocab_size: int = 50000,
                 batch_size: int = 128,
                 embedding_size: int = 128,
                 n_steps: int = 10001,
                 architecture: str = 'skipgram',
                 loss_type: str = 'sampled_softmax_loss',
                 optimizer: str = 'adagrad',
                 num_skips: int = 4,
                 skip_window: int = 2,
                 n_neg_samples: int = 64,
                 learning_rate: float = 1.0,
                 valid_size: int = 16,
                 valid_window: int = 100):
        super().__init__(vocab_size, batch_size, embedding_size, n_steps, architecture, loss_type, optimizer)
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.n_neg_samples = n_neg_samples
        self.learning_rate = learning_rate
        self.valid_size = valid_size
        self.valid_window = valid_window

        self.chooseSamples()
        self.chooseGenerator()
        self.__init__model()

    def chooseSamples(self):
        valid_examples = np.array(random.sample(range(self.valid_window), self.valid_size))
        self.valid_examples = valid_examples
    
    def chooseGenerator(self):
        if self.architecture == 'skipgram':
            self.generator = skipgram
        elif self.architecture == 'cbow':
            self.generator = cbow
        else:
            raise ValueError("Architecture must be either 'skipgram' or 'cbow'.")
    
    def __init__model(self):
        self.embeddings = tf.Variable(tf.random.uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
        self.weights = tf.Variable(tf.random.truncated_normal([self.vocab_size, self.embedding_size], stddev=1.0 / np.sqrt(self.embedding_size)))
        self.biases = tf.Variable(tf.zeros([self.vocab_size]))

        if self.optimizer == 'adagrad':
            self.optimizer = tf.optimizers.Adagrad(learning_rate=self.learning_rate)
        elif self.optimizer == 'SGD':
            self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        
        # Compute the similarity distance metrics between individual embeddings
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keepdims=True))
        self.normalized_embeddings = self.embeddings / norm
        self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
        self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.valid_dataset)
        self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings, transpose_b=True)
        
    @tf.function
    def train_step(self, batch_data, batch_labels):
        with tf.GradientTape() as tape:
        
            if self.architecture == 'skipgram':
                embed = tf.nn.embedding_lookup(self.embeddings, batch_data)
            elif self.architecture == 'cbow':
                embed = tf.zeros([self.batch_size, self.embedding_size])
                for j in range(self.num_skips):
                    embed += tf.nn.embedding_lookup(self.embeddings, batch_data[:, j])
                embed /= self.num_skips
        
            if self.loss_type == 'sampled_softmax_loss':
                loss = tf.nn.sampled_softmax_loss(weights=self.weights,
                                                  biases=self.biases,
                                                  labels=batch_labels,
                                                  inputs=embed,
                                                  num_sampled=self.n_neg_samples,
                                                  num_classes=self.vocab_size)
            elif self.loss_type == 'nce_loss':
                loss = tf.nn.nce_loss(weights=self.weights,
                                      biases=self.biases,
                                      labels=batch_labels,
                                      inputs=embed,
                                      num_sampled=self.n_neg_samples,
                                      num_classes=self.vocab_size)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, [self.embeddings, self.weights, self.biases])
        self.optimizer.apply_gradients(zip(gradients, [self.embeddings, self.weights, self.biases]))
        return loss
    
    def fit(self, words):
        print(f"\nTokenizing words...")
        self.data = self.tokenMapping(words)
    
        print(f"\nTraining model...")
        average_loss = 0
        loss_values = []
        with tqdm(total=self.n_steps, desc="Progress", bar_format="{l_bar}{bar} | Elapsed: {elapsed} | ETA: {remaining} | {rate_fmt}{postfix}", leave=False) as pbar:
            for step in range(self.n_steps):
                batch_data, batch_labels = self.generator(self.data, self.batch_size, self.num_skips, self.skip_window)
                loss = self.train_step(batch_data, batch_labels)
                average_loss += loss
                if step % 500 == 0 and step > 0:
                    average_loss /= 500
                    pbar.set_postfix({'loss': average_loss.numpy(), 'step': step})
                    loss_values.append(average_loss)
                    average_loss = 0
                pbar.update(1)
    
        self.final_embeddings = self.normalized_embeddings.numpy()
        print(f"\nTraining has completed successfully with a final loss of {loss_values[-1]}.\n")
        return loss_values
    
    def fit_from_tokens(self, data, count, dictionary, reverse_dictionary):
        self.data = data
        self.count = count
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary

        print(f"\nTraining model...")
        average_loss = 0
        for step in tqdm(range(self.n_steps), desc="Overall Progress", mininterval=1.0, bar_format="{l_bar}{bar} | Elapsed: {elapsed} | ETA: {remaining} | {rate_fmt}"):
            step_loss = 0
            num_batches = len(self.data) // self.batch_size
            data_index = 0
            for batch in tqdm(range(num_batches), desc="Batch Progress", mininterval=1.0, bar_format="{l_bar}{bar} | Elapsed: {elapsed} | ETA: {remaining} | {rate_fmt}"):
                batch_data, batch_labels = self.generator(self.data, self.batch_size, self.num_skips, self.skip_window, data_index=data_index)
                loss = self.train_step(batch_data, batch_labels)
                step_loss += loss
                data_index += self.batch_size
            step_loss /= num_batches
            average_loss += step_loss
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                tqdm.write(f'Average loss at step {step}: {average_loss}')
                average_loss = 0
        
        self.final_embeddings = self.normalized_embeddings.numpy()
        print(f"\nTraining has completed successfully.\n")
        return self
    