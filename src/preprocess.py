import numpy as np
import pandas as pd
import re
import gensim.downloader as api
from tqdm import tqdm
import os
class DataPreprocessor:
    def __init__(self):
        """Load word2vec model"""
        self.w2v = api.load('word2vec-google-news-300')
        self.vocab = [x for x in self.w2v.key_to_index.keys()]
        
    def load_data(self, filepath):
        """Load and parse review data"""
        reviews = []
        sentiment_ratings = []
        product_types = []
        helpfulness_ratings = []
        
        with open(filepath) as f:
            for line in f.readlines()[1:]:
                fields = line.rstrip().split('\t')
                reviews.append(fields[0])
                sentiment_ratings.append(fields[1])
                product_types.append(fields[2])
                helpfulness_ratings.append(fields[3])
                
        return reviews, sentiment_ratings, product_types, helpfulness_ratings
    
    def process_text(self, reviews):
        """Text processing"""
        # Tokenization
        tokenized_sents = [re.findall("[^ ]+", txt) for txt in reviews]
        tokens = []
        for s in tokenized_sents:
            tokens.extend(s)
        types = set(tokens)
        
        # Get words in word2vec vocabulary
        vocab_dict = {word: idx for idx, word in enumerate(self.vocab)}
        indices = [vocab_dict[x] for x in types if x in vocab_dict]
        vocab_set = set(self.vocab)
        types_inc = [x for x in types if x in vocab_set]
        
        return types_inc, indices
    
    def generate_or_load_embeddings(self, reviews, types_inc, indices, cache_dir='embeddings_cache'):
        """Generate or load cached embeddings"""
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache file name (based on input data hash)
        data_hash = hash(str(reviews))
        cache_file = os.path.join(cache_dir, f'embeddings_{data_hash}.npy')
        
        # If cache exists, load directly
        if os.path.exists(cache_file):
            print("Loading cached embeddings...")
            return np.load(cache_file)
        
        # Otherwise, generate new embeddings
        print("Generating new embeddings...")
        M = self.w2v[indices]
        types_inc_dict = {word: idx for idx, word in enumerate(types_inc)}
        embeddings = []
        
        for rev in tqdm(reviews, desc="Generating embeddings"):
            tokens = re.findall("[^ ]+", rev)
            token_indices = [types_inc_dict[t] for t in tokens if t in types_inc_dict]
            if token_indices:
                this_vec = np.mean(M[token_indices], axis=0)
                embeddings.append(this_vec)
        
        embeddings = np.array(embeddings).squeeze()
        
        # Save to cache
        np.save(cache_file, embeddings)
        
        return embeddings
    
    def split_data(self, embeddings, labels, test_size=0.2):
        """Split data into training and testing sets"""
        np.random.seed(10)  # Add random seed to keep results consistent
        n_samples = len(embeddings)
        n_train = int(n_samples * (1 - test_size))
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train = embeddings[train_idx]
        X_test = embeddings[test_idx]
        y_train = [labels[i] for i in train_idx]
        y_test = [labels[i] for i in test_idx]
        
        return X_train, X_test, y_train, y_test 