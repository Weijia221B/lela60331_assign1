# Data Description

1. `Compiled_Reviews.txt`: Training loss curve visualization
2. `embeddings.npy`: Generated word embeddings

## Compiled_Reviews.txt
36547 reviews taken from Amazon
- Each review includes three labels: 
  - Sentiment (positive, negative)
  - Product type (one of 24 categories)
  - Helpfulness (helpful, unhelpful, neutral)


## embeddings.npy
embeddings.npy is created by processing review data from Compiled_Reviews.txt, tokenizing the text, and generating 300-dimensional embeddings using a pre-trained Word2Vec model. Each review's embedding is obtained by using average of the vectors of its tokens. 
