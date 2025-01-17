# LELA60331_assign1 - Amazon Review Analysis
This project implements an Amazon review analysis system based on logistic regression that performs two tasks: sentiment analysis and helpfulness classification.

## Project Structure
```
amazon-review-analysis/
├── README.md
├── requirements.txt
├── src/
│   ├── preprocess.py
│   ├── model.py 
│   └── main.py
├── data/
│   └── README.md
│   └── Compiled_Reviews.txt
└── output/
│    └── README.md
└── refs/

```

## Features

- Sentiment analysis (positive/negative)
- Review helpfulness classification（helpful/unhelpful/neutral)
- Word2Vec text vectorization
- Multi-class logistic regression classifier
- Detailed evaluation metrics (precision, recall)
- Training process visualization

## System Requirements

- Python 3.7+
- 8GB+ RAM
- Stable internet connection (required for first-time Word2Vec model download)
- Operating System: Windows/Linux/MacOS

## Installation

1. Clone the repository:
```
git clone https://github.com/Weijia221B/lela60331_assign1.git
cd amazon-review-analysis
```

2. Create and activate virtual environment:
   
For Windows:
```
python -m venv venv
amazon_analysis\Scripts\activate.bat
```

For Linux/Mac:
```
python -m venv amazon_analysis
source amazon_analysis/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```


## Data Preparation (Optional, you can directly use Compiled_Reviews.txt saved in data/)

1. Download the dataset:
```
wget https://raw.githubusercontent.com/cbannard/lela60331_24-25/refs/heads/main/coursework/Compiled_Reviews.txt
```
(or you can download the data together with the project)

2. Move the file to data directory:
```
mkdir -p data
mv Compiled_Reviews.txt data/
```

## Usage Guide

### 1. Sentiment Analysis Task

Run:
```
# using sigmoid
python src/main.py \
    --data_path data/Compiled_Reviews.txt \
    --task sentiment \
    --use_sigmoid \
    --learning_rate 0.01 \
    --n_iters 2500 \
    --plot_loss \

# using softmax
python src/main.py \
    --data_path data/Compiled_Reviews.txt \
    --task sentiment \
    --learning_rate 0.01 \
    --n_iters 2500 \
    --plot_loss
```

### 2. Helpfulness Classification (Reproducing Original Results)

Run the following command:
```
python src/main.py \
--data_path data/Compiled_Reviews.txt \
--task helpfulness \
--learning_rate 0.01 \
--n_iters 2500 \
--plot_loss \
--embeddings_cache_dir data
```

### 3. Save Word Embeddings

To save generated word embeddings for later use:
```
python src/main.py \
--data_path data/Compiled_Reviews.txt \
--task helpfulness \
--save_embeddings \
--output_dir output \
--embeddings_cache_dir data # re-use saved word embeddings
```


## Command Line Arguments

- `--data_path`: Path to data file (required)
- `--task`: Analysis task
  - `helpfulness`: Review helpfulness classification (default)
  - `sentiment`: Sentiment analysis
- `--learning_rate`: Learning rate (default: 0.01)
- `--n_iters`: Number of training iterations (default: 2500)
- `--plot_loss`: Whether to plot and save loss curve
- `--save_embeddings`: Whether to save generated embeddings
- `--output_dir`: Output directory (default: 'output')
- `--embeddings_cache_dir`: directory for saving word embeddings(avoid repetive processing)

## Output Files

All output files are saved in the specified output directory (default: 'output/'):

1. `loss_curve.png`: Training loss curve visualization
2. `embeddings.npy`: Generated word embeddings (if --save_embeddings is used)
3. `metrics.json`: Detailed evaluation metrics for each class
4. `model_weights.npy`: Trained model weights

## Example Output Structure
```
output/
├── loss_curve.png
├── embeddings.npy
├── metrics.json
└── model_weights.npy
```

## Result Reproduction Guide

To ensure exact reproduction of original results:

1. Fixed Random Seeds
   - Random seed 10 is used in both preprocessing and model training
   - This ensures consistent data splitting and model initialization

2. Data Split Ratio
   - Training set: 80%
   - Test set: 20%

3. Hyperparameters
   - Learning rate: 0.01
   - Number of iterations: 2500
   - Word2Vec dimensions: 300

## Troubleshooting

1. Word2Vec Model Download Issues
   - Ensure stable internet connection
   - Try downloading during off-peak hours
   - Check if you have sufficient disk space

2. Memory Issues
   - Ensure at least 8GB available RAM
   - Close other memory-intensive applications
   - Consider reducing batch size if needed

3. Result Discrepancies
   - Verify Python and package versions match requirements
   - Confirm correct data file is being used
   - Check all parameter settings
   - Ensure random seeds are set correctly

## Package Versions
```
Required package versions for exact reproduction:
numpy==1.19.2
pandas==1.2.4
gensim==4.0.1
tqdm==4.61.0
matplotlib==3.4.2
scikit-learn==0.24.2
```


