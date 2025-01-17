import argparse
import json
import os
from preprocess import DataPreprocessor
from model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

def ensure_output_dir(output_dir):
    """Ensure output directory exists"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def plot_loss(loss_history, save_path):
    """Plot and save loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history)), loss_history[1:])
    plt.xlabel("number of epochs")
    plt.ylabel("loss")
    plt.savefig(save_path)
    plt.close()

def save_metrics(metrics, save_path):
    """Save evaluation metrics to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def main(args):
    # Ensure output directory exists
    ensure_output_dir(args.output_dir)
    
    # Set random seed
    np.random.seed(10)
    
    # Data preprocessing
    preprocessor = DataPreprocessor()
    print("Loading data...")
    reviews, sentiment_ratings, product_types, helpfulness_ratings = preprocessor.load_data(args.data_path)
    
    print("Processing text...")
    types_inc, indices = preprocessor.process_text(reviews)
    
    # Use new generate_or_load_embeddings method
    embeddings = preprocessor.generate_or_load_embeddings(
        reviews, 
        types_inc, 
        indices,
        cache_dir=args.embeddings_cache_dir
    )
    
    # Save wordembeddings
    if args.save_embeddings:
        np.save(os.path.join(args.output_dir, "embeddings.npy"), embeddings)
    
    # Select labels based on task type
    if args.task == 'sentiment':
        labels = sentiment_ratings
        num_classes = 2
        # Convert sentiment ratings to binary labels (0/1)
        labels = [1 if rating == "positive" else 0 for rating in labels]
        use_sigmoid = args.use_sigmoid
    elif args.task == 'helpfulness':
        labels = helpfulness_ratings
        num_classes = 3
        use_sigmoid = False
    else:
        raise ValueError("Task must be either 'sentiment' or 'helpfulness'")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = preprocessor.split_data(embeddings, labels)
    
    # Train model
    model = LogisticRegression(
        num_features=300,
        num_classes=num_classes,
        learning_rate=args.learning_rate,
        use_sigmoid=use_sigmoid
    )
    
    loss_history = model.fit(X_train, y_train, n_iters=args.n_iters)
    
    # Save model weights
    np.save(os.path.join(args.output_dir, "model_weights.npy"), model.weights)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Save results
    save_metrics(metrics, os.path.join(args.output_dir, "metrics.json"))
    
    # Print results
    print("\nTest Results:")
    try:
        if args.task == 'sentiment':  # Binary classification task
            for metric_name, value in metrics.items():
                try:
                    if isinstance(value, dict):
                        print(f"\n{metric_name}:")
                        for sub_metric, sub_value in value.items():
                            print(f"  {sub_metric}: {sub_value:.4f}")
                    else:
                        print(f"{metric_name}: {value:.4f}")
                except Exception as e:
                    print(f"Warning: Failed to print metric {metric_name}")
                    continue
        else:  # Multi-class task
            for class_name, class_metrics in metrics.items():
                try:
                    print(f"\n{class_name}:")
                    for metric_name, value in class_metrics.items():
                        print(f"  {metric_name}: {value:.4f}")
                except Exception as e:
                    print(f"Warning: Failed to print metrics for class {class_name}")
                    continue
    except Exception as e:
        print(f"Warning: Error occurred while printing metrics, continuing...")
    
    # Plot loss curve
    if args.plot_loss:
        plot_loss(loss_history, os.path.join(args.output_dir, "loss_curve.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Amazon Review Analysis')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the review data file')
    parser.add_argument('--task', type=str, default='helpfulness',
                      choices=['sentiment', 'helpfulness'],
                      help='Analysis task to perform')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for training')
    parser.add_argument('--n_iters', type=int, default=2500,
                      help='Number of training iterations')
    parser.add_argument('--plot_loss', action='store_true',
                      help='Whether to plot and save the loss curve')
    parser.add_argument('--save_embeddings', action='store_true',
                      help='Whether to save the generated embeddings')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save outputs')
    parser.add_argument('--use_sigmoid', action='store_true',
                      help='Use sigmoid for binary classification (only valid for sentiment task)')
    parser.add_argument('--embeddings_cache_dir', type=str, default='embeddings_cache',
                      help='Directory to cache generated embeddings')
    
    args = parser.parse_args()
    
    if args.use_sigmoid and args.task != 'sentiment':
        raise ValueError("Sigmoid can only be used with sentiment analysis task")
    
    main(args) 