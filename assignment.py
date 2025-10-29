from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

def get_data():
    """
    Loads and preprocesses the Fashion MNIST dataset.
    Returns training and test data as numpy arrays.
    """
    # Load Fashion MNIST dataset
    print("Downloading Fashion MNIST dataset...")
    # Use data_id instead of name for more reliable fetching
    X, y = fetch_openml(data_id=40996, return_X_y=True, parser='auto')
    
    # Convert to numpy arrays and normalize
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int64')
    
    # Normalize pixel values to be between 0 and 1
    X = X / 255.0
    
    # Split into train and test sets (60000 train, 10000 test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )
    
    return (X_train, y_train), (X_test, y_test)

def build_model():
    """
    Builds a simple neural network model using MLPClassifier.
    
    TODO: Implement this function
    
    Your model should have:
    - Hidden layer with 128 neurons
    - ReLU activation function
    - Adam optimizer
    - Output layer with 10 neurons (automatically created for 10 classes)
    
    Hints:
    - Use MLPClassifier with hidden_layer_sizes=(128,)
    - Set activation='relu'
    - Set solver='adam'
    - Set max_iter=10 (we'll train for more epochs later)
    - Set random_state=42 for reproducibility
    
    Return: An MLPClassifier instance
    
    Example:
        model = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation='relu',
            solver='adam',
            max_iter=10,
            random_state=42
        )
    """
    # Your code here
    pass

def train_model(model, train_images, train_labels, epochs=5):
    """
    Trains the neural network model.
    
    TODO: Implement this function
    
    Steps:
    1. Set model.max_iter = epochs
    2. Call model.fit(train_images, train_labels)
    3. The model will automatically train and show progress
    
    Args:
        model: MLPClassifier model to train
        train_images: Training images array (60000, 784)
        train_labels: Training labels array (60000,)
        epochs: Number of training epochs (iterations)
    
    Returns:
        Trained model
        
    Hint: 
        model.max_iter = epochs
        model.fit(train_images, train_labels)
    """
    # Your code here
    pass

def evaluate_model(model, test_images, test_labels):
    """
    Evaluates the model on the test set.
    
    TODO: Implement this function
    
    Steps:
    1. Use model.predict_proba() to get probability predictions
    2. Calculate loss using log_loss (from sklearn.metrics)
    3. Use model.score() to get accuracy (returns 0-1)
    4. Convert accuracy to percentage (multiply by 100)
    
    Args:
        model: Trained MLPClassifier model
        test_images: Test images array (10000, 784)
        test_labels: Test labels array (10000,)
    
    Returns:
        test_loss: Average test loss (float)
        test_acc: Test accuracy as percentage (float, 0-100)
        
    Hints:
        y_pred_proba = model.predict_proba(test_images)
        test_loss = log_loss(test_labels, y_pred_proba)
        test_acc_decimal = model.score(test_images, test_labels)
        test_acc = test_acc_decimal * 100
    """
    # Your code here
    pass

