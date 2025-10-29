# Simple Deep Network for Image Classification

## Problem Description

In this assignment, you will build, train, and evaluate a simple deep neural network for image classification using **Scikit-learn's MLPClassifier**. You will use the Fashion MNIST dataset, which consists of 28x28 grayscale images of 10 different types of clothing.

## Learning Objectives

- Understand the basics of building a neural network with Scikit-learn
- Learn how to configure a Multi-Layer Perceptron (MLP) classifier
- Train and evaluate a neural network model
- Work with the Fashion MNIST dataset

## Why Scikit-learn?

This version uses **Scikit-learn** instead of TensorFlow/PyTorch because:
- ✅ **Simplest Installation**: Just `pip install scikit-learn` - works everywhere!
- ✅ **No Path Issues**: Lightweight package, no Windows path length problems
- ✅ **Beginner-Friendly**: Clean, simple API
- ✅ **Fast Setup**: Get coding immediately, zero configuration
- ✅ **Same Concepts**: Still learning neural networks, backpropagation, optimization

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or manually:
   ```bash
   pip install scikit-learn numpy
   ```

   **That's it!** No complex installation, no version conflicts! 🎉

## Instructions

1. Open the `assignment.py` file.
2. You will find three function definitions. Your task is to implement them.
   * **Task 1**: Implement the `build_model` function to create an MLPClassifier with a hidden layer of 128 neurons, ReLU activation, and Adam optimizer.
   * **Task 2**: Implement the `train_model` function to train the model on the training data.
   * **Task 3**: Implement the `evaluate_model` function to evaluate the trained model on the test data. Return loss (float) and accuracy (float percentage).

## Testing Your Solution

Run the test file to verify your implementation:

```bash
python -m unittest test
```

Or run the sample solution directly:

```bash
python sample_submission.py
```

## Expected Results

After training for 5 epochs:
- Training accuracy: ~85-88%
- Test accuracy: ~82-85%

## Need Help?

Check `sample_submission.py` for a complete working solution.

## Model Architecture

Your model should have the following structure:
```
Input (784) → Dense(128, ReLU) → Dense(10, Softmax) → Output
```

The Fashion MNIST images (28×28) are automatically flattened to 784 features.

## Tips

1. Use `MLPClassifier()` from `sklearn.neural_network`
2. Set `hidden_layer_sizes=(128,)` for one hidden layer with 128 neurons
3. Use `activation='relu'` for ReLU activation
4. Use `solver='adam'` for Adam optimizer
5. Set `max_iter=epochs` to control training duration
6. Use `model.score()` to get accuracy (returns 0-1, multiply by 100)
7. Use `log_loss()` from `sklearn.metrics` for loss calculation

## Example Code Structure

```python
from sklearn.neural_network import MLPClassifier

# Build model
model = MLPClassifier(
    hidden_layer_sizes=(128,),
    activation='relu',
    solver='adam',
    max_iter=5
)

# Train model
model.fit(train_images, train_labels)

# Evaluate model
accuracy = model.score(test_images, test_labels) * 100
```

## Comparison with Other Frameworks

| Feature | Scikit-learn ✅ | TensorFlow | PyTorch |
|---------|----------------|------------|---------|
| Installation | ✅ Simple | ⚠️ Complex | ⚠️ Complex |
| Windows Compatible | ✅ Yes | ❌ Path issues | ❌ Version issues |
| Code Length | ✅ Shortest | Medium | Longer |
| Learning Curve | ✅ Easiest | Moderate | Moderate |
| Production Ready | ✅ Yes | ✅ Yes | ✅ Yes |

## What You'll Learn

Even though we're using Scikit-learn, you'll still learn:
- ✅ Neural network architecture
- ✅ Hidden layers and activation functions
- ✅ Backpropagation (handled internally)
- ✅ Optimization algorithms (Adam)
- ✅ Training and evaluation
- ✅ Loss functions and metrics

The concepts are **identical** - just with a simpler API!

## Troubleshooting

### Dataset Download Issues?
If the dataset fails to download, it will be cached for next time. Just re-run the script.

### Import Errors?
Make sure scikit-learn is installed:
```bash
pip install --upgrade scikit-learn numpy
```

### Slow Training?
The first epoch is slower due to internal optimization. Subsequent epochs are faster.

Good luck! 🚀

