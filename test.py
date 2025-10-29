import unittest
from sklearn.neural_network import MLPClassifier
from assignment import get_data, build_model, train_model, evaluate_model

class TestSimpleDeepNetwork(unittest.TestCase):
    def test_network(self):
        """Test the complete neural network pipeline"""
        print("\n" + "="*60)
        print("Testing Simple Deep Network Implementation")
        print("="*60)
        
        # Test data loading
        print("\n1. Testing data loading...")
        (train_images, train_labels), (test_images, test_labels) = get_data()
        
        # Verify data shapes
        self.assertEqual(train_images.shape[0], 60000)
        self.assertEqual(train_images.shape[1], 784)  # 28x28 flattened
        self.assertEqual(train_labels.shape[0], 60000)
        self.assertEqual(test_images.shape[0], 10000)
        self.assertEqual(test_labels.shape[0], 10000)
        print("   ✓ Data shapes are correct")
        print(f"   ✓ Training samples: {train_images.shape[0]}")
        print(f"   ✓ Test samples: {test_images.shape[0]}")
        
        # Test model building
        print("\n2. Testing model building...")
        model = build_model()
        self.assertIsInstance(model, MLPClassifier)
        print("   ✓ Model is an MLPClassifier")
        
        # Check model configuration
        self.assertEqual(model.hidden_layer_sizes, (128,))
        self.assertEqual(model.activation, 'relu')
        self.assertEqual(model.solver, 'adam')
        print("   ✓ Hidden layer has 128 units")
        print("   ✓ Activation is ReLU")
        print("   ✓ Optimizer is Adam")
        
        # Test model training (only 2 epochs for faster testing)
        print("\n3. Testing model training (2 epochs for speed)...")
        trained_model = train_model(model, train_images[:10000], train_labels[:10000], epochs=2)
        self.assertIsNotNone(trained_model)
        print("   ✓ Model trained successfully")
        
        # Test model evaluation
        print("\n4. Testing model evaluation...")
        test_loss, test_acc = evaluate_model(trained_model, test_images, test_labels)
        
        # Verify return types
        self.assertIsInstance(test_loss, float)
        self.assertIsInstance(test_acc, (float, int))
        print(f"   ✓ Test loss: {test_loss:.4f}")
        print(f"   ✓ Test accuracy: {test_acc:.2f}%")
        
        # Check for reasonable accuracy (should be > 60% even with limited training)
        self.assertGreater(test_acc, 60.0)
        print("   ✓ Accuracy is above 60%")
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60 + "\n")

if __name__ == '__main__':
    unittest.main(verbosity=2)

