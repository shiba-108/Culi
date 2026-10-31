import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_model import NeuralNetwork


class TestNeuralModel(unittest.TestCase):
    def test_parameter_count(self):
        model = NeuralNetwork(layer_sizes=[2, 4, 1], seed=1)
        self.assertEqual(model.parameter_count, 17)

    def test_training_reduces_loss(self):
        model = NeuralNetwork(layer_sizes=[2, 4, 1], learning_rate=0.5, seed=7)
        samples = [([0.0, 0.0], [0.0]), ([0.0, 1.0], [1.0]), ([1.0, 0.0], [1.0]), ([1.0, 1.0], [0.0])]
        result = model.train(samples, epochs=1000)
        self.assertLess(result.final_loss, result.initial_loss)

    def test_training_changes_parameters(self):
        model = NeuralNetwork(layer_sizes=[2, 4, 1], learning_rate=0.5, seed=7)
        before = model.parameter_signature()
        samples = [([0.0, 0.0], [0.0]), ([1.0, 1.0], [0.0])]
        model.train(samples, epochs=10)
        self.assertNotEqual(before, model.parameter_signature())

    def test_save_load_roundtrip(self):
        model = NeuralNetwork(layer_sizes=[2, 3, 1], learning_rate=0.2, seed=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = NeuralNetwork.load(path)
            self.assertEqual(model.to_dict(), loaded.to_dict())
            self.assertEqual(model.predict([0.2, 0.7]), loaded.predict([0.2, 0.7]))


if __name__ == "__main__":
    unittest.main()
