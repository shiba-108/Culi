import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recurrent_model import RecurrentNetwork


class TestRecurrentModel(unittest.TestCase):
    def test_parameter_count(self):
        model = RecurrentNetwork(input_size=1, hidden_size=4, output_size=1, seed=1)
        self.assertEqual(model.parameter_count, 29)

    def test_training_reduces_loss(self):
        model = RecurrentNetwork(input_size=1, hidden_size=4, output_size=1, learning_rate=0.1, seed=7)
        samples = [
            ([[0.0], [0.0]], [0.0]),
            ([[0.0], [1.0]], [1.0]),
            ([[1.0], [0.0]], [1.0]),
            ([[1.0], [1.0]], [0.0]),
        ]
        result = model.train(samples, epochs=2000)
        self.assertLess(result.final_loss, result.initial_loss)

    def test_training_changes_parameters(self):
        model = RecurrentNetwork(input_size=1, hidden_size=4, output_size=1, learning_rate=0.1, seed=7)
        before = model.parameter_signature()
        samples = [([[0.0], [0.0]], [0.0]), ([[1.0], [1.0]], [0.0])]
        model.train(samples, epochs=10)
        self.assertNotEqual(before, model.parameter_signature())

    def test_save_load_roundtrip(self):
        model = RecurrentNetwork(input_size=1, hidden_size=3, output_size=1, learning_rate=0.2, seed=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = RecurrentNetwork.load(path)
            self.assertEqual(model.to_dict(), loaded.to_dict())
            self.assertEqual(model.predict_sequence([[0.2], [0.7]]), loaded.predict_sequence([[0.2], [0.7]]))


if __name__ == "__main__":
    unittest.main()
