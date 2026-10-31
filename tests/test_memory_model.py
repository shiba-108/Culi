import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from memory_model import GatedMemoryNetwork


class TestMemoryModel(unittest.TestCase):
    def test_parameter_count(self):
        model = GatedMemoryNetwork(input_size=1, hidden_size=4, output_size=1, seed=1)
        self.assertEqual(model.parameter_count, 97)

    def test_training_reduces_loss(self):
        model = GatedMemoryNetwork(input_size=1, hidden_size=4, output_size=1, learning_rate=0.1, seed=7)
        samples = [([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]], [0.0]), ([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [1.0])]
        result = model.train(samples, epochs=20)
        self.assertLess(result.final_loss, result.initial_loss)

    def test_save_load_roundtrip(self):
        model = GatedMemoryNetwork(input_size=1, hidden_size=3, output_size=1, learning_rate=0.1, seed=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = GatedMemoryNetwork.load(path)
            self.assertEqual(model.to_dict(), loaded.to_dict())


if __name__ == "__main__":
    unittest.main()
