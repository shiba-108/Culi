import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recurrent_model import RecurrentNetwork
from recurrent_adapter import RecurrentModelAdapter


class TestRecurrentAdapter(unittest.TestCase):
    def test_adapter_predicts_xor(self):
        model = RecurrentNetwork(input_size=1, hidden_size=4, output_size=1, learning_rate=0.1, seed=7)
        samples = [
            ([[0.0], [0.0]], [0.0]),
            ([[0.0], [1.0]], [1.0]),
            ([[1.0], [0.0]], [1.0]),
            ([[1.0], [1.0]], [0.0]),
        ]
        model.train(samples, epochs=5000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            adapter = RecurrentModelAdapter(path)
            self.assertEqual(adapter.generate("0 0"), "0")
            self.assertEqual(adapter.generate("0 1"), "1")
            self.assertEqual(adapter.generate("1 0"), "1")
            self.assertEqual(adapter.generate("1 1"), "0")


if __name__ == "__main__":
    unittest.main()
