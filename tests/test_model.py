"""Unit tests for multimodal fusion model architecture and training."""

import unittest
import numpy as np

from src.fusion_model import NumpyMultimodalClassifier


class TestMultimodalModel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.x_struct = rng.normal(size=(80, 20)).astype(np.float32)
        self.x_text = rng.normal(size=(80, 16)).astype(np.float32)
        self.y = rng.integers(0, 2, size=80).astype(np.float64)

    def test_forward_shapes_and_probability_bounds(self):
        for mode in ["structured", "text", "multimodal"]:
            model = NumpyMultimodalClassifier(
                mode=mode, struct_dim=20, text_dim=16, hidden_dim=24, seed=42
            )
            probs, gate = model.predict_with_gate(
                x_struct=self.x_struct if mode != "text" else None,
                x_text=self.x_text if mode != "structured" else None,
            )
            self.assertEqual(probs.shape, (80,))
            self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
            if mode == "multimodal":
                self.assertIsNotNone(gate)
                self.assertEqual(gate.shape, (80, 48))

    def test_training_convergence(self):
        model = NumpyMultimodalClassifier(
            mode="multimodal", struct_dim=20, text_dim=16, hidden_dim=24, lr=0.01, seed=42
        )
        hist = model.fit(
            self.x_struct[:60],
            self.x_text[:60],
            self.y[:60],
            self.x_struct[60:],
            self.x_text[60:],
            self.y[60:],
            epochs=10,
            batch_size=20,
            patience=5,
        )
        self.assertGreaterEqual(len(hist["train_loss"]), 1)
        self.assertLess(hist["train_loss"][-1], hist["train_loss"][0] + 0.5)


if __name__ == "__main__":
    unittest.main()
