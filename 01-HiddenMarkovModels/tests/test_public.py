import unittest
import numpy as np
from src.pipeline import (
    encode_observations,
    build_model,
    detect_intrusion,
    compare_models
)

class TestPublic(unittest.TestCase):

    def test_encoding_runs(self):
        raw = [{"failed_logins": 3, "bytes_sent": 2000, "dns_queries": 50}]
        encoded = encode_observations(raw)
        self.assertTrue(isinstance(encoded, list))

    def test_model_builds(self):
        sequences = [[0,1,2],[1,2,3]]
        model = build_model(sequences)
        self.assertTrue(hasattr(model, "forward"))

    def test_output_types(self):
        sequences = [[0,1,2]]
        model = build_model(sequences)
        score = model.forward([0,1,2])
        self.assertTrue(isinstance(score, float))

    
class TestPublicEncoding(unittest.TestCase):

    def test_encoding_output_type(self):
        raw = [{"failed_logins": 2, "bytes_sent": 5000, "dns_queries": 30}]
        encoded = encode_observations(raw)

        self.assertIsInstance(encoded, list)
        self.assertIsInstance(encoded[0], int)

    def test_encoding_range(self):
        raw = [
            {"failed_logins": 0, "bytes_sent": 1000, "dns_queries": 10},
            {"failed_logins": 20, "bytes_sent": 50000, "dns_queries": 300},
        ]
        encoded = encode_observations(raw)

        for val in encoded:
            self.assertGreaterEqual(val, 0)
            self.assertLess(val, 50)  # loose upper bound

    def test_encoding_consistency(self):
        raw = [{"failed_logins": 5, "bytes_sent": 20000, "dns_queries": 80}]
        e1 = encode_observations(raw)
        e2 = encode_observations(raw)

        self.assertEqual(e1, e2)


class TestPublicModel(unittest.TestCase):

    def test_model_has_required_methods(self):
        sequences = [[0, 1, 2], [1, 2, 3]]
        model = build_model(sequences)

        self.assertTrue(hasattr(model, "forward"))
        self.assertTrue(hasattr(model, "viterbi"))

    def test_model_forward_runs(self):
        sequences = [[0, 1, 2], [1, 2, 3]]
        model = build_model(sequences)

        score = model.forward([0, 1, 2])
        self.assertIsInstance(score, float)

    def test_model_viterbi_runs(self):
        sequences = [[0, 1, 2], [1, 2, 3]]
        model = build_model(sequences)

        path = model.viterbi([0, 1, 2])
        self.assertEqual(len(path), 3)


class TestPublicDetection(unittest.TestCase):

    def test_detect_intrusion_returns_float(self):
        sequences = [[0, 1, 2], [1, 2, 3]]
        model = build_model(sequences)

        score = detect_intrusion(model, [0, 1, 2])
        self.assertIsInstance(score, float)

    def test_empty_sequence(self):
        sequences = [[0, 1, 2]]
        model = build_model(sequences)

        score = detect_intrusion(model, [])
        self.assertTrue(isinstance(score, float))
  
class TestPublicComparison(unittest.TestCase):

    def test_compare_models_output(self):
        sequences = [[0, 1, 2], [1, 2, 3]]

        model_a = build_model(sequences)
        model_b = build_model(sequences)

        result = compare_models([0, 1, 2], model_a, model_b)

        self.assertIn(result, ["A", "B"])

class TestPublicRobustness(unittest.TestCase):

    def test_variable_length_sequences(self):
        sequences = [
            [0, 1],
            [1, 2, 3, 4],
            [2, 3, 4, 5, 6, 7]
        ]

        model = build_model(sequences)

        for seq in sequences:
            score = detect_intrusion(model, seq)
            self.assertIsInstance(score, float)

    def test_repeated_training_stability(self):
        """
        Building model twice should not crash or diverge badly.
        """

        sequences = [[0, 1, 2], [1, 2, 3]]

        model1 = build_model(sequences)
        model2 = build_model(sequences)

        s1 = model1.forward([0, 1, 2])
        s2 = model2.forward([0, 1, 2])

        self.assertFalse(np.isnan(s1))
        self.assertFalse(np.isnan(s2))

    def test_large_symbol_values(self):
        """
        Encoding may produce larger symbol values.
        Model should still handle gracefully.
        """

        sequences = [[10, 15, 20], [5, 25, 30]]
        model = build_model(sequences)

        score = detect_intrusion(model, [10, 15, 20])
        self.assertIsInstance(score, float)

class TestPublicAlgorithmCorrectness(unittest.TestCase):

    def test_forward_less_than_zero(self):
        """Log-probabilities must always be negative."""
        model = build_model([[0, 1, 2], [1, 2, 3]])
        score = model.forward([0, 1, 2])
        self.assertLess(score, 0)