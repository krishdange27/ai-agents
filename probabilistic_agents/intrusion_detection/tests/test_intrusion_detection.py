"""
test_intrusion_detection.py
Tests for the HMM-based intrusion detection agent.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import generate_raw_logs
from src.agent import encode_observations, build_model, detect_intrusion, compare_models


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_logs():
    return generate_raw_logs("krishdange27", n_sequences=10)

@pytest.fixture
def encoded_sequences(raw_logs):
    return [encode_observations(seq) for seq in raw_logs]

@pytest.fixture
def trained_model(encoded_sequences):
    return build_model(encoded_sequences)


# ── Encoding tests ────────────────────────────────────────────────────────────

class TestObservationEncoding:

    def test_encoding_returns_list_of_ints(self, raw_logs):
        encoded = encode_observations(raw_logs[0])
        assert isinstance(encoded, list)
        assert all(isinstance(o, int) for o in encoded)

    def test_encoding_length_matches_input(self, raw_logs):
        seq = raw_logs[0]
        encoded = encode_observations(seq)
        assert len(encoded) == len(seq)

    def test_symbols_within_valid_range(self, raw_logs):
        for seq in raw_logs:
            encoded = encode_observations(seq)
            assert all(0 <= o < 50 for o in encoded), \
                f"Symbol out of range: {encoded}"

    def test_empty_sequence_encodes_to_empty(self):
        assert encode_observations([]) == []


# ── Model tests ───────────────────────────────────────────────────────────────

class TestModelBuilding:

    def test_model_has_valid_transition_matrix(self, trained_model):
        import numpy as np
        row_sums = trained_model.A.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_model_has_valid_emission_matrix(self, trained_model):
        import numpy as np
        row_sums = trained_model.B.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_model_has_valid_initial_distribution(self, trained_model):
        import numpy as np
        assert abs(trained_model.pi.sum() - 1.0) < 1e-6

    def test_model_forward_runs(self, trained_model, encoded_sequences):
        result = trained_model.forward(encoded_sequences[0])
        assert isinstance(result, float)

    def test_model_viterbi_runs(self, trained_model, encoded_sequences):
        states = trained_model.viterbi(encoded_sequences[0])
        assert isinstance(states, list)
        assert len(states) == len(encoded_sequences[0])


# ── Detection tests ───────────────────────────────────────────────────────────

class TestAnomalyDetection:

    def test_score_is_a_float(self, trained_model, encoded_sequences):
        score = detect_intrusion(trained_model, encoded_sequences[0])
        assert isinstance(score, float)

    def test_score_is_negative_log_likelihood(self, trained_model, encoded_sequences):
        score = detect_intrusion(trained_model, encoded_sequences[0])
        assert score <= 0.0

    def test_empty_sequence_returns_neutral_score(self, trained_model):
        score = detect_intrusion(trained_model, [])
        assert score == 0.0

    def test_anomalous_scores_lower_than_normal(self, trained_model, encoded_sequences):
        normal_score = detect_intrusion(trained_model, encoded_sequences[0])
        # Hand-crafted anomalous sequence: maxed-out suspicious features
        anomalous_raw = [
            {"failed_logins": 20, "bytes_sent": 49000, "dns_queries": 195}
        ] * 10
        anomalous_encoded = encode_observations(anomalous_raw)
        anomalous_score = detect_intrusion(trained_model, anomalous_encoded)
        assert anomalous_score < normal_score

    def test_large_observation_symbols_handled(self, trained_model):
        large_seq = [999, 888, 777]
        score = detect_intrusion(trained_model, large_seq)
        assert isinstance(score, float)


# ── Model comparison tests ────────────────────────────────────────────────────

class TestModelComparison:

    def test_compare_returns_A_or_B(self, trained_model, encoded_sequences):
        result = compare_models(encoded_sequences[0], trained_model, trained_model)
        assert result in ("A", "B")

    def test_identical_models_returns_A(self, trained_model, encoded_sequences):
        result = compare_models(encoded_sequences[0], trained_model, trained_model)
        assert result == "A"

    def test_empty_sequence_returns_A(self, trained_model):
        result = compare_models([], trained_model, trained_model)
        assert result == "A"


# ── Robustness tests ──────────────────────────────────────────────────────────

class TestRobustness:

    def test_variable_length_sequences(self, trained_model):
        for length in [1, 5, 10, 20]:
            seq = [5] * length
            score = detect_intrusion(trained_model, seq)
            assert isinstance(score, float)

    def test_repeated_training_stays_stable(self, encoded_sequences):
        model_a = build_model(encoded_sequences)
        model_b = build_model(encoded_sequences)
        score_a = detect_intrusion(model_a, encoded_sequences[0])
        score_b = detect_intrusion(model_b, encoded_sequences[0])
        assert abs(score_a - score_b) < 5.0

    def test_single_step_sequence(self, trained_model):
        score = detect_intrusion(trained_model, [0])
        assert isinstance(score, float)
