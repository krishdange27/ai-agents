# Intrusion Detection Agent Results

## Engineering Question

How do model initialization and training stability affect the quality of learned probabilistic representations?

The objective was not simply to train a Hidden Markov Model, but to investigate how design choices influence what the model learns and how effectively it separates normal behaviour from anomalous activity.

---

## Agent Design

The agent models network activity as a sequence of discrete observations and learns patterns of normal behaviour using a Hidden Markov Model.

Anomaly detection is performed by computing the log-likelihood of an observation sequence under the trained model.

Sequences with sufficiently low likelihood are flagged as anomalous.

### Key Design Decisions

#### Asymmetric State Initialization

Symmetric initialization frequently caused Baum-Welch training to converge toward nearly identical hidden states.

Introducing asymmetric initialization encouraged state specialization and produced more meaningful hidden-state representations.

#### Scaled Forward-Backward Computation

Long observation sequences quickly caused probability underflow during training.

Scaled forward-backward passes were implemented to maintain numerical stability and enable reliable Baum-Welch optimization.

#### Statistical Thresholding

Detection thresholds are derived from the distribution of training scores:

```text
threshold = μ − 1.5σ
```

This provides a simple and interpretable mechanism for anomaly detection.

---

## Experimental Setup

### Training Data

The model was trained on 30 sequences of synthetic normal network traffic averaging 19 observations per sequence.

Training data was generated across two behavioural regimes:

* Baseline Activity
* Active User Sessions

This provides the Hidden Markov Model with meaningful behavioural variation while remaining representative of normal system operation.

### Evaluation Data

Three anomalous traffic patterns were generated:

* Brute-Force Login Activity
* Reconnaissance Activity
* Data Exfiltration Activity

Each sequence was scored using the trained HMM and compared against the anomaly threshold.

---

## Results

| Metric                    |  Value |
| ------------------------- | -----: |
| Normal Mean Score         | -14.24 |
| Normal Standard Deviation |   4.74 |
| Detection Threshold       | -21.36 |
| Anomalous Mean Score      | -26.86 |
| Separation Gap            |  12.61 |

### Classification Performance

| Sequence Type              | Status               |
| -------------------------- | -------------------- |
| Normal Traffic             | Correctly Classified |
| Brute-Force Activity       | Correctly Flagged    |
| Reconnaissance Activity    | Correctly Flagged    |
| Data Exfiltration Activity | Correctly Flagged    |

---

## Key Observation

The most important factor was not increasing model complexity, but improving training stability.

Asymmetric initialization enabled the model to learn genuinely distinct hidden states, while scaled forward-backward computation ensured reliable optimization throughout training.

The resulting model produced a separation gap of 12.61 between normal and anomalous behaviour, allowing all evaluation scenarios to be correctly classified.

---

## Engineering Insight

This experiment highlights a common principle in probabilistic machine learning:

> A stable model often matters more than a larger model.

The majority of performance gains came from improving how the model learned rather than increasing the number of states or parameters.

Training stability, initialization strategy, and likelihood calibration had a greater impact on anomaly detection quality than model complexity alone.

---

## Limitations

The model correctly identifies all three attack scenarios as anomalous.

However, Brute-Force and Reconnaissance sequences receive nearly identical likelihood scores because both encode to observation symbols absent from the training distribution.

The agent is designed as an anomaly detector rather than a multi-class attack classifier.

Distinguishing between different attack categories would require an additional classification layer built on top of the anomaly detection system.

---

## Demo

Run:

```bash
cd probabilistic_agents/intrusion_detection
python3 demo.py
```

The demonstration:

* Scores normal network activity
* Evaluates anomalous traffic sequences
* Computes likelihood-based anomaly scores
* Applies statistical thresholding
* Reports classification decisions and anomaly separation metrics
