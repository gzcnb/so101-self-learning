"""Run the extracted online compensator against a synthetic plant, without hardware."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.rls import AERIAL_HOME, CerebellumRLS


def main():
    learner = CerebellumRLS(np.zeros(4), mode="rls")
    reference = AERIAL_HOME.copy()
    zero = np.zeros(4)
    # Synthetic steady tracking errors in encoder ticks.
    disturbance = np.array([12.0, 25.0, -18.0, 9.0])
    errors = []
    for step in range(1000):
        compensation, features = learner.predict(reference, zero, zero)
        measured = reference + compensation - disturbance
        learner.update(features, compensation, reference, measured)
        errors.append(float(np.mean(np.abs(reference - measured))))
    print(json.dumps({
        "experiment": "synthetic constant-disturbance tracking",
        "hardware_connected": False,
        "initial_mean_absolute_error_ticks": round(float(np.mean(errors[:30])), 3),
        "final_mean_absolute_error_ticks": round(float(np.mean(errors[-30:])), 3),
        "maximum_compensation_ticks": learner.max_ff,
        "learned_weights_shape": list(learner.W.shape),
    }, indent=2))


if __name__ == "__main__":
    main()
