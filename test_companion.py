"""Regression checks for generation and decoding."""
import copy
import itertools
import pathlib
import unittest
from unittest.mock import patch

import numpy as np

from decode import calibrate, decode, viterbi
from synthetic_site import generate_day
from utils import load_config


class CompanionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(pathlib.Path(__file__).with_name("config.yaml"))

    def test_days_are_repeatable_and_separate(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["channels"] = [dict(cfg["channels"][0], name=name) for name in ("a", "b")]
        commissioning = generate_day(cfg["seed"], cfg)
        evaluation = generate_day(cfg["seed"], cfg, day=1)
        repeated = generate_day(cfg["seed"], cfg, day=1)
        for key in evaluation:
            np.testing.assert_array_equal(evaluation[key], repeated[key])
        self.assertFalse(np.array_equal(commissioning["states"][:, 1],
                                        evaluation["states"][:, 0]))

    def test_factorized_matches_exhaustive_search(self):
        rng = np.random.default_rng(3)
        cards = [2, 3]
        states = np.array(list(itertools.product(range(2), range(3))))
        for sparse in (False, True):
            means = [rng.normal(size=c) for c in cards]
            transitions = [np.log(rng.dirichlet(np.ones(c), size=c)) for c in cards]
            if sparse:
                for trans in transitions:
                    trans[np.tril_indices(len(trans), -1)] = -np.inf
            initials = [np.log(rng.dirichlet(np.ones(c))) for c in cards]
            y, sigma = rng.normal(size=3), 0.7
            emission = -0.5 * ((y[:, None] - sum(m[states[:, k]]
                            for k, m in enumerate(means))) / sigma) ** 2
            initial = sum(p[states[:, k]] for k, p in enumerate(initials))
            joint = sum(a[states[:, k, None], states[None, :, k]]
                        for k, a in enumerate(transitions))

            def score(path):
                return initial[path[0]] + sum(emission[t, p] for t, p in enumerate(path)) + sum(
                    joint[a, b] for a, b in zip(path, path[1:]))

            best = max(itertools.product(range(len(states)), repeat=len(y)), key=score)
            actual = viterbi(y, transitions, means, sigma, initials)
            np.testing.assert_array_equal(actual, states[list(best)])

    def test_restriction_preserves_allowed_path_costs(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["channels"] = [{"support_order": [1, 0, 1]}]
        trans = np.log(np.array([[0.8, 0.2], [0.3, 0.7]]))
        model = {"levels": [np.array([0., 2.])], "transitions": [trans],
                 "grid": np.array([0.]), "offset": 0., "sigma": 0.5}
        y = np.array([2., 0., 2., 0., 2.])
        allowed = {tuple([1, 0, 1][p] for p in phases) for phases in
                   itertools.combinations_with_replacement(range(3), len(y))}

        def score(path):
            return -np.log(2) + sum(trans[a, b] for a, b in zip(path, path[1:])) - 0.5 * sum(
                ((y - model["levels"][0][list(path)]) / model["sigma"]) ** 2)

        actual = tuple(decode(y, cfg, model, restricted=True, background="constant")["states"][:, 0])
        self.assertIn(actual, allowed)
        self.assertAlmostEqual(score(actual), max(map(score, allowed)))

    def test_oversized_grid_is_rejected_before_allocation(self):
        model = calibrate(self.cfg)
        model["grid"] = range(100_000)
        with self.assertRaisesRegex(ValueError, "too large"):
            decode(np.zeros(self.cfg["minutes_per_day"]), self.cfg, model)
        day = generate_day(self.cfg["seed"], self.cfg)
        self.cfg["decoder"]["grid_step_w"] = 0.0001
        with patch("decode.generate_day", return_value=day), patch(
                "decode.np.linspace", side_effect=AssertionError("allocated before guard")):
            with self.assertRaisesRegex(ValueError, "too large"):
                calibrate(self.cfg)


if __name__ == "__main__":
    unittest.main()
