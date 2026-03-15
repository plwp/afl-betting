import unittest

import pandas as pd

from squiggle import _build_enhanced_round_features, _normalize_round_id


class SquiggleFeatureTests(unittest.TestCase):
    def test_normalize_round_id_handles_regular_and_finals_rounds(self):
        self.assertEqual(_normalize_round_id("03"), "3")
        self.assertEqual(_normalize_round_id(7), "7")
        self.assertEqual(_normalize_round_id("Elimination Final"), "25")
        self.assertEqual(_normalize_round_id("Grand Final"), "28")

    def test_enhanced_features_use_only_prior_rounds_for_top_models(self):
        df = pd.DataFrame([
            {
                "year": 2024, "round_num": "1", "home_team": "A", "away_team": "B",
                "source": "ModelA", "prob": 0.8, "correct": 1, "date": "2024-03-10",
            },
            {
                "year": 2024, "round_num": "1", "home_team": "A", "away_team": "B",
                "source": "ModelB", "prob": 0.2, "correct": 0, "date": "2024-03-10",
            },
            {
                "year": 2024, "round_num": "2", "home_team": "C", "away_team": "D",
                "source": "ModelA", "prob": 0.1, "correct": 0, "date": "2024-03-17",
            },
            {
                "year": 2024, "round_num": "2", "home_team": "C", "away_team": "D",
                "source": "ModelB", "prob": 0.9, "correct": 1, "date": "2024-03-17",
            },
        ])

        result = _build_enhanced_round_features(df, top_n=1)

        round_one = result[result["round_num"] == "1"].iloc[0]
        round_two = result[result["round_num"] == "2"].iloc[0]

        # No prior history in round 1, so use the round consensus.
        self.assertAlmostEqual(round_one["squiggle_top3_prob"], 0.5)
        # ModelA was best after round 1, so round 2 must use its probability only.
        self.assertAlmostEqual(round_two["squiggle_top3_prob"], 0.1)
        self.assertAlmostEqual(round_two["squiggle_model_spread"], 0.5656854, places=6)


if __name__ == "__main__":
    unittest.main()
