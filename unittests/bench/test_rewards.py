import unittest

from locodellm.bench import export_failure_reward


class TestExportFailureReward(unittest.TestCase):
    def test_partial_credit_is_additive_without_gating(self):
        """Preserves partial credit even when an earlier criterion fails."""
        for criteria, expected in [
            ((False, False, False, False), 0.0),
            ((True, True, True, True), 1.0),
            ((True, True, False, False), 0.5),
            ((False, True, True, True), 0.75),
            ((False, False, False, True), 0.25),
        ]:
            with self.subTest(criteria=criteria):
                self.assertEqual(
                    export_failure_reward(
                        reference_valid=criteria[0],
                        export_call_valid=criteria[1],
                        failure_valid=criteria[2],
                        reproduction_valid=criteria[3],
                    ),
                    expected,
                )

    def test_unverified_values_are_not_coerced_to_success(self):
        """Rejects strings, numbers and missing judgments instead of scoring truthiness."""
        criteria = dict.fromkeys(
            (
                "reference_valid",
                "export_call_valid",
                "failure_valid",
                "reproduction_valid",
                "answer_extractable",
                "example_created",
                "test_created",
                "test_passed",
                "failure_confirmed",
            ),
            False,
        )
        for name in criteria:
            for value in ("false", 1, 0, None):
                with (
                    self.subTest(criterion=name, value=value),
                    self.assertRaisesRegex(TypeError, f"{name} must be a boolean"),
                ):
                    export_failure_reward(**{**criteria, name: value})

    def test_extractable_answer_credit_is_a_floor_not_a_bonus(self):
        """Credits an unverified example without inflating validated scores."""
        for criteria, expected in [
            ((False, False, False, False), 0.05),
            ((True, False, False, False), 0.25),
            ((True, True, True, False), 0.75),
            ((True, True, True, True), 1.0),
        ]:
            with self.subTest(criteria=criteria):
                self.assertEqual(
                    export_failure_reward(
                        reference_valid=criteria[0],
                        export_call_valid=criteria[1],
                        failure_valid=criteria[2],
                        reproduction_valid=criteria[3],
                        answer_extractable=True,
                    ),
                    expected,
                )

    def test_progress_bonuses_add_to_the_original_score_without_normalization(self):
        """Preserves the original reward and adds each milestone independently."""
        for original, progress, expected in [
            ((False, False, False, False), (True, False, False, False), 0.30),
            ((True, True, False, False), (True, True, True, False), 1.25),
            ((True, True, True, True), (True, True, True, True), 2.0),
            ((False, False, False, False), (False, False, False, True), 0.30),
        ]:
            with self.subTest(original=original, progress=progress):
                self.assertAlmostEqual(
                    export_failure_reward(
                        reference_valid=original[0],
                        export_call_valid=original[1],
                        failure_valid=original[2],
                        reproduction_valid=original[3],
                        answer_extractable=True,
                        example_created=progress[0],
                        test_created=progress[1],
                        test_passed=progress[2],
                        failure_confirmed=progress[3],
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
