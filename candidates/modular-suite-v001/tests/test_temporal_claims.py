import unittest
from training_runtime.temporal_claims import validate


class TemporalClaimsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(epoch=e, metrics={"FN": v})
                     for e, v in [(0, 0), (5, 1), (10, 0), (15, 0)]]

    def claim(self, start, epochs, relation):
        return dict(metric="FN", start_epoch=start, end_epoch=15,
                    observed_epochs=epochs, relation=relation)

    def test_equal_endpoints_do_not_mean_constant_history(self):
        with self.assertRaisesRegex(ValueError, "False scoped"):
            validate(self.rows, [self.claim(0, [0, 5, 10, 15], "constant")])
        validate(self.rows, [self.claim(0, [0, 5, 10, 15], "variable")])
        validate(self.rows, [self.claim(0, [0, 5, 10, 15], "unchanged_endpoints")])
        validate(self.rows, [self.claim(10, [10, 15], "constant")])

    def test_omitted_middle_or_unmeasured_endpoint_rejected(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate(self.rows, [self.claim(0, [0, 10, 15], "constant")])
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate(self.rows, [self.claim(11, [15], "constant")])

    def test_nonfinite_rejected(self):
        self.rows[1]["metrics"]["FN"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            validate(self.rows, [self.claim(0, [0, 5, 10, 15], "variable")])
