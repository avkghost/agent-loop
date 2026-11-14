import unittest

import config
from delivery import PROFILES, get_profile


class ProfileTests(unittest.TestCase):
    def test_the_active_profile_is_defined(self):
        self.assertIn(config.ACTIVE_PROFILE, PROFILES)

    def test_budgets_are_positive_and_the_safe_profile_is_no_larger_than_the_live_one(self):
        for profile in PROFILES.values():
            self.assertGreater(profile.delivery_budget, 0)
            self.assertGreater(profile.reconcile_budget, 0)
        self.assertLessEqual(PROFILES["safe"].delivery_budget, PROFILES["live"].delivery_budget)
        self.assertLessEqual(PROFILES["safe"].reconcile_budget, PROFILES["live"].reconcile_budget)

    def test_an_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            get_profile("missing")
