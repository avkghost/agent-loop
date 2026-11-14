import unittest

from eventledger import FakeStore, StorageError


class StoreTests(unittest.TestCase):
    def test_a_commit_stores_a_copy_of_the_image(self):
        store = FakeStore()
        image = {"balances": {"a": 1}}
        store.commit(image)
        image["balances"]["a"] = 99
        self.assertEqual(store.committed, [{"balances": {"a": 1}}])
        self.assertEqual(store.calls, 1)

    def test_a_failing_commit_stores_nothing_and_raises(self):
        store = FakeStore(fail_commits={1})
        with self.assertRaises(StorageError):
            store.commit({"balances": {}})
        self.assertEqual(store.committed, [])
        store.commit({"balances": {}})
        self.assertEqual(len(store.committed), 1)
