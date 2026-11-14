"""Account state: balances and freeze flags. Balances change only through credit(). snapshot() and restore() let the
engine roll back a unit of work; they copy balances and do not touch freeze flags."""


class AccountFrozen(Exception):
    """The account refuses changes while it is frozen. Nothing is changed when this is raised."""


class AccountState:
    def __init__(self):
        self._balances = {}
        self._frozen = set()

    def freeze(self, account):
        self._frozen.add(account)

    def unfreeze(self, account):
        self._frozen.discard(account)

    def frozen_accounts(self):
        return sorted(self._frozen)

    def require_open(self, account):
        if account in self._frozen:
            raise AccountFrozen(f"account {account} is frozen")

    def balance(self, account):
        return self._balances.get(account, 0)

    def balances(self):
        return dict(self._balances)

    def credit(self, account, amount):
        self._balances[account] = self.balance(account) + amount

    def snapshot(self):
        return dict(self._balances)

    def restore(self, snapshot):
        self._balances = dict(snapshot)
