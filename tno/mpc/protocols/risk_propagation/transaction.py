"""
Configuration of a transaction
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Transaction:
    """
    Class containing a simple transaction
    """

    sender: str  #: Sender of the transaction
    receiver: str  #: Receiver of the transaction
    amount: int  #: Amount of the transaction

    def __str__(self) -> str:
        """
        :return: string representation of transaction's info
        """
        return f"from: {self.sender}, to: {self.receiver}, amount: {self.amount}"

    def __post_init__(self) -> None:
        """
        Ensures that amount is an integer value.

        :raise ValueError: raised when the amount is not an integer value
        """
        if not isinstance(self.amount, int):
            raise ValueError("amount must be int")
