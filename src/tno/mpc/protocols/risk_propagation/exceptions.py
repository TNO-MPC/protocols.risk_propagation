"""
Collection of custom exceptions.
"""


class IncorrectStartException(Exception):
    """
    Raised when the protocol is started in an incorrect state.
    """


class WrongDeserializationException(Exception):
    """
    Raised when deserialization is done incorrectly.
    """
