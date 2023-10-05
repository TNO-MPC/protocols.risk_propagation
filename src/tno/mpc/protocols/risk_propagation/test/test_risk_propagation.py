"""
Module containing tests that can be ran using pytest to test the risk propagation functionality
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from tno.mpc.communication.httphandlers import logger
from tno.mpc.encryption_schemes.utils import FixedPoint

from tno.mpc.protocols.risk_propagation import Player

fxp = FixedPoint.fxp
logger.setLevel(logging.WARNING)


@pytest.mark.asyncio
async def test_risk_propagation(
    alice: Player, bob: Player, charlie: Player, iterations: int = 3
) -> None:
    """
    Tests the running of the full protocol

    :param alice: First player
    :param bob: Second player
    :param charlie: Third player
    :param iterations: Number of iterations
    """
    await asyncio.gather(
        *[
            alice.run_protocol(iterations),
            bob.run_protocol(iterations),
            charlie.run_protocol(iterations),
        ]
    )
    correct_outcome = {
        "alice": {"a": fxp(0.35), "b": fxp(0.1125), "c": fxp(0.275)},
        "bob": {"d": fxp(0.0875), "e": fxp(0.1), "f": fxp(0.075)},
        "charlie": {"g": fxp(0.3), "h": fxp(0.3), "i": fxp(0.125)},
    }
    assert alice.risk_scores == correct_outcome["alice"]
    assert bob.risk_scores == correct_outcome["bob"]
    assert charlie.risk_scores == correct_outcome["charlie"]
