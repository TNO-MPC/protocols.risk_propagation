import os
from pathlib import Path

import pytest

from tno.mpc.protocols.distributed_keygen import DistributedPaillier


# Comment out the skip to generate new key files to use for testing.
@pytest.mark.skip(reason="No need to generate key files for each run")
def test_store_key_to_file(
    distributed_schemes_fresh: tuple[DistributedPaillier, ...],
) -> None:
    """
    Test which generates different keys and store them to the file system. These files are also included in the package.

    :param distributed_schemes_fresh: The schemes to store to the file system.
    """
    base_path = Path(f"{os.path.dirname(__file__)}/test_data/keys")
    for index, key in enumerate(distributed_schemes_fresh):
        with open(
            base_path.joinpath(
                f"distributed_key_threshold_{key.corruption_threshold}_{len(distributed_schemes_fresh)}parties_{index}.obj"
            ),
            "wb",
        ) as file:
            file.write(DistributedPaillier.store_private_key(key))
