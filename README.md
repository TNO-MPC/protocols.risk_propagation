# TNO PET Lab - secure Multi-Party Computation (MPC) - Protocols - Risk Propagation

Secure Risk Propagation initially developed within the MPC4AML project.

### PET Lab

The TNO PET Lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of PET solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed PET functionalities to boost the development of new protocols and solutions.

The package `tno.mpc.protocols.risk_propagation` is part of the [TNO Python Toolbox](https://github.com/TNO-PET).

_Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws._  
_This implementation of cryptographic software has not been audited. Use at your own risk._

## Documentation

Documentation of the `tno.mpc.protocols.risk_propagation` package can be found
[here](https://docs.pet.tno.nl/mpc/protocols/risk_propagation/2.3.4).

## Install

Easily install the `tno.mpc.protocols.risk_propagation` package using `pip`:

```console
$ python -m pip install tno.mpc.protocols.risk_propagation
```

_Note:_ If you are cloning the repository and wish to edit the source code, be
sure to install the package in editable mode:

```console
$ python -m pip install -e 'tno.mpc.protocols.risk_propagation'
```

If you wish to run the tests you can use:

```console
$ python -m pip install 'tno.mpc.protocols.risk_propagation[tests]'
```

_Note:_ A significant performance improvement can be achieved by installing the GMPY2 library.

```console
$ python -m pip install 'tno.mpc.protocols.risk_propagation[gmpy]'
```

## Protocol description

Risk propagation is an algorithm that propagates risk scores through a (transaction) network.
Using distributed partial homomorphic encryption, the secure risk propagation implementation performs this algorithm on a network that is distributed among multiple parties.
As input of the algorithm, every party possesses a list of nodes (i.e. bank accounts) with initial risk scores and has a list of transactions (weighted, directed edges) from and to its bank accounts.
Using encrypted incoming risk scores scores from other parties, every party can securely update its risk scores using the formula for risk propagation.
After a set number of iterations, the eventual risk scores are revealed to the parties that own the accounts, using the distributed private key.

In [ERCIM News 126 (July 2021)](https://ercim-news.ercim.eu/en126/special/privacy-preserving-collaborative-money-laundering-detection), we presented a more elaborate protocol descriptions. Figure 1 demonstrates a high-level overview of the idea behind the protocol. Figure 2 visualizes the decentralized approach.

<figure>
  <img src="assets/risk-propagation-overview.svg" width=100% alt="Risk Propagation High Level Overview"/>
  <figcaption>

**Figure 1.** _We consider a three-bank scenario (Orange, Blue, and Purple). In this scenario the first (left) account at bank Orange is classified as high risk (due to e.g., large cash deposits) by bank Orange.
This account wishes to launder its resources. To stay under the radar, the resources are funnelled through multiple accounts, at various banks, before arriving at their eventual destination, e.g., the account at bank Purple (right).
To detect money laundering, we wish to follow (propagate) the risky money and classify the endpoint as high risk too. Full (global) knowledge of the network enables us to propagate the risk.
However, how can we achieve something similar when there is only partial (local) knowledge of the entire network available? This is where MPC comes into play._

  </figcaption>
</figure>

<figure>
  <img src="assets/approach.svg" width=100% alt="Risk Propagation Decentralized Approach"/>
  <figcaption>

**Figure 2.** _In our approach, the data is analyzed in a decentralized manner. From left-to-right, we visualize encryption,
propagation and decryption. The parties encrypt their data using the additive homomorphic encryption scheme, no
communication takes place. Once the data is encrypted locally, the distributed propagation (computation) over the
encrypted data takes place. During this computation the data remains encrypted, the parties communicate intermediate
(encrypted) results, and there is no central party. During the decryption phase, we need to decrypt the encrypted values
with every party's key. We view the decryption from the green party's perspective. The lock on the risk scores belonging to
the green party is opened part-by-part and the final opening happens by the green party. This ensures that only the green
party gets to see the decrypted, propagated risk scores of his own bank accounts._

  </figcaption>
</figure>

## Usage

The protocol is symmetric. For determining a secure set of parameters for the distributed keygen we refer to [protocols.distributed_keygen](https://github.com/TNO-MPC/protocols.distributed_keygen/#usage).

### Input

For the input two numpy arrays are expected. The structure and types are given in the tables below:

Input of the accounts:

| id     | risk_score |
| ------ | ---------- |
| string | float 64   |

Input of the transactions:

| id_source | bank_source | id_destination | bank_destination | amount |
| --------- | ----------- | -------------- | ---------------- | ------ |
| string    | string      | string         | string           | int32  |

A string is expected to consist out of at most 100 unicode characters.

For an example of how to setup and run the protocol, see `scripts/example_usage.py`.
