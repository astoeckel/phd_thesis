# Harnessing Neural Dynamics as a Computational Resource
### PhD Thesis, Andreas Stöckel, 2021, University of Waterloo

## Abstract

Researchers study nervous systems at levels of scale spanning several orders of magnitude, both in terms of time and space.
While some parts of the brain are well understood at specific levels of description, there are few overarching theories that systematically bridge low-level mechanism and high-level function.
The Neural Engineering Framework (NEF) is an attempt at providing such a theory.
The NEF enables researchers to systematically map dynamical systems‒corresponding to some hypothesised brain function‒onto biologically constrained spiking neural networks.
In this thesis, we present several extensions to the NEF that broaden both the range of neural resources that can be harnessed for spatiotemporal computation and the range of available biological constraints.
Specifically, we suggest a method for harnessing the dynamics inherent in passive dendritic trees for computation, allowing us to construct single-layer spiking neural networks that, for some functions, achieve substantially lower errors than larger multi-layer networks.
Furthermore, we suggest “temporal tuning” as a unifying approach to harnessing temporal resources for computation through time.
This allows modellers to directly constrain networks to temporal tuning observed in nature, in ways not previously well-supported by the NEF.

We then explore specific examples of neurally plausible dynamics using these techniques.
In particular, we propose a new “information erasure” technique for constructing LTI systems generating temporal bases.
Such LTI systems can be used to establish an optimal basis for spatiotemporal computation.
We demonstrate how this captures “time cells” that have been observed throughout the brain.
As well, we demonstrate the viability of our extensions by constructing an adaptive filter model of the cerebellum that successfully reproduces key features of eyeblink conditioning observed in neurobiological experiments.

Outside the cognitive sciences, our work can help exploit resources available on existing neuromorphic computers, and inform future neuromorphic hardware design.
In machine learning, our spatiotemporal NEF populations map cleanly onto the Legendre Memory Unit (LMU), a promising artificial neural network architecture for stream-to-stream processing that outperforms competing approaches.
We find that one of our LTI systems derived through “information erasure” may serve as a computationally less expensive alternative to the LTI system commonly used in the LMU.

## Compiling the document

Make sure to have the following installed:

* `fuse-overlayfs`
* A fully copy of TeX Live 2020
* The following `pip` packages:
  * `pygments`
  * `numpy==1.19.4`
  * `scipy==1.7.1`
  * `sympy==1.7.1`
  * `matplotlib==3.3.4`
  * `h5py==3.1.0`
  * `tqdm==4.59.0`
  * `nengo==3.1.0`
  * `nengo-extras==0.4.0`
  * `cython==0.29.22`
  * `brian2==2.4.2`
  * `scikit-learn==0.24.2`
  * `autograd==1.3`
  * `cvxopt==1.2.6`
* Revision `7a788b59f` of `libbioneuronqp` from https://github.com/astoeckel/libbioneuronqp
* Revision `04e992684` of `nengo-bio` from https://github.com/astoeckel/nengo-bio
* Our `srinivasa` colour scheme from `contrib/srinivasa` (install via `pip install -e .`)

We will provide an operating system image with all prerequisites installed at a future point in time.

Once all prerequisites are fulfilled, simply execute `./build.sh`.

## Pre-built datafiles and docker containers

In case you prefer not to re-run all experiments, an archive with all pre-built data files, figures, and docker container images may be found at

https://osf.io/y64xu/files/


## Running the experiments

All longer-runing experiments are executed within a Docker container and linked to a specific revision of this experiment.
The experiments and generated files are listed under `code/Manifest.toml`.

Executing the experiments requires Linux with Python and a working Docker installation, as well as a computer with at least 32 CPU cores and 128 GB of RAM.

Simply run the
```
./scripts/run_experiments.py
```
script to re-run all experiments. This will take several weeks.

Use the `--help` argument to obtain list of available commands; use `--list` to list all experiments and their build status.

