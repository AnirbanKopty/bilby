#!/usr/bin/env python

from setuptools import setup
import sys
import os

python_version = sys.version_info
if python_version < (3, 9):
    sys.exit("Python < 3.9 is not supported, aborting setup")


def get_long_description():
    """Finds the README and reads in the description"""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.rst")) as f:
        long_description = f.read()
    return long_description


def get_requirements(kind=None):
    if kind is None:
        fname = "requirements.txt"
    else:
        fname = f"{kind}_requirements.txt"
    with open(fname, "r") as ff:
        requirements = ff.readlines()
    return requirements


# get version info from __init__.py
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()
    return filecontents


long_description = get_long_description()

setup(
    name="bilby",
    description="A user-friendly Bayesian inference library",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://git.ligo.org/lscsoft/bilby",
    author="Greg Ashton, Moritz Huebner, Paul Lasky, Colm Talbot",
    author_email="paul.lasky@monash.edu",
    license="MIT",
    packages=[
        "bilby",
        "bilby.bilby_mcmc",
        "bilby.core",
        "bilby.core.prior",
        "bilby.core.sampler",
        "bilby.core.utils",
        "bilby.gw",
        "bilby.gw.detector",
        "bilby.gw.eos",
        "bilby.gw.likelihood",
        "bilby.gw.sampler",
        "bilby.hyper",
        "cli_bilby",
    ],
    package_dir={"bilby": "bilby", "cli_bilby": "cli_bilby"},
    package_data={
        "bilby.gw": ["prior_files/*"],
        "bilby.gw.detector": ["noise_curves/*.txt", "detectors/*"],
        "bilby.gw.eos": ["eos_tables/*.dat"],
    },
    python_requires=">=3.9",
    install_requires=get_requirements(),
    extras_require={
        "gw": get_requirements("gw"),
        "mcmc": get_requirements("mcmc"),
        "all": (
            get_requirements("sampler")
            + get_requirements("gw")
            + get_requirements("mcmc")
            + get_requirements("optional")
        ),
    },
    entry_points={
        "console_scripts": [
            "bilby_plot=cli_bilby.plot_multiple_posteriors:main",
            "bilby_result=cli_bilby.bilby_result:main",
        ],
        "bilby.samplers": [
            "bilby_mcmc=bilby.bilby_mcmc.sampler:Bilby_MCMC",
            "cpnest=bilby.core.sampler.cpnest:Cpnest",
            "dnest4=bilby.core.sampler.dnest4:DNest4",
            "dynesty=bilby.core.sampler.dynesty:Dynesty",
            "dynamic_dynesty=bilby.core.sampler.dynamic_dynesty:DynamicDynesty",
            "emcee=bilby.core.sampler.emcee:Emcee",
            "kombine=bilby.core.sampler.kombine:Kombine",
            "nessai=bilby.core.sampler.nessai:Nessai",
            "nestle=bilby.core.sampler.nestle:Nestle",
            "ptemcee=bilby.core.sampler.ptemcee:Ptemcee",
            "ptmcmcsampler=bilby.core.sampler.ptmcmc:PTMCMCSampler",
            "pymc=bilby.core.sampler.pymc:Pymc",
            "pymultinest=bilby.core.sampler.pymultinest:Pymultinest",
            "pypolychord=bilby.core.sampler.polychord:PyPolyChord",
            "ultranest=bilby.core.sampler.ultranest:Ultranest",
            "zeus=bilby.core.sampler.zeus:Zeus",
            "fake_sampler=bilby.core.sampler.fake_sampler:FakeSampler",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
