# This file nedds to exist if we want to be able to pip install it & upload the package to pypi

from setuptools import setup, find_packages


# Function to read requirements from requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as file:
        requirements = file.read().splitlines()
    return requirements


# Read the requirements from requirements.txt
requirements = parse_requirements("requirements.txt")

setup(
    name="evo_pipetting_optimiser",
    version="0.0.1a3",
    packages=find_packages(
        include=["evo_pipetting_optimiser", "evo_pipetting_optimiser.*"]
    ),
    description="Package to extend robotools to optimise evo pipetting commands, grouping transfers while keeping transfer order the same",
    author="Ivy Brain",
    author_email="ivy.brain@csl.com.au",
    url="https://github.com/CSL-R-D/evo_pipetting_optimiser",
    install_requires=requirements,
)
