from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name = "ML-ops",
    version= "0.1",
    author="sameeh",
    packages=find_packages(),
    install_requires = requirements,
)
