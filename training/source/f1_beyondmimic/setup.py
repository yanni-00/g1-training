from setuptools import find_packages, setup

setup(
    name="f1-beyondmimic", version="0.1.0", packages=find_packages(),
    python_requires=">=3.10", install_requires=["numpy", "torch"],
    description="Isolated F1 BeyondMimic task adapted from the project's G1 route",
)
