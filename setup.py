# Modify from https://github.com/deepmind/dm-haiku/blob/main/setup.py
from setuptools import find_namespace_packages
from setuptools import setup


def _get_version():
    with open("torchrng/__init__.py") as fp:
        for line in fp:
            if line.startswith("__version__"):
                g = {}
                exec(line, g)  # pylint: disable=exec-used
                return g["__version__"]
        raise ValueError("`__version__` not defined in `torchrng/__init__.py`")


def _parse_requirements(requirements_txt_path):
    with open(requirements_txt_path) as fp:
        return fp.read().splitlines()


_VERSION = _get_version()

setup(
    name="torchrng",
    version=_VERSION,
    url="https://github.com/h-terao/torchrng",
    license="Apache 2.0",
    author="h-terao",
    description="JAX-like PRNG API for PyTorch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # Contained modules and scripts.
    packages=find_namespace_packages(exclude=["*_test.py", "examples"]),
    install_requires=_parse_requirements("requirements.txt"),
    requires_python=">=3.6",
    include_package_data=True,
    zip_safe=False,
)
