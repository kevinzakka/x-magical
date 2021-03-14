import os

from setuptools import find_packages, setup

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DESCRIPTION = (
    "X-MAGICAL is a benchmark suite for cross-embodiment visual imitation."
)
TESTS_REQUIRE = [  # Adapted from https://github.com/HumanCompatibleAI/imitation/blob/master/setup.py
    "black",
    "flake8",
    "flake8-blind-except",
    "flake8-builtins",
    "flake8-debugger",
    "flake8-isort",
    "pytest",
    "pytest-notebook",
    "pytest-xdist",
    "pytype",
]


def readme():
    """Load README for use as package's long description."""
    with open(os.path.join(THIS_DIR, "README.md"), "r") as fp:
        return fp.read()


def get_version():
    locals_dict = {}
    with open(os.path.join(THIS_DIR, "xmagical", "version.py"), "r") as fp:
        exec(fp.read(), globals(), locals_dict)
    return locals_dict["__version__"]


setup(
    name="x-magical",
    version=get_version(),
    author="Kevin Zakka",
    license="ISC",
    description=DESCRIPTION,
    python_requires=">=3.8",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "gym==0.17.*",
        "numpy>=1.17.4",
        "pygame>=2.0.0",
        "pyglet==1.5.*",
        "pymunk~=5.6.0",
    ],
    tests_require=TESTS_REQUIRE,
    url="https://github.com/kevinzakka/x-magical/",
)
