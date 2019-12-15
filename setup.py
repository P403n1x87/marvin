from setuptools import setup, find_packages
from os import path

from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="marvin",
    version="0.1.0",
    description="A collection of machine learning utilities for Python 3",
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/p403n1x87/marvin",  # Optional
    # This should be your name or the name of the organization which owns the
    # project.
    author="Gabriele N. Tornetta",  # Optional
    # This should be a valid email address corresponding to the author listed
    # above.
    author_email="phoenix1987@gmail.com",  # Optional
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # These classifiers are *not* checked by 'pip install'. See instead
        # 'python_requires' below.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="machine-learning sklearn",  # Optional
    packages=find_packages(),  # Required
    python_requires=">=3.6",
    install_requires=["sklearn"],  # Optional
    extras_require={"dev": ["check-manifest"], "test": ["pytest", "coverage"],},  # Optional
    project_urls={  # Optional
        "Bug Reports": "https://github.com/p403n1x87/marvin/issues",
        # "Funding": "https://donate.pypi.org",
        # "Say Thanks!": "http://saythanks.io/to/example",
        "Source": "https://github.com/p403n1x87/marvin/",
    },
)
