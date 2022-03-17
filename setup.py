import setuptools
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text()


setuptools.setup(
    name="hnne",
    version="0.1.5",
    author="Marios Koulakis, Saquib Sarfraz",
    author_email="marios.koulakis@gmail.com",
    description="A fast hierarchical dimensionality reduction algorithm.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=['hnne'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: Free for non-commercial use",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="dimension dimensionality reduction t-sne umap hierarchical clustering finch",
    install_requires=[
        "scipy",
        "numpy==1.20",
        "sklearn",
        "tqdm",
        "pynndescent",
        "typer",
        "pandas",
        "cython",
    ],
    test_suite="pytest",
    tests_require=["pytest"],
    project_urls={
        "Documentation": "https://hnne.readthedocs.io/en/latest/",
        "Repository": "https://github.com/koulakis/h-nne"
    }
)
