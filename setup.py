from pathlib import Path

import setuptools

this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text()

test_dependencies = ["pytest", "unittest", "torchvision"]


setuptools.setup(
    name="hnne",
    version="0.1.10",
    author="Marios Koulakis, Saquib Sarfraz",
    author_email="marios.koulakis@gmail.com, saquibsarfraz@gmail.com",
    description="A fast hierarchical dimensionality reduction algorithm.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=["hnne"],
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
        "numba>=0.51.2",
        "pynndescent",
        "scipy",
        "numpy>=1.18",
        "scikit-learn",
        "tqdm",
        "typer",
        "pandas",
        "cython",
    ],
    test_suite="pytest",
    tests_require=test_dependencies,
    extras_require={"test": test_dependencies},
    project_urls={
        "Documentation": "https://hnne.readthedocs.io/en/latest",
        "Repository": "https://github.com/koulakis/h-nne",
        "Publication": "https://arxiv.org/abs/2203.12997",
    },
)
