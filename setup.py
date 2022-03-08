import setuptools


setuptools.setup(
    name="hnne",
    version="0.0.1",
    author="Marios Koulakis, Saquib Sarfraz",
    author_email="marios.koulakis@gmail.com",
    description="A fast hierarchical dimensionality reduction algorithm.",
    packages=['hnne'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent - Tested on Ubuntu 18.04",
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
)
