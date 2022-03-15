import setuptools


setuptools.setup(
    name="hnne",
    version="0.1.0",
    author="Marios Koulakis, Saquib Sarfraz",
    author_email="marios.koulakis@gmail.com",
    description="A fast hierarchical dimensionality reduction algorithm.",
    packages=['hnne'],
    classifiers=[
        "Development Status :: 3 - Alpha",
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
)
