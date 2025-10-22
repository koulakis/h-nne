from pathlib import Path

import setuptools

this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text()

test_dependencies = ["pytest", "torchvision", "pre_commit"]


setuptools.setup(
    name="hnne",
    version="2.0.0",
    author="Marios Koulakis, Saquib Sarfraz",
    author_email="marios.koulakis@gmail.com, saquibsarfraz@gmail.com",
    description="A fast hierarchical dimensionality reduction algorithm.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=["hnne"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
        "finch-clust >= 0.2.2",
    ],
    test_suite="pytest",
    tests_require=test_dependencies,
    extras_require={"test": test_dependencies, "dev": ["hnne-benchmarking"]},
    project_urls={
        "Documentation": "https://hnne.readthedocs.io/en/latest",
        "Repository": "https://github.com/koulakis/h-nne",
        "Publication": "https://openaccess.thecvf.com/content/CVPR2022/papers/Sarfraz_Hierarchical_Nearest_Neighbor_Graph_Embedding_for_Efficient_Dimensionality_Reduction_CVPR_2022_paper.pdf",  # noqa: E501
    },
)
