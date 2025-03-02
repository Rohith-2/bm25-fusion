from setuptools import setup, find_packages

setup(
    name="bm25_fusion",
    version="0.1.4",
    author="Rohith Ramakrishnan",
    author_email="rrohith2001@gmail.com",
    description="An ultra-fast BM25 retriever with support for multiple variants, metadata filtering, and stopword removal.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Rohith-2/bm25-fusion",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "numba",
        "nltk"
    ],
)