import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="concise-analysis",
    version="0.0.1",
    author="Krzysztof Piesiewicz",
    author_email="krz.piesiewicz@gmail.com",
    description="A package which provides tools for quick data analysis. Its main aim is to make a user write less code. It contains functions for data visualization, feature selection, and evaluation of ML models. It is build on top of: numpy, pandas, scikit-learn, matplotlib, and plotly.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krzpiesiewicz/concise-analysis",
    packages=setuptools.find_packages(exclude=['tests', 'examples']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas>=1.0.5",
        "numpy>=1.19.0",
        "scikit-learn>=0.22.0",
        "matplotlib>=3.2.2",
        "IPython>=5.5.0",
        "plotly>=4.5.0",
    ],
    test_requirements=["pytest>=6.2.0"],
    python_requires='>=3.6',
)
