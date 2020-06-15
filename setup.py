import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "PyOptDE", # Replace with your own username
    version = "1.0",
    author = "Lucas Resende",
    author_email = "lucasresenderc@gmail.com",
    description = "An implementation of the Differential Evolution algorithm.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/lucasresenderc/PyOptDE",
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6',
    install_requires = [
        "numpy",
        "matplotlib",
        "scipy",
        "numpy"
    ]
)