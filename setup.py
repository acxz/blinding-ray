import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="blinding-ray",
    version="0.0.1",
    author="acxz",
    description="blinding-ray description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/acxz/blinding-ray",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        'chess',
        'open-spiel',
        'ray[rllib]',
        'reconchess',
    ],
)
