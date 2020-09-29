import setuptools

setuptools.setup(
    name="equiv_dens",
    version="0.1",
    author="Mihail Bogojeski",
    author_email="m.bogojeski@tu-berlin.de",
    description="A package for density functional approximation using machine learning",
    long_description="A package for density functional approximation using machine learning",
    url="https://github.com/MihailBogojeski/equiv_dens_ml",
    package_dir={"": "src"},
    packages=setuptools.find_packages('src'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
