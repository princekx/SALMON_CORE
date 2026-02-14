from setuptools import setup, find_packages

setup(
    name="salmon",
    version="2.0.0",
    description="SALMON v2: Southeast Asia Large Scale Monitoring tool",
    author="Prince Xavier",
    packages=find_packages(),
    install_requires=[
        "iris",
        "numpy",
        "pyyaml",
        "click",
        "pandas",
        "pydantic",
        "scipy",
        "scikit-image",
        "tqdm"
    ],
    entry_points={
        'console_scripts': [
            'salmon=salmon.cli:main',
        ],
    },
)
