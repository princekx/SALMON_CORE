from setuptools import setup, find_packages

setup(
    name="salmon_core",
    version="2.0.0",
    description="SALMON_CORE: Southeast Asia Large Scale Monitoring tool",
    author="Prince Xavier",
    packages=find_packages(),
    install_requires=[
        "scitools-iris",  # IMPORTANT: use 'scitools-iris', NOT bare 'iris' (that is illumon-iris)
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
