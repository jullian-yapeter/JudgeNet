from distutils.core import setup

setup(
    name="judgenet",
    version="0.0.1",
    packages=["judgenet"],
    install_requires=[
        "numpy",
        "torch",
        "tqdm",
        "transformers"
    ]
)
