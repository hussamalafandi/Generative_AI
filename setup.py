from setuptools import setup, find_packages

setup(
    name="genai_core",
    version="0.1.0",
    description="A modular generative AI training framework with PyTorch",
    author="Hussam Alafandi",
    author_email="hosam.alafandi@gmail.com",
    packages=find_packages(exclude=["outputs*", "notebooks*", "scripts*"]),
    include_package_data=True,
    install_requires=[
        "torch>=1.10",
        "torchvision",
        "tqdm",
        "wandb",
        "pyyaml",
        "numpy",
        "scipy",
        "matplotlib",
        "torchmetrics",
    ],
    python_requires=">=3.8",
)
