from setuptools import setup, find_packages

setup(
    name="federated_learning_core_model",
    version="0.0.1",
    description="Core Model functionality for the project",
    author="Marcus Warren",
    author_email="marcuswrrn@gmail.com",
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[
        "torch==2.5.0",
        "torchvision==0.20.0",
        "matplotlib>=3.9",
        "pandas"
    ],
)