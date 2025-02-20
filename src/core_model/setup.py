from setuptools import setup, find_packages

setup(
    name="flcore",
    version="0.0.3",
    description="Core Model functionality for the project",
    author="Marcus Warren",
    author_email="marcuswrrn@gmail.com",
    packages=find_packages(),  # Automatically finds all packages in the src directory
    install_requires=[
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "matplotlib>=3.9",
        "pandas"
    ],
)