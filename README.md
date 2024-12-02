Follow these steps to set up and install the project:
1. Create a Virtual Environment

To keep dependencies isolated, create a virtual environment:

python3 -m venv .venv

2. Activate the Virtual Environment

Activate the virtual environment based on your operating system:

    Linux/Mac:

source .venv/bin/activate

Windows (cmd):

.venv\Scripts\activate

Windows (PowerShell):

    .venv\Scripts\Activate.ps1

3. Install Required Packages

Install the dependencies listed in requirements.txt:

pip install -r requirements.txt

4. Build and Install the Core Model Logic

Build the core model logic into a distributable package:

python -m build

Install the generated package:

pip install dist/fl_core-0.0.1-py3-none-any.whl

    Note: The core model logic is packaged as a standalone module for modularity and reusability.

Usage

Add a short section on how to use your project or core logic, with an example if applicable:

python main.py
