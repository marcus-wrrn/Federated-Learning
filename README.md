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

Make sure you are in the `./src/core_model/` directory.

Run the command

`python -m build`

Install the generated package:

pip install ./dist/flcore-0.0.2-py3-none-any.whl 

    Note: The core model logic is packaged as a standalone module for modularity and reusability. 
    After installation you should be able to use flcore as a standalone python package. 
python main.py


### Run Coordination Server

make sure you are in the `./src/coordination` directory in the terminal. 

Run 

`flask --app server run --debug`

To start the server in development mode