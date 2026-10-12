# CovBIM

## Installation and Testing

### Prerequisites

- Git
- Python 3.9 or newer
- A C++ compiler and CMake
- SSH access to the GitHub repository

### Clone the repository

```bash
git clone git@github.com:EmilienGanier/CovBIM.git
cd CovBIM
```

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install the project and run tests for safety

```bash
pip install -e .
pytest
```

The editable installation builds and installs the project into the active virtual environment. Pytest then runs the project's tests.
