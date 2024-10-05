# Composite Laminate Toolbox

This repository contains a Python toolbox for modeling composite laminate materials, designed to handle various operations such as stress-strain calculations, laminate theory, and failure criteria for composite laminae and laminates.

## Directory Structure
toolbox/
├── __init__.py       # Initialization file for the package
├── lamina.py         # Defines the Lamina class for individual ply properties
├── laminate.py       # Defines the Laminate class for layup calculations
├── material.py       # Defines the Material class for material properties
└── utils.py          # Utility functions for tensor and vector operations
.gitignore             # Files to be ignored by Git
requirements.txt       # List of Python dependencies


## Features

- **Material Modeling**: 
  - Define anisotropic materials with different moduli and thermal expansion coefficients.

- **Lamina Modeling**:
  - Compute stiffness matrices (Q, Q̅) and perform transformations for rotated laminae.
  - Calculate stresses, strains, and thermal effects on individual laminae.
  
- **Laminate Modeling**:
  - Perform Classical Laminate Theory (CLT) calculations, including computation of [A], [B], [D] matrices.
  - Convert deformations to forces/moments and vice-versa for multilayer laminates.

- **Failure Criteria**:
  - Max stress, Max strain, Tsai-Hill, Hoffman, and Tsai-Wu failure criteria for assessing laminate integrity.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/FelipeMAssis/ToolboxCompositos.git
```

2. Navigate into the directory:

```bash
cd ToolboxCompositos
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The toolbox is organized into separate modules for materials, laminae, laminates, and utility functions. Below is an example of how to use the `Material`, `Lamina`, and `Laminate` classes:

```python
from toolbox.material import Material
from toolbox.lamina import Lamina
from toolbox.laminate import Laminate

# Define material properties (E11, E22, G12, v12)
material = Material(E11=140000, E22=10000, G12=5000, v12=0.3)

# Define lamina properties (material, thickness, angle)
lamina = Lamina(material, t=0.125, theta=45)

# Define laminate layup
layup = [lamina, lamina]  # A simple two-ply layup
laminate = Laminate(layup)

# Compute laminate stiffness matrices
A, B, D = laminate.ABD()
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Contact

For any questions or feedback, please open an issue or contact the repository owner.