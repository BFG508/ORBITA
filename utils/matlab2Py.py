"""
Script: matlab2Py.py

Description:
    This utility script translates the auto-generated symbolic equations 
    from MATLAB (.m) into a fully functional and optimized Python (.py) module.
    It uses regular expressions to adapt the syntax, operators, and indexing 
    while preserving the mathematical integrity of the original formulas.
"""

import re
import os

def translateESTHER2ORBITA(input_file, output_file):
    """
    Reads a MATLAB file containing the analytical solution and translates 
    it into a Python-compatible script using NumPy.
    
    Args:
        input_file  (str): Path to the original MATLAB (.m) file.
        output_file (str): Path where the translated Python (.py) file will be saved.
    """
    print(f"Reading the massive MATLAB file: {input_file}...")
    
    with open(input_file, 'r') as f:
        matlab_code = f.read()

    # 1. Header and function definition.
    py_code = re.sub(
        r'function\s*\[(.*?)\]\s*=\s*computeGeneralSolution\((.*?)\)', 
        r'import numpy as np\n\ndef computeGeneralSolution(\2):', 
        matlab_code
    )

    # 2. Delete MATLAB comments.
    py_code = re.sub(r'%.*', '', py_code)

    # 3. Fix state vector indexing
    for i in range(1, 7):
        py_code = py_code.replace(f'X0({i})', f'X0[{i-1}]')

    # 4. Convert mathematical element-wise operators.
    py_code = py_code.replace('.*', '*')
    py_code = py_code.replace('./', '/')
    py_code = py_code.replace('.^', '**')
    py_code = py_code.replace('^', '**')

    # 5. Map trigonometric functions to the NumPy library.
    py_code = re.sub(r'\bsin\(', 'np.sin(', py_code)
    py_code = re.sub(r'\bcos\(', 'np.cos(', py_code)

    # 6. Remove trailing semicolons.
    py_code = re.sub(r';\s*\n', '\n', py_code)
    
    # 7. Remove the MATLAB 'end' keyword at the EOF.
    py_code = py_code.replace('\nend', '')

    # 8. Add Python indentation.
    lines = py_code.split('\n')
    py_lines = []
    
    for line in lines:
        if line.startswith('import') or line.startswith('def compute'):
            py_lines.append(line)
        elif line.strip() == '':
            py_lines.append(line)
        else:
            py_lines.append('    ' + line)
            
    # 9. Return the state vector as NumPy arrays (ORBITA format).
    # Grouping the output variables into 3D position and velocity vectors.
    py_lines.append('    pos = np.array([xB, yB, zB])')
    py_lines.append('    vel = np.array([xdotB, ydotB, zdotB])')
    py_lines.append('    return pos, vel')

    # 10. Save the generated code to the output file.
    # Creates the target directory if it does not exist.
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_file, 'w') as f:
        f.write('\n'.join(py_lines))

    print(f"Success! Python analytical file saved to: {output_file}")


# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    translateESTHER2ORBITA('src/computeGeneralSolution.m', 'src/analytical.py')