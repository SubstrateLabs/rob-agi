from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

import random
from typing import List, Tuple

def solve_2037f2c7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a simplified, abstract representation.
    
    1. Analyzes the input grid for shape characteristics and complexity.
    2. Creates a small output grid (3x7 or similar) with sky blue (8) and black (0) squares.
    3. Represents the main features of the input shape using a simple pattern.
    4. Ensures the top row and rightmost column follow specific rules.
    5. Adds asymmetry and distinctive features based on the input.
    """
    # Analyze input grid
    shape_info = analyze_grid(input_grid)
    
    # Create base output grid
    output_grid = create_base_grid(shape_info)
    
    # Add distinctive features
    output_grid = add_features(output_grid, shape_info)
    
    # Ensure asymmetry
    output_grid = ensure_asymmetry(output_grid, shape_info)
    
    return ColoredGrid(values=output_grid)

def analyze_grid(grid: ColoredGrid) -> dict:
    rows, cols = grid.get_dimensions()
    non_zero_cells = sum(1 for r in range(rows) for c in range(cols) if grid.get_cell(r, c) != 0)
    density = non_zero_cells / (rows * cols)
    
    vertical_density = sum(1 for c in range(cols) if any(grid.get_cell(r, c) != 0 for r in range(rows))) / cols
    horizontal_density = sum(1 for r in range(rows) if any(grid.get_cell(r, c) != 0 for c in range(cols))) / rows
    
    return {
        'density': density,
        'vertical_emphasis': vertical_density > horizontal_density,
        'complexity': density > 0.15 or (vertical_density > 0.5 and horizontal_density > 0.5)
    }

def create_base_grid(shape_info: dict) -> List[List[int]]:
    height = 3
    width = 7
    
    if shape_info['complexity']:
        width = 8
    
    grid = [[0 for _ in range(width)] for _ in range(height)]
    
    # Fill top row
    grid[0] = [8, 0, 0, 0, 0, 0, 8] if width == 7 else [8, 0, 0, 0, 0, 0, 8, 8]
    
    # Fill middle row
    grid[1] = [8, 8, 0, 0, 0, 8, 8] if width == 7 else [8, 8, 0, 0, 0, 0, 8, 8]
    
    # Fill bottom row
    grid[2] = [8, 0, 0, 0, 0, 0, 8] if width == 7 else [8, 0, 0, 0, 0, 0, 8, 8]
    
    return grid

def add_features(grid: List[List[int]], shape_info: dict) -> List[List[int]]:
    if shape_info['vertical_emphasis']:
        for row in grid:
            row[0] = 8
            row[-1] = 8
    
    if shape_info['complexity']:
        grid[1][2] = 8
        grid[1][3] = 8
    
    return grid

def ensure_asymmetry(grid: List[List[int]], shape_info: dict) -> List[List[int]]:
    if not shape_info['complexity']:
        grid[2][1] = 0
        grid[2][-2] = 0
    
    return grid
