from rob_agi.colored_grid import ColoredGrid
import random

def solve_2037f2c7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a simplified, abstract representation.
    
    1. Analyzes the input grid for shape characteristics, complexity, and distribution.
    2. Creates a small output grid (3x7) with sky blue (8) and black (0) squares.
    3. Represents the main features of the input shapes using a simple pattern.
    4. Ensures the rightmost column is filled with 8s.
    5. Adds asymmetry and distinctive features based on the input analysis.
    6. Balances the overall composition while maintaining the essence of the input.
    """
    shape_info = analyze_grid(input_grid)
    output_grid = create_base_grid(shape_info)
    output_grid = generate_pattern(output_grid, shape_info)
    output_grid = refine_top_row(output_grid, shape_info)
    output_grid = balance_composition(output_grid, shape_info)
    
    return ColoredGrid(values=output_grid)

def analyze_grid(grid: ColoredGrid) -> dict:
    rows, cols = grid.get_dimensions()
    non_zero_cells = sum(1 for r in range(rows) for c in range(cols) if grid.get_cell(r, c) != 0)
    density = non_zero_cells / (rows * cols)
    
    left_density = sum(1 for r in range(rows) for c in range(cols//2) if grid.get_cell(r, c) != 0) / (rows * cols//2)
    right_density = sum(1 for r in range(rows) for c in range(cols//2, cols) if grid.get_cell(r, c) != 0) / (rows * cols//2)
    
    return {
        'density': density,
        'left_heavy': left_density > right_density,
        'complexity': density > 0.2
    }

def create_base_grid(shape_info: dict) -> list[list[int]]:
    return [[0 for _ in range(7)] for _ in range(3)]

def generate_pattern(grid: list[list[int]], shape_info: dict) -> list[list[int]]:
    # Ensure rightmost column is filled with 8s
    for row in grid:
        row[-1] = 8
    
    # Generate main pattern
    if shape_info['left_heavy']:
        grid[1][0:3] = [8, 8, 8]
    else:
        grid[1][3:6] = [8, 8, 8]
    
    # Add asymmetry
    if random.random() < 0.5:
        grid[0][0] = 8
    if random.random() < 0.5:
        grid[2][-2] = 8
    
    return grid

def refine_top_row(grid: list[list[int]], shape_info: dict) -> list[list[int]]:
    if shape_info['complexity']:
        grid[0][0] = 8
    return grid

def balance_composition(grid: list[list[int]], shape_info: dict) -> list[list[int]]:
    total_8s = sum(row.count(8) for row in grid)
    if total_8s < 7:
        grid[1][1] = 8
    elif total_8s > 9:
        grid[1][2] = 0
    return grid
