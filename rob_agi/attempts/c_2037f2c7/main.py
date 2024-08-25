from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

import random
from typing import List, Tuple

def solve_2037f2c7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a simplified, abstract representation.
    
    1. Analyzes the input grid for density, shape distribution, and emphasis.
    2. Determines the output grid size based on input complexity.
    3. Generates a base pattern with sky blue (8) and black (0) squares.
    4. Adjusts the pattern based on input characteristics (vertical/horizontal emphasis, multiple shapes).
    5. Ensures a pixelated look with appropriate asymmetry.
    6. Simplifies the output for less complex inputs.
    """
    # Step 1: Analyze input grid
    overall_density, vertical_emphasis, num_shapes = analyze_grid(input_grid)
    
    # Step 2: Determine output grid size
    output_height, output_width = determine_output_size(overall_density, num_shapes)
    
    # Step 3 & 4: Generate and adjust output grid
    output_grid = generate_output_grid(overall_density, vertical_emphasis, num_shapes, output_height, output_width)
    
    # Step 5 & 6: Ensure pixelated look and simplify if necessary
    output_grid = post_process_grid(output_grid, overall_density)
    
    return ColoredGrid(values=output_grid)

def analyze_grid(grid: ColoredGrid) -> Tuple[float, bool, int]:
    rows, cols = grid.get_dimensions()
    non_zero_cells = sum(1 for r in range(rows) for c in range(cols) if grid.get_cell(r, c) != 0)
    overall_density = non_zero_cells / (rows * cols)
    
    vertical_density = sum(1 for c in range(cols) if any(grid.get_cell(r, c) != 0 for r in range(rows))) / cols
    horizontal_density = sum(1 for r in range(rows) if any(grid.get_cell(r, c) != 0 for c in range(cols))) / rows
    vertical_emphasis = vertical_density > horizontal_density
    
    non_zero_regions = grid.find_connected_regions(lambda x: x != 0)
    num_shapes = len([region for region in non_zero_regions if len(region) > 5])
    
    return overall_density, vertical_emphasis, num_shapes

def determine_output_size(density: float, num_shapes: int) -> Tuple[int, int]:
    if density < 0.1 or num_shapes == 1:
        return 3, 7
    elif density < 0.15:
        return 3, 8
    else:
        return 4, 8

def generate_output_grid(density: float, vertical_emphasis: bool, num_shapes: int, height: int, width: int) -> List[List[int]]:
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    # Generate top row
    output[0] = [8] * width
    num_gaps = random.randint(1, 3)
    for _ in range(num_gaps):
        gap_pos = random.randint(1, width-2)
        output[0][gap_pos] = 0
    
    # Generate middle rows
    for r in range(1, height-1):
        output[r] = [8, 8] + [0] * (width-4) + [8, 8]
        num_fills = int((width-4) * density * 0.7)
        for _ in range(num_fills):
            fill_pos = random.randint(2, width-3)
            output[r][fill_pos] = 8
    
    # Generate bottom row
    output[-1] = [8] + [0] * (width-2) + [8]
    if density > 0.2:
        num_fills = random.randint(1, 2)
        for _ in range(num_fills):
            fill_pos = random.randint(1, width-2)
            output[-1][fill_pos] = 8
    
    # Adjust for vertical emphasis
    if vertical_emphasis:
        for r in range(height):
            output[r][0] = 8
            output[r][-1] = 8
    
    # Adjust for multiple shapes
    if num_shapes > 1:
        mid = width // 2
        for r in range(1, height):
            output[r][mid-1:mid+1] = [0, 0]
    
    return output

def post_process_grid(grid: List[List[int]], density: float) -> List[List[int]]:
    height, width = len(grid), len(grid[0])
    
    # Ensure pixelated look
    for r in range(height):
        if 0 not in grid[r]:
            grid[r][random.randint(0, width-1)] = 0
    for c in range(width):
        if all(grid[r][c] != 0 for r in range(height)):
            grid[random.randint(0, height-1)][c] = 0
    
    # Add asymmetry
    if random.random() < 0.5:
        change_pos = random.randint(1, width-2)
        grid[0][change_pos] = 8 if grid[0][change_pos] == 0 else 0
    
    # Simplify for low density inputs
    if density < 0.1 and height > 2:
        grid = grid[:-1]
    
    return grid
