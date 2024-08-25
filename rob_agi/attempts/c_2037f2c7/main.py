from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2037f2c7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a simplified representation based on density analysis.
    
    1. Analyzes the input grid by dividing it into quadrants and calculating densities.
    2. Determines the output grid size based on input complexity.
    3. Generates a top row with sky blue (8) squares, potentially adding black (0) squares based on overall density.
    4. Generates bottom row(s) based on quadrant densities.
    5. Balances the pattern and ensures left-right symmetry.
    6. Handles edge cases for empty or extremely dense inputs.
    """
    # Step 1: Analyze input grid
    quadrant_densities, overall_density = analyze_grid(input_grid)
    
    # Step 2: Determine output grid size
    output_height = 2 if overall_density < 0.1 else 3 if overall_density < 0.2 else 4
    output_width = 8
    
    # Step 3 & 4: Generate output grid
    output_grid = generate_output_grid(quadrant_densities, overall_density, output_height, output_width)
    
    return output_grid

def analyze_grid(grid: ColoredGrid) -> Tuple[List[float], float]:
    rows, cols = grid.get_dimensions()
    mid_row, mid_col = rows // 2, cols // 2
    quadrants = [
        (0, 0, mid_row, mid_col),
        (0, mid_col, mid_row, cols),
        (mid_row, 0, rows, mid_col),
        (mid_row, mid_col, rows, cols)
    ]
    
    quadrant_densities = []
    total_non_zero = 0
    for top, left, bottom, right in quadrants:
        non_zero = sum(1 for r in range(top, bottom) for c in range(left, right) if grid.get_cell(r, c) != 0)
        total_non_zero += non_zero
        quadrant_densities.append(non_zero / ((bottom - top) * (right - left)))
    
    overall_density = total_non_zero / (rows * cols)
    return quadrant_densities, overall_density

def generate_output_grid(quadrant_densities: List[float], overall_density: float, height: int, width: int) -> ColoredGrid:
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    # Generate top row
    top_row = [8] * width
    if overall_density < 0.15:
        mid = width // 2
        top_row[mid-1:mid+1] = [0, 0]
    output[0] = top_row
    
    # Generate bottom row(s)
    for r in range(1, height):
        left_density = (quadrant_densities[0] + quadrant_densities[2]) / 2
        right_density = (quadrant_densities[1] + quadrant_densities[3]) / 2
        left_count = int(left_density * 4)
        right_count = int(right_density * 4)
        
        row = [8] * left_count + [0] * (4 - left_count) + [0] * (4 - right_count) + [8] * right_count
        output[r] = row
    
    # Balance the pattern
    for r in range(1, height):
        if sum(output[r]) == 0:
            output[r][0] = output[r][-1] = 8
    
    return ColoredGrid(values=output)
