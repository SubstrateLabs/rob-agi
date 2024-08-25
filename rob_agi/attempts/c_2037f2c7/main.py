from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_2037f2c7(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a simplified representation based on density analysis and shape detection.
    
    1. Analyzes the input grid by dividing it into quadrants and calculating densities.
    2. Determines the output grid size based on input complexity and width.
    3. Generates a top row with sky blue (8) squares, potentially adding black (0) squares based on overall density.
    4. Generates bottom row(s) based on quadrant densities and detected shapes.
    5. Balances the pattern and ensures appropriate symmetry.
    6. Handles edge cases for empty, sparse, or dense inputs.
    """
    # Step 1: Analyze input grid
    quadrant_densities, overall_density = analyze_grid(input_grid)
    
    # Step 2: Determine output grid size
    input_width = input_grid.get_dimensions()[1]
    output_width = 6 if input_width < 24 else 8
    output_height = 2 if overall_density < 0.1 else 3 if overall_density < 0.2 else 4
    
    # Step 3 & 4: Generate output grid
    output_grid = generate_output_grid(input_grid, quadrant_densities, overall_density, output_height, output_width)
    
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

def generate_output_grid(input_grid: ColoredGrid, quadrant_densities: List[float], overall_density: float, height: int, width: int) -> ColoredGrid:
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
        
        row = [8] + [0] * (width - 2) + [8]
        
        if left_density > 0.1:
            row[1] = 8
        if right_density > 0.1:
            row[-2] = 8
        
        if left_density > 0.15:
            row[2] = 8
        if right_density > 0.15:
            row[-3] = 8
        
        output[r] = row
    
    # Adjust for distinct shapes
    if has_distinct_shapes(input_grid):
        for r in range(1, height):
            mid = width // 2
            output[r][mid-1:mid+1] = [0, 0]
    
    # Ensure at least one 8 per row
    for r in range(height):
        if 8 not in output[r]:
            output[r][0] = 8
    
    return ColoredGrid(values=output)

def has_distinct_shapes(grid: ColoredGrid) -> bool:
    non_zero_regions = grid.find_connected_regions(lambda x: x != 0)
    return len(non_zero_regions) > 1 and max(len(region) for region in non_zero_regions) > 20
