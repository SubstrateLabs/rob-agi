from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_67636eac(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts 3x3 shapes from the input grid and arranges them vertically in a new grid.
    
    1. Scans the input grid to identify non-black shapes.
    2. Extracts a 3x3 subgrid centered on each shape.
    3. Sorts the shapes based on their original positions.
    4. Creates a new grid with the extracted shapes arranged vertically.
    
    Args:
    input_grid (ColoredGrid): The input grid containing shapes.
    
    Returns:
    ColoredGrid: A new grid with extracted shapes arranged vertically.
    """
    shapes = []
    rows, cols = input_grid.get_dimensions()
    visited = set()

    # Identify and extract shapes
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and input_grid.get_cell(r, c) != 0:
                color = input_grid.get_cell(r, c)
                region = input_grid.find_connected_regions(color)[0]
                visited.update(region)
                
                # Calculate center of the shape
                min_r = min(x[0] for x in region)
                max_r = max(x[0] for x in region)
                min_c = min(x[1] for x in region)
                max_c = max(x[1] for x in region)
                center_r = (min_r + max_r) // 2
                center_c = (min_c + max_c) // 2
                
                # Extract 3x3 subgrid
                subgrid = input_grid.extract_subgrid(center_r - 1, center_c - 1, 3, 3)
                shapes.append((min_r, min_c, subgrid))

    # Sort shapes based on original position
    shapes.sort(key=lambda x: (x[0], x[1]))

    # Create output grid
    output_height = 3 * len(shapes)
    output_grid = ColoredGrid(values=[[0 for _ in range(3)] for _ in range(output_height)])

    # Populate output grid
    for i, (_, _, shape) in enumerate(shapes):
        for r in range(3):
            for c in range(3):
                output_grid.set_cell(i * 3 + r, c, shape.get_cell(r, c))

    return output_grid
