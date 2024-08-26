from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_67636eac(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts 3x3 shapes from the input grid and arranges them in a new grid.
    
    1. Scans the input grid to identify non-black shapes.
    2. Extracts a 3x3 subgrid centered on each shape.
    3. Determines the orientation (horizontal or vertical) based on shape distribution.
    4. Arranges shapes in the order they appear in the input grid (left-to-right, top-to-bottom).
    5. Creates a new grid with the extracted shapes arranged accordingly.
    
    Args:
    input_grid (ColoredGrid): The input grid containing shapes.
    
    Returns:
    ColoredGrid: A new grid with extracted shapes arranged horizontally or vertically.
    """
    shapes = []
    rows, cols = input_grid.get_dimensions()
    visited = set()
    unique_rows = set()
    unique_cols = set()

    # Identify and extract shapes
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and input_grid.get_cell(r, c) != 0:
                color = input_grid.get_cell(r, c)
                region = input_grid.find_connected_regions(color)[0]
                visited.update(region)
                
                # Calculate center of the shape
                center_r = sum(x[0] for x in region) // len(region)
                center_c = sum(x[1] for x in region) // len(region)
                
                # Extract 3x3 subgrid
                subgrid = input_grid.extract_subgrid(center_r - 1, center_c - 1, 3, 3)
                shapes.append((r, c, subgrid))
                unique_rows.add(r)
                unique_cols.add(c)

    # Determine orientation
    vertical_orientation = len(unique_rows) >= len(unique_cols)

    # Sort shapes based on their original position (left-to-right, top-to-bottom)
    shapes.sort(key=lambda x: (x[1], x[0]) if vertical_orientation else (x[0], x[1]))

    # Create output grid
    if vertical_orientation:
        output_width = 3
        output_height = 3 * len(shapes)
    else:
        output_width = 3 * len(shapes)
        output_height = 3
    
    output_grid = ColoredGrid(values=[[0 for _ in range(output_width)] for _ in range(output_height)])

    # Populate output grid
    for i, (_, _, shape) in enumerate(shapes):
        for r in range(3):
            for c in range(3):
                if vertical_orientation:
                    output_grid.set_cell(i * 3 + r, c, shape.get_cell(r, c))
                else:
                    output_grid.set_cell(r, i * 3 + c, shape.get_cell(r, c))

    return output_grid
