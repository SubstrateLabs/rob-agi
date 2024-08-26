from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_67636eac(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Extracts 3x3 shapes from the input grid and arranges them in a new grid.
    
    1. Scans the input grid to identify non-black shapes.
    2. Extracts a 3x3 subgrid centered on each shape.
    3. Determines if shapes should be arranged horizontally or vertically.
    4. Sorts the shapes based on their original positions.
    5. Creates a new grid with the extracted shapes arranged accordingly.
    
    Args:
    input_grid (ColoredGrid): The input grid containing shapes.
    
    Returns:
    ColoredGrid: A new grid with extracted shapes arranged horizontally or vertically.
    """
    shapes = []
    rows, cols = input_grid.get_dimensions()
    visited = set()
    min_row, max_row, min_col, max_col = rows, 0, cols, 0

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
                
                # Update min/max trackers
                min_row = min(min_row, center_r)
                max_row = max(max_row, center_r)
                min_col = min(min_col, center_c)
                max_col = max(max_col, center_c)
                
                # Extract 3x3 subgrid
                subgrid = input_grid.extract_subgrid(center_r - 1, center_c - 1, 3, 3)
                shapes.append((center_r, center_c, subgrid))

    # Determine arrangement (horizontal or vertical)
    is_horizontal = (max_col - min_col) > (max_row - min_row)

    # Sort shapes based on arrangement
    if is_horizontal:
        shapes.sort(key=lambda x: (x[0], x[1]))  # Sort by row, then column
    else:
        shapes.sort(key=lambda x: (x[1], x[0]))  # Sort by column, then row

    # Create output grid
    if is_horizontal:
        output_width = 3 * len(shapes)
        output_height = 3
    else:
        output_width = 3
        output_height = 3 * len(shapes)

    output_grid = ColoredGrid(values=[[0 for _ in range(output_width)] for _ in range(output_height)])

    # Populate output grid
    for i, (_, _, shape) in enumerate(shapes):
        for r in range(3):
            for c in range(3):
                if is_horizontal:
                    output_grid.set_cell(r, i * 3 + c, shape.get_cell(r, c))
                else:
                    output_grid.set_cell(i * 3 + r, c, shape.get_cell(r, c))

    return output_grid
