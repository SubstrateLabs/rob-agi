from rob_agi.colored_grid import ColoredGrid
from typing import Dict, Set

def solve_c8b7cc0f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 3x3 output grid based on the following rules:
    1. Ignores black (0) and blue (1) colors.
    2. Finds the non-black, non-blue color with the most distinct rows and columns in the input grid.
    3. Creates a 3x3 output grid with the top row filled with the most frequent color.
    4. The second row is filled based on the number of distinct rows (R) where the color appears:
       - If R >= 1, fill one cell; if R >= 2, fill two cells; if R >= 3, fill all three cells.
    5. The third row is filled with one cell if the color appears in 3 or more distinct columns.
    6. If no valid colors are found, returns an all-black 3x3 grid.
    """
    color_info: Dict[int, Dict[str, Set[int]]] = {}

    # Analyze the input grid
    for row in range(len(input_grid.values)):
        for col in range(len(input_grid.values[0])):
            color = input_grid.values[row][col]
            if color not in [0, 1]:  # Ignore black and blue
                if color not in color_info:
                    color_info[color] = {"rows": set(), "columns": set()}
                color_info[color]["rows"].add(row)
                color_info[color]["columns"].add(col)

    # Find the target color
    if not color_info:
        return ColoredGrid(values=[[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    target_color = max(color_info, key=lambda x: len(color_info[x]["rows"]) + len(color_info[x]["columns"]))

    # Calculate output grid values
    R = min(len(color_info[target_color]["rows"]), 3)
    C = len(color_info[target_color]["columns"])

    # Create the output grid
    output_values = [
        [target_color, target_color, target_color],
        [target_color if i < R else 0 for i in range(3)],
        [target_color if C >= 3 else 0, 0, 0]
    ]

    # Return the output grid
    return ColoredGrid(values=output_values)
