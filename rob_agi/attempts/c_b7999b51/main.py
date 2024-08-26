from rob_agi.colored_grid import ColoredGrid
from collections import OrderedDict
from typing import Dict, List, Tuple

def solve_b7999b51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing non-black color regions into columns.
    
    The function performs the following steps:
    1. Analyzes the input grid to gather information about each non-black color, preserving left-to-right order.
    2. Creates a new grid with columns representing each color, maintaining their order of appearance and relative heights.
    3. Aligns all colors to the top of the output grid.
    4. Optimizes the output by removing any completely black rows from the bottom.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with compressed color columns, preserving left-to-right order.
    """
    # Step 1: Analyze the input grid
    color_info: OrderedDict[int, Dict[str, int]] = OrderedDict()
    rows, cols = input_grid.get_dimensions()
    
    for c in range(cols):
        for r in range(rows):
            color = input_grid.values[r][c]
            if color != 0:  # Non-black color
                if color not in color_info:
                    color_info[color] = {"left": c, "top": r, "bottom": r, "width": 0}
                info = color_info[color]
                info["top"] = min(info["top"], r)
                info["bottom"] = max(info["bottom"], r)
                info["width"] += 1

    # Step 2 & 3: Create and fill the output grid
    output_width = len(color_info)
    output_height = max(info["bottom"] - info["top"] + 1 for info in color_info.values())
    output_grid = [[0 for _ in range(output_width)] for _ in range(output_height)]

    for col, (color, info) in enumerate(color_info.items()):
        height = info["bottom"] - info["top"] + 1
        for row in range(height):
            output_grid[row][col] = color

    # Step 4: Optimize the output (remove black rows from the bottom)
    while output_grid and all(cell == 0 for cell in output_grid[-1]):
        output_grid.pop()

    return ColoredGrid(values=output_grid)
