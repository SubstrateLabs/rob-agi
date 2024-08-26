from rob_agi.colored_grid import ColoredGrid
from typing import Dict, List, Tuple

def solve_b7999b51(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing non-black color regions into columns.
    
    The function performs the following steps:
    1. Analyzes the input grid to gather information about each non-black color.
    2. Sorts colors based on their leftmost appearance in the input grid.
    3. Creates a new grid with columns representing each color, maintaining their sorted order and relative heights.
    4. Aligns all colors to the top of the output grid.
    5. Optimizes the output by removing any completely black rows from the bottom.
    
    The transformation follows these rules:
    - Each non-black color is compressed into a single column.
    - Colors are ordered from left to right based on their leftmost occurrence in the input.
    - The height of each color column is determined by its vertical span in the input.
    - All color columns are aligned to the top of the output grid.
    - Any remaining empty (black) rows at the bottom are removed.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with compressed color columns, preserving left-to-right order and relative heights.
    """
    # Step 1: Analyze the input grid
    color_info: Dict[int, Dict[str, int]] = {}
    rows, cols = input_grid.get_dimensions()
    
    for c in range(cols):
        for r in range(rows):
            color = input_grid.values[r][c]
            if color != 0:  # Non-black color
                if color not in color_info:
                    color_info[color] = {"left": c, "top": r, "bottom": r}
                else:
                    info = color_info[color]
                    info["left"] = min(info["left"], c)
                    info["top"] = min(info["top"], r)
                    info["bottom"] = max(info["bottom"], r)

    # Step 2: Sort colors based on leftmost appearance
    sorted_colors = sorted(color_info.items(), key=lambda x: x[1]["left"])

    # Step 3 & 4: Create and fill the output grid
    output_width = len(sorted_colors)
    output_height = max(info["bottom"] - info["top"] + 1 for _, info in sorted_colors)
    output_grid = [[0 for _ in range(output_width)] for _ in range(output_height)]

    for col, (color, info) in enumerate(sorted_colors):
        height = info["bottom"] - info["top"] + 1
        for row in range(height):
            output_grid[row][col] = color

    # Step 5: Optimize the output (remove black rows from the bottom)
    while output_grid and all(cell == 0 for cell in output_grid[-1]):
        output_grid.pop()

    return ColoredGrid(values=output_grid)
