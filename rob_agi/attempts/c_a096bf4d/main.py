from rob_agi.colored_grid import ColoredGrid
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

def solve_a096bf4d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Analyze the grid to identify section size, border color, and main interior color.
    2. Process the grid row by row, propagating special colors within each row.
    3. Handle the bottom row separately, applying specific rules for color propagation.
    4. Perform post-processing steps including color replacements and consistency checks.
    5. Preserve the border structure and colors from the input grid.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = input_grid.deep_copy()
    section_size = 5 if rows % 5 == 0 else 4
    border_color = identify_border_color(input_grid)
    main_interior_color = identify_main_interior_color(input_grid)

    # Process each row of sections
    for row in range(0, rows - section_size, section_size):
        process_row(input_grid, output_grid, row, section_size, border_color, main_interior_color)

    # Handle the bottom row separately
    process_bottom_row(input_grid, output_grid, rows - section_size, section_size, border_color, main_interior_color)

    # Post-processing
    post_process(output_grid)

    return output_grid

def identify_border_color(grid: ColoredGrid) -> int:
    return grid.values[1][1]

def identify_main_interior_color(grid: ColoredGrid) -> int:
    interior_colors = [grid.values[i][j] for i in range(2, len(grid.values)-2) for j in range(2, len(grid.values[0])-2)]
    return max(set(interior_colors), key=interior_colors.count)

def process_row(input_grid: ColoredGrid, output_grid: ColoredGrid, row: int, section_size: int, border_color: int, main_color: int):
    special_colors = identify_special_colors(input_grid, row, section_size)
    for color, position in special_colors:
        propagate_color(output_grid, row, section_size, color, position)
    fill_remaining(output_grid, row, section_size, main_color)

def identify_special_colors(grid: ColoredGrid, row: int, section_size: int) -> List[Tuple[int, Tuple[int, int]]]:
    special_colors = []
    for col in range(1, len(grid.values[0]) - 1, section_size):
        for i in range(1, section_size - 1):
            for j in range(1, section_size - 1):
                color = grid.values[row + i][col + j]
                if color not in [0, grid.values[1][1]] and color not in [c for c, _ in special_colors]:
                    special_colors.append((color, (i, j)))
    return sorted(special_colors, key=lambda x: x[0], reverse=True)

def propagate_color(grid: ColoredGrid, row: int, section_size: int, color: int, position: Tuple[int, int]):
    for col in range(1, len(grid.values[0]) - 1, section_size):
        grid.values[row + position[0]][col + position[1]] = color

def fill_remaining(grid: ColoredGrid, row: int, section_size: int, main_color: int):
    for col in range(1, len(grid.values[0]) - 1, section_size):
        for i in range(1, section_size - 1):
            for j in range(1, section_size - 1):
                if grid.values[row + i][col + j] == 0:
                    grid.values[row + i][col + j] = main_color

def process_bottom_row(input_grid: ColoredGrid, output_grid: ColoredGrid, row: int, section_size: int, border_color: int, main_color: int):
    special_colors = identify_special_colors(input_grid, row, section_size)
    for color, position in special_colors:
        propagate_color(output_grid, row, section_size, color, position)
    fill_remaining(output_grid, row, section_size, main_color)

def post_process(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 6:
                grid.values[r][c] = 1
            elif grid.values[r][c] == 4:
                replace_color_4(grid, r, c)

def replace_color_4(grid: ColoredGrid, r: int, c: int):
    rows, cols = grid.get_dimensions()
    for i in range(max(0, r-1), min(rows, r+2)):
        for j in range(max(0, c-1), min(cols, c+2)):
            if grid.values[i][j] in [2, 3, 7, 8]:
                grid.values[r][c] = grid.values[i][j]
                return
