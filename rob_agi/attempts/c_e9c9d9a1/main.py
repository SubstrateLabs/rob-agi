from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_e9c9d9a1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by filling rectangles formed by green (3) lines.
    
    The solution:
    1. Identifies all horizontal and vertical green (3) lines.
    2. Creates a conceptual grid of rectangles.
    3. Fills these rectangles based on their position:
       - Top row: red (2) for leftmost, yellow (4) for rightmost, black (0) for others
       - Bottom row: blue (1) for leftmost, sky blue (8) for rightmost, black (0) for others
       - Middle rows: orange (7) for all except leftmost and rightmost, which remain black (0)
    4. Preserves the green (3) lines and any non-black (0) cells from the input.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid according to the pattern.
    """
    def find_lines(grid: ColoredGrid) -> Tuple[List[int], List[int]]:
        h_lines = [i for i, row in enumerate(grid.values) if all(cell == 3 for cell in row)]
        v_lines = [j for j in range(len(grid.values[0])) if all(row[j] == 3 for row in grid.values)]
        return h_lines, v_lines
    
    def fill_rectangle(grid: ColoredGrid, top: int, left: int, bottom: int, right: int, color: int) -> None:
        for r in range(top + 1, bottom):
            for c in range(left + 1, right):
                if grid.values[r][c] == 0:
                    grid.values[r][c] = color
    
    def is_top_row(row_index: int, total_rows: int) -> bool:
        return row_index == 0
    
    def is_bottom_row(row_index: int, total_rows: int) -> bool:
        return row_index == total_rows - 1
    
    def is_leftmost_column(col_index: int, total_cols: int) -> bool:
        return col_index == 0
    
    def is_rightmost_column(col_index: int, total_cols: int) -> bool:
        return col_index == total_cols - 1
    
    def get_fill_color(row_index: int, col_index: int, total_rows: int, total_cols: int) -> int:
        if is_top_row(row_index, total_rows):
            if is_leftmost_column(col_index, total_cols):
                return 2  # Red
            elif is_rightmost_column(col_index, total_cols):
                return 4  # Yellow
            else:
                return 0  # Black
        elif is_bottom_row(row_index, total_rows):
            if is_leftmost_column(col_index, total_cols):
                return 1  # Blue
            elif is_rightmost_column(col_index, total_cols):
                return 8  # Sky blue
            else:
                return 0  # Black
        else:
            if is_leftmost_column(col_index, total_cols) or is_rightmost_column(col_index, total_cols):
                return 0  # Black
            else:
                return 7  # Orange
    
    # Create a copy of the input grid
    output_grid = input_grid.deep_copy()
    
    # Find horizontal and vertical green lines
    h_lines, v_lines = find_lines(input_grid)
    
    # Fill rectangles based on their position
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            top, bottom = h_lines[i], h_lines[i + 1]
            left, right = v_lines[j], v_lines[j + 1]
            
            color = get_fill_color(i, j, len(h_lines) - 1, len(v_lines) - 1)
            fill_rectangle(output_grid, top, left, bottom, right, color)
    
    return output_grid
