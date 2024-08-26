from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5a5a2103(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a 4x4 pattern to each section of the grid.
    
    The transformation works as follows:
    1. Identifies the dividing line color (which may not always be 8).
    2. Finds horizontal and vertical dividing lines.
    3. For each row of sections:
       a. Finds the first non-zero, non-dividing-line color in the leftmost section.
       b. If a color is found, generates a 4x4 pattern for this color.
       c. Applies this pattern across all sections in the entire row, respecting dividing lines.
       d. If no color is found, leaves the row unchanged.
    4. Preserves the original dividing lines in the output.
    5. Maintains the original colors in sections where no pattern is applied.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    
    def identify_dividing_color(grid: List[List[int]]) -> int:
        first_row = grid[0]
        first_col = [row[0] for row in grid]
        return first_row[0] if all(cell == first_row[0] for cell in first_row) else first_col[0]
    
    def find_dividing_lines(grid: List[List[int]], dividing_color: int) -> Tuple[List[int], List[int]]:
        rows, cols = len(grid), len(grid[0])
        horizontal_lines = [i for i in range(rows) if all(cell == dividing_color for cell in grid[i])]
        vertical_lines = [j for j in range(cols) if all(grid[i][j] == dividing_color for i in range(rows))]
        return horizontal_lines, vertical_lines
    
    def find_first_color(grid: List[List[int]], row: int, end_col: int, dividing_color: int) -> int:
        return next((color for color in grid[row][:end_col] if color not in [0, dividing_color]), 0)
    
    def generate_pattern(color: int) -> List[List[int]]:
        return [
            [color, color, 0, color],
            [0, color, color, 0],
            [color, color, color, color],
            [color, 0, 0, color]
        ]
    
    def apply_pattern_to_row(grid: List[List[int]], pattern: List[List[int]], 
                             row_start: int, row_end: int, v_lines: List[int]) -> None:
        for section_start, section_end in zip([0] + v_lines, v_lines + [len(grid[0])]):
            if section_start == section_end:
                continue
            for row in range(row_start, row_end):
                for col in range(section_start, section_end):
                    if grid[row][col] != dividing_color:
                        grid[row][col] = pattern[(row - row_start) % 4][(col - section_start) % 4]
    
    def restore_dividing_lines(grid: List[List[int]], dividing_color: int, 
                               h_lines: List[int], v_lines: List[int]) -> None:
        for row in h_lines:
            for col in range(len(grid[0])):
                grid[row][col] = dividing_color
        for col in v_lines:
            for row in range(len(grid)):
                grid[row][col] = dividing_color
    
    dividing_color = identify_dividing_color(input_grid.values)
    h_lines, v_lines = find_dividing_lines(input_grid.values, dividing_color)
    new_grid = [row[:] for row in input_grid.values]
    
    row_sections = zip([0] + h_lines, h_lines + [len(new_grid)])
    for row_start, row_end in row_sections:
        if row_start == row_end:
            continue
        color = find_first_color(new_grid, row_start, v_lines[0] if v_lines else len(new_grid[0]), dividing_color)
        if color != 0:
            pattern = generate_pattern(color)
            apply_pattern_to_row(new_grid, pattern, row_start, row_end, v_lines)
    
    restore_dividing_lines(new_grid, dividing_color, h_lines, v_lines)
    return ColoredGrid(values=new_grid)
