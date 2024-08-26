from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_5a5a2103(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying a 4x4 pattern to each section of the grid.
    
    The transformation works as follows:
    1. Identifies the dividing lines in the grid (color 8).
    2. For each row of sections:
       a. Finds the first non-zero, non-dividing-line color in the row.
       b. If a color is found, generates a 4x4 pattern for this color.
       c. Applies this pattern across the entire row, respecting dividing lines.
       d. If no color is found, leaves the row unchanged.
    3. Preserves the original dividing lines in the output.
    4. Applies the pattern to all subsections in each row, even if they originally contained different colors.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid.
    """
    
    def find_dividing_lines(grid: List[List[int]]) -> Tuple[List[int], List[int]]:
        rows, cols = len(grid), len(grid[0])
        horizontal_lines = [i for i in range(rows) if all(cell == 8 for cell in grid[i])]
        vertical_lines = [j for j in range(cols) if all(grid[i][j] == 8 for i in range(rows))]
        return horizontal_lines, vertical_lines
    
    def find_pattern_color(row: List[int]) -> int:
        return next((color for color in row if color not in [0, 8]), 0)
    
    def generate_pattern(color: int) -> List[List[int]]:
        return [
            [color, color, 0, color],
            [0, color, color, 0],
            [color, color, color, color],
            [color, 0, 0, color]
        ]
    
    def apply_pattern(grid: List[List[int]], pattern: List[List[int]], start_row: int, end_row: int,
                      start_col: int, end_col: int) -> None:
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                if grid[row][col] != 8:
                    grid[row][col] = pattern[(row - start_row) % 4][(col - start_col) % 4]
    
    # Find dividing lines
    horizontal_lines, vertical_lines = find_dividing_lines(input_grid.values)
    
    # Create a new grid with the same dimensions as the input
    new_grid = [row[:] for row in input_grid.values]
    
    # Process each row of sections
    row = 0
    while row < len(new_grid):
        if row in horizontal_lines:
            row += 1
            continue
        
        # Find the end of this section row
        next_horizontal = next((line for line in horizontal_lines if line > row), len(new_grid))
        
        # Find the pattern color for this row
        pattern_color = find_pattern_color(new_grid[row])
        
        if pattern_color != 0:
            pattern = generate_pattern(pattern_color)
            
            # Apply the pattern across the row
            col = 0
            while col < len(new_grid[0]):
                if col in vertical_lines:
                    col += 1
                    continue
                
                # Find the end of this section column
                next_vertical = next((line for line in vertical_lines if line > col), len(new_grid[0]))
                
                apply_pattern(new_grid, pattern, row, next_horizontal, col, next_vertical)
                col = next_vertical
        
        row = next_horizontal
    
    return ColoredGrid(values=new_grid)
