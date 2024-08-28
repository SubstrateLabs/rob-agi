from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_137f0df0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a red background, preserving gray blocks,
    adding blue squares at specific intervals, and maintaining black squares.
    
    1. Creates a full red (2) background.
    2. Transfers the original gray (5) blocks to their corresponding positions.
    3. Adds blue (1) squares:
       - In gaps between gray column groups.
       - At the edges if no gray squares are touching them.
       - In rows immediately above and below gray rows.
    4. Preserves all original black (0) squares in their positions.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid according to the specified pattern.
    """
    def find_gray_columns(grid: List[List[int]]) -> List[int]:
        return [c for c in range(len(grid[0])) if any(row[c] == 5 for row in grid)]

    def find_gray_rows(grid: List[List[int]]) -> List[int]:
        return [r for r, row in enumerate(grid) if 5 in row]

    def create_transformed_grid(input_grid: List[List[int]], gray_columns: List[int], gray_rows: List[int]) -> List[List[int]]:
        rows, cols = len(input_grid), len(input_grid[0])
        new_grid = [[2 for _ in range(cols)] for _ in range(rows)]  # Fill with red

        # Transfer gray blocks and preserve all original black squares
        for r in range(rows):
            for c in range(cols):
                if input_grid[r][c] in [0, 5]:
                    new_grid[r][c] = input_grid[r][c]

        # Process horizontal blue squares
        for r in range(rows):
            if r not in gray_rows and (r-1 in gray_rows or r+1 in gray_rows):
                for c in range(cols):
                    if c not in gray_columns and new_grid[r][c] == 2:
                        new_grid[r][c] = 1

        # Process vertical blue squares
        for c in range(cols):
            if c not in gray_columns:
                for r in range(rows):
                    if r not in gray_rows and new_grid[r][c] == 2:
                        new_grid[r][c] = 1

        # Process blue squares in gray rows
        for r in gray_rows:
            for c in range(cols):
                if c not in gray_columns and all(new_grid[r][gc] != 5 for gc in gray_columns):
                    new_grid[r][c] = 1

        return new_grid

    input_values = input_grid.values
    gray_columns = find_gray_columns(input_values)
    gray_rows = find_gray_rows(input_values)
    new_grid = create_transformed_grid(input_values, gray_columns, gray_rows)
    
    return ColoredGrid(values=new_grid)
