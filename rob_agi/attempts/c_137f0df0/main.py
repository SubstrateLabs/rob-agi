from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_137f0df0(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a red background, preserving gray blocks,
    adding blue squares at specific intervals, and maintaining black squares.
    
    1. Creates a full red (2) background.
    2. Transfers the original gray (5) blocks to their corresponding positions.
    3. Adds blue (1) squares on the edges and between gray block columns:
       - On edges with no gray squares touching them.
       - At the top and bottom of columns between gray block groups.
    4. Preserves all original black (0) squares in their positions.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid according to the specified pattern.
    """
    def find_gray_columns(grid: List[List[int]]) -> List[int]:
        return [c for c in range(len(grid[0])) if any(row[c] == 5 for row in grid)]

    def create_transformed_grid(input_grid: List[List[int]], gray_columns: List[int]) -> List[List[int]]:
        rows, cols = len(input_grid), len(input_grid[0])
        new_grid = [[2 for _ in range(cols)] for _ in range(rows)]  # Fill with red

        # Transfer gray blocks and preserve all original black squares
        for r in range(rows):
            for c in range(cols):
                if input_grid[r][c] == 5:
                    new_grid[r][c] = 5
                elif input_grid[r][c] == 0:
                    new_grid[r][c] = 0

        # Process edges
        for edge in ['left', 'right', 'top', 'bottom']:
            if edge == 'left' and 0 not in gray_columns:
                for r in range(rows):
                    if new_grid[r][0] == 2:
                        new_grid[r][0] = 1
            elif edge == 'right' and (cols - 1) not in gray_columns:
                for r in range(rows):
                    if new_grid[r][cols-1] == 2:
                        new_grid[r][cols-1] = 1
            elif edge == 'top' and not any(input_grid[0][c] == 5 for c in range(cols)):
                for c in range(cols):
                    if new_grid[0][c] == 2:
                        new_grid[0][c] = 1
            elif edge == 'bottom' and not any(input_grid[rows-1][c] == 5 for c in range(cols)):
                for c in range(cols):
                    if new_grid[rows-1][c] == 2:
                        new_grid[rows-1][c] = 1

        # Process columns between gray blocks
        gray_columns.sort()
        for i in range(len(gray_columns) - 1):
            for c in range(gray_columns[i] + 1, gray_columns[i+1]):
                if new_grid[0][c] == 2:
                    new_grid[0][c] = 1
                if new_grid[rows-1][c] == 2:
                    new_grid[rows-1][c] = 1

        # Handle columns before first and after last gray column
        for c in range(cols):
            if c < gray_columns[0] or c > gray_columns[-1]:
                if new_grid[0][c] == 2:
                    new_grid[0][c] = 1
                if new_grid[rows-1][c] == 2:
                    new_grid[rows-1][c] = 1

        return new_grid

    input_values = input_grid.values
    gray_columns = find_gray_columns(input_values)
    new_grid = create_transformed_grid(input_values, gray_columns)
    
    return ColoredGrid(values=new_grid)
