from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

class Shape:
    def __init__(self, colors: List[List[int]], top: int, left: int):
        self.colors = colors
        self.height = len(colors)
        self.width = len(colors[0])
        self.top = top
        self.left = left

    def can_place_at(self, grid: List[List[int]], top: int, left: int) -> bool:
        if top + self.height > len(grid) or left + self.width > len(grid[0]):
            return False
        return all(grid[top + i][left + j] == 0
                   for i in range(self.height)
                   for j in range(self.width))

    def place(self, grid: List[List[int]], top: int, left: int):
        for i in range(self.height):
            for j in range(self.width):
                if left + j < len(grid[0]):  # Ensure we don't go out of bounds
                    grid[top + i][left + j] = self.colors[i][j]

def solve_d282b262(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving all non-zero content to the right side of the grid.
    
    1. Analyzes the input grid to find the bounds of non-zero content.
    2. Extracts the non-zero content.
    3. Creates a new grid with the extracted content placed on the right side,
       maintaining a 3-column gap on the right.
    4. Aligns the content to the top of the grid.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with non-zero content moved to the right side.
    """
    grid = input_grid.values
    rows, cols = len(grid), len(grid[0])

    # Step 1: Analyze the input grid
    left_bound = min((col for col in range(cols) for row in range(rows) if grid[row][col] != 0), default=0)
    right_bound = max((col for col in range(cols) for row in range(rows) if grid[row][col] != 0), default=cols-1)
    top_bound = min((row for row in range(rows) for col in range(cols) if grid[row][col] != 0), default=0)
    bottom_bound = max((row for row in range(rows) for col in range(cols) if grid[row][col] != 0), default=rows-1)

    # Step 2: Calculate dimensions and extract non-zero content
    width = right_bound - left_bound + 1
    height = bottom_bound - top_bound + 1
    content = [[grid[row][col] for col in range(left_bound, right_bound + 1)] 
               for row in range(top_bound, bottom_bound + 1)]

    # Step 3 & 4: Create new output grid and place extracted content
    output_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    new_left = cols - width - 3
    new_top = 0

    for i in range(height):
        for j in range(width):
            output_grid[new_top + i][new_left + j] = content[i][j]

    return ColoredGrid(values=output_grid)
