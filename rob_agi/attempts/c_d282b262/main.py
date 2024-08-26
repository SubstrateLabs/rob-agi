from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

class Shape:
    def __init__(self, colors: List[List[int]], top: int, left: int):
        self.colors = colors
        self.height = len(colors)
        self.width = len(colors[0])
        self.top = top
        self.left = left
        self.new_top = 0
        self.new_left = 0

def extract_shape(grid: List[List[int]], top: int, left: int, bottom: int, right: int) -> List[List[int]]:
    return [[grid[r][c] for c in range(left, right + 1)] for r in range(top, bottom + 1)]

def can_place_shape(new_grid: List[List[int]], shape: List[List[int]], top: int, left: int) -> bool:
    rows, cols = len(new_grid), len(new_grid[0])
    shape_height, shape_width = len(shape), len(shape[0])
    if top + shape_height > rows or left + shape_width > cols:
        return False
    return all(new_grid[top + i][left + j] == 0 for i in range(shape_height) for j in range(shape_width) if shape[i][j] != 0)

def place_shape(new_grid: List[List[int]], shape: List[List[int]], top: int, left: int) -> None:
    for i in range(len(shape)):
        for j in range(len(shape[0])):
            if shape[i][j] != 0:
                new_grid[top + i][left + j] = shape[i][j]

def solve_d282b262(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving all non-zero shapes to the right side of the grid.
    
    1. Identifies and extracts all non-zero shapes from the input grid.
    2. Sorts shapes based on their original top position.
    3. Places shapes on the right side of the grid, maintaining vertical order.
    4. Compacts the arrangement vertically and horizontally.
    5. Ensures shapes are aligned to the right as much as possible.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with non-zero shapes moved to the right side.
    """
    grid = input_grid.values
    rows, cols = len(grid), len(grid[0])

    # Step 1: Identify and extract shapes
    shapes = []
    visited = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r, c) not in visited:
                region = input_grid.find_connected_regions(grid[r][c])[0]
                visited.update(region)
                top = min(x for x, _ in region)
                left = min(y for _, y in region)
                bottom = max(x for x, _ in region)
                right = max(y for _, y in region)
                shape = extract_shape(grid, top, left, bottom, right)
                shapes.append(Shape(shape, top, left))

    # Step 2: Sort shapes based on original top position
    shapes.sort(key=lambda s: s.top)

    # Step 3 & 4: Place shapes and compact
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    current_row = 0
    for shape in shapes:
        placed = False
        for col in range(cols - shape.width, -1, -1):
            if can_place_shape(new_grid, shape.colors, current_row, col):
                place_shape(new_grid, shape.colors, current_row, col)
                placed = True
                current_row += shape.height + 1  # Add a gap of 1 row
                break
        if not placed:
            raise ValueError("Not enough space to place all shapes")

    # Step 5: Compact vertically and align to right
    compact_vertically(new_grid)
    align_right(new_grid)

    return ColoredGrid(values=new_grid)

def compact_vertically(grid):
    rows, cols = len(grid), len(grid[0])
    for col in range(cols - 1, -1, -1):
        non_zero = [row for row in range(rows) if grid[row][col] != 0]
        if non_zero:
            for i, row in enumerate(non_zero):
                while row > i and all(grid[row-1][c] == 0 for c in range(col, cols)):
                    grid[row-1][col:], grid[row][col:] = grid[row][col:], grid[row-1][col:]
                    row -= 1

def align_right(grid):
    rows, cols = len(grid), len(grid[0])
    for row in range(rows):
        non_zero = [col for col in range(cols) if grid[row][col] != 0]
        if non_zero:
            shift = cols - max(non_zero) - 1
            if shift > 0:
                grid[row] = [0] * shift + grid[row][:-shift]
