from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

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

def can_place_shape(new_grid: List[List[int]], shape: Shape, top: int, left: int) -> bool:
    if top + shape.height > len(new_grid) or left + shape.width > len(new_grid[0]):
        return False
    return all(new_grid[top + i][left + j] == 0 for i in range(shape.height) for j in range(shape.width))

def place_shape(new_grid: List[List[int]], shape: Shape, top: int, left: int) -> None:
    for i in range(shape.height):
        for j in range(shape.width):
            new_grid[top + i][left + j] = shape.colors[i][j]

def solve_d282b262(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving all non-zero shapes to the right side of the grid.
    
    1. Identifies and extracts all non-zero shapes from the input grid.
    2. Sorts shapes based on their original top position (top-to-bottom order).
    3. Calculates new positions for shapes on the right side of the grid, starting from the top.
    4. Adjusts vertical positions to avoid overlap and maximize space usage.
    5. Creates a new grid and places shapes in their new positions.
    6. Performs a final adjustment to eliminate unnecessary vertical gaps.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with non-zero shapes moved to the right side.
    """
    grid = input_grid.values
    rows, cols = len(grid), len(grid[0])

    # Step 1: Identify and extract shapes
    shapes = []
    for color in range(1, 10):  # Assuming colors are 1-9
        regions = input_grid.find_connected_regions(color)
        for region in regions:
            top = min(r for r, _ in region)
            left = min(c for _, c in region)
            bottom = max(r for r, _ in region)
            right = max(c for _, c in region)
            shape_grid = extract_shape(grid, top, left, bottom, right)
            shapes.append(Shape(shape_grid, top, left))

    # Step 2: Sort shapes based on original top position
    shapes.sort(key=lambda s: s.top)

    # Step 3 & 4: Calculate new positions and adjust for overlap
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    current_right_column = cols - 1
    for shape in shapes:
        shape.new_left = max(0, current_right_column - shape.width + 1)
        shape.new_top = 0
        while not can_place_shape(new_grid, shape, shape.new_top, shape.new_left):
            shape.new_top += 1
            if shape.new_top + shape.height > rows:
                shape.new_top = 0
                shape.new_left -= 1
                if shape.new_left < 0:
                    raise ValueError("Not enough space to place all shapes")
        place_shape(new_grid, shape, shape.new_top, shape.new_left)
        current_right_column = shape.new_left - 1

    # Step 5: Final vertical adjustment
    for shape in shapes:
        while shape.new_top > 0 and can_place_shape(new_grid, shape, shape.new_top - 1, shape.new_left):
            shape.new_top -= 1
        place_shape(new_grid, shape, shape.new_top, shape.new_left)

    return ColoredGrid(values=new_grid)
