from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

class Shape:
    def __init__(self, colors: List[List[int]], top: int, left: int):
        self.colors = colors
        self.height = len(colors)
        self.width = len(colors[0])
        self.top = top
        self.left = left

def solve_d282b262(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving all non-zero shapes to the right side of the grid.
    
    1. Identifies and extracts all non-zero shapes from the input grid.
    2. Sorts shapes based on their original leftmost column and topmost row.
    3. Calculates new positions for shapes on the right side of the grid.
    4. Creates a new grid and places shapes in their new positions.
    5. Maintains a 3-column gap on the right and aligns shapes to the top when possible.
    
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
            shape_grid = [[grid[r][c] for c in range(left, right + 1)] for r in range(top, bottom + 1)]
            shapes.append(Shape(shape_grid, top, left))

    # Step 2: Sort shapes
    shapes.sort(key=lambda s: (s.left, s.top))

    # Step 3: Calculate new positions
    current_right_column = cols - 4
    for shape in shapes:
        shape.new_left = current_right_column - shape.width + 1
        shape.new_top = 0
        current_right_column -= shape.width

    # Step 4 & 5: Create new grid and place shapes
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for shape in shapes:
        # Adjust top position if shape would extend beyond bottom
        if shape.new_top + shape.height > rows:
            shape.new_top = rows - shape.height

        for i in range(shape.height):
            for j in range(shape.width):
                new_grid[shape.new_top + i][shape.new_left + j] = shape.colors[i][j]

    return ColoredGrid(values=new_grid)
