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
    3. Places shapes on the right side of the grid, starting from the top-right corner.
    4. Adjusts positions to avoid overlap and maximize space usage.
    5. Performs a final adjustment to eliminate unnecessary vertical gaps.
    
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

    # Step 3 & 4: Place shapes and adjust for overlap
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for shape in shapes:
        placed = False
        for col in range(cols - 1, -1, -1):
            for row in range(rows):
                if can_place_shape(new_grid, shape, row, col):
                    place_shape(new_grid, shape, row, col)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            raise ValueError("Not enough space to place all shapes")

    # Step 5: Final vertical adjustment
    for col in range(cols - 1, -1, -1):
        for row in range(rows - 1, 0, -1):
            if new_grid[row][col] != 0 and new_grid[row - 1][col] == 0:
                # Move shape up
                shape_height = 1
                while row + shape_height < rows and new_grid[row + shape_height][col] != 0:
                    shape_height += 1
                for r in range(row - 1, -1, -1):
                    if all(new_grid[r + i][col] == 0 for i in range(shape_height)):
                        for i in range(shape_height):
                            new_grid[r + i][col] = new_grid[row + i][col]
                            new_grid[row + i][col] = 0
                        break

    return ColoredGrid(values=new_grid)
