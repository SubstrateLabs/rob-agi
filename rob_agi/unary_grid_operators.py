from typing import List

def rotate_90_clockwise(grid: List[List[int]]) -> List[List[int]]:
    """Rotate the grid 90 degrees clockwise."""
    return [list(row) for row in zip(*grid[::-1])]

def rotate_90_counterclockwise(grid: List[List[int]]) -> List[List[int]]:
    """Rotate the grid 90 degrees counterclockwise."""
    return [list(row) for row in zip(*grid)][::-1]

def flip_horizontal(grid: List[List[int]]) -> List[List[int]]:
    """Flip the grid horizontally."""
    return [row[::-1] for row in grid]

def flip_vertical(grid: List[List[int]]) -> List[List[int]]:
    """Flip the grid vertically."""
    return grid[::-1]

def to_binary(grid: List[List[int]]) -> List[List[int]]:
    """Convert the grid to binary using a threshold of 5."""
    return [[1 if cell >= 5 else 0 for cell in row] for row in grid]

def get_edge_cells(grid: List[List[int]]) -> List[List[int]]:
    """Get all cells on the edge of the grid."""
    rows, cols = len(grid), len(grid[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                result[r][c] = grid[r][c]
    return result

def scale(grid: List[List[int]]) -> List[List[int]]:
    """Scale the grid by a factor of 2."""
    return [[cell for cell in row for _ in range(2)] for row in grid for _ in range(2)]

def invert_colors(grid: List[List[int]]) -> List[List[int]]:
    """Invert the colors of the grid assuming a max value of 9."""
    return [[9 - cell for cell in row] for row in grid]

def crop(grid: List[List[int]]) -> List[List[int]]:
    """Crop the grid by removing 1 cell from each side."""
    return [row[1:-1] for row in grid[1:-1]]

def pad(grid: List[List[int]]) -> List[List[int]]:
    """Pad the grid with 1 cell of value 0 on each side."""
    rows, cols = len(grid), len(grid[0])
    new_grid = [[0] * (cols + 2) for _ in range(rows + 2)]
    for r in range(rows):
        for c in range(cols):
            new_grid[r + 1][c + 1] = grid[r][c]
    return new_grid

def replace_color(grid: List[List[int]]) -> List[List[int]]:
    """Replace all occurrences of color 1 with color 2."""
    return [[2 if cell == 1 else cell for cell in row] for row in grid]

def find_bounding_box(grid: List[List[int]]) -> List[List[int]]:
    """Find the bounding box of color 1 and set it to color 2."""
    rows, cols = len(grid), len(grid[0])
    top = bottom = left = right = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                if top is None:
                    top = bottom = r
                    left = right = c
                else:
                    bottom = max(bottom, r)
                    left = min(left, c)
                    right = max(right, c)
    
    if top is not None:
        new_grid = [row[:] for row in grid]
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if r == top or r == bottom or c == left or c == right:
                    new_grid[r][c] = 2
        return new_grid
    return grid
