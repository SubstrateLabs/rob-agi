from typing import List, Tuple, Optional

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

def to_binary(grid: List[List[int]], threshold: int = 5) -> List[List[int]]:
    """Convert the grid to binary using a threshold (default 5)."""
    return [[1 if cell >= threshold else 0 for cell in row] for row in grid]

def get_edge_cells(grid: List[List[int]]) -> List[List[int]]:
    """Get all cells on the edge of the grid."""
    rows, cols = len(grid), len(grid[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                result[r][c] = grid[r][c]
    return result

def scale(grid: List[List[int]], factor: int) -> List[List[int]]:
    """Scale the grid by a given factor."""
    return [[cell for cell in row for _ in range(factor)] for row in grid for _ in range(factor)]

def invert_colors(grid: List[List[int]], max_value: int = 9) -> List[List[int]]:
    """Invert the colors of the grid."""
    return [[max_value - cell for cell in row] for row in grid]

def crop(grid: List[List[int]], top: int, left: int, height: int, width: int) -> List[List[int]]:
    """Crop the grid to the specified rectangle."""
    return [row[left:left+width] for row in grid[top:top+height]]

def pad(grid: List[List[int]], top: int, right: int, bottom: int, left: int, fill_value: int = 0) -> List[List[int]]:
    """Pad the grid with a specified value."""
    rows, cols = len(grid), len(grid[0])
    new_rows = rows + top + bottom
    new_cols = cols + left + right
    new_grid = [[fill_value] * new_cols for _ in range(new_rows)]
    for r in range(rows):
        for c in range(cols):
            new_grid[r + top][c + left] = grid[r][c]
    return new_grid

def replace_color(grid: List[List[int]], old_color: int, new_color: int) -> List[List[int]]:
    """Replace all occurrences of one color with another."""
    return [[new_color if cell == old_color else cell for cell in row] for row in grid]

def find_bounding_box(grid: List[List[int]], target_color: int) -> Optional[Tuple[int, int, int, int]]:
    """Find the bounding box (top, left, bottom, right) of a specific color."""
    rows, cols = len(grid), len(grid[0])
    top = bottom = left = right = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == target_color:
                if top is None:
                    top = bottom = r
                    left = right = c
                else:
                    bottom = max(bottom, r)
                    left = min(left, c)
                    right = max(right, c)
    return (top, left, bottom, right) if top is not None else None
