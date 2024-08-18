from typing import List, Tuple

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
    """Convert the grid to binary using a fixed threshold of 5."""
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

def scale(grid: List[List[int]], factor: int) -> List[List[int]]:
    """Scale the grid by a given factor."""
    return [[cell for cell in row for _ in range(factor)] for row in grid for _ in range(factor)]
