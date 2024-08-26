from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_22a4bbc2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by changing qualifying lines to red.
    
    A qualifying line is a horizontal or vertical line of the same color,
    2-3 units long. When multiple qualifying lines are adjacent, they form
    larger red rectangles. The function identifies all such lines and
    changes them to red (color 2).
    """
    new_grid = input_grid.deep_copy()
    qualifying_lines = find_qualifying_lines(input_grid)
    cells_to_change = mark_cells_for_change(input_grid, qualifying_lines)
    
    for row, col in cells_to_change:
        new_grid.values[row][col] = 2
    
    return new_grid

def find_qualifying_lines(grid: ColoredGrid) -> List[Tuple[int, int, int, int, int]]:
    """
    Returns a list of tuples (row, col, length, direction, color)
    where direction is 0 for horizontal and 1 for vertical
    """
    rows, cols = grid.get_dimensions()
    qualifying_lines = []

    # Check horizontal lines
    for r in range(rows):
        for c in range(cols - 1):
            color = grid.values[r][c]
            if color == 0:  # Skip black (empty) cells
                continue
            length = 1
            while c + length < cols and grid.values[r][c + length] == color:
                length += 1
            if 2 <= length <= 3:
                qualifying_lines.append((r, c, length, 0, color))

    # Check vertical lines
    for c in range(cols):
        for r in range(rows - 1):
            color = grid.values[r][c]
            if color == 0:  # Skip black (empty) cells
                continue
            length = 1
            while r + length < rows and grid.values[r + length][c] == color:
                length += 1
            if 2 <= length <= 3:
                qualifying_lines.append((r, c, length, 1, color))

    return qualifying_lines

def mark_cells_for_change(grid: ColoredGrid, lines: List[Tuple[int, int, int, int, int]]) -> Set[Tuple[int, int]]:
    """
    Returns a set of (row, col) tuples representing cells to be changed to red
    """
    cells_to_change = set()
    for row, col, length, direction, color in lines:
        if direction == 0:  # Horizontal
            cells_to_change.update((row, col + i) for i in range(length))
        else:  # Vertical
            cells_to_change.update((row + i, col) for i in range(length))
    return cells_to_change
