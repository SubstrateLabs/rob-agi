from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9c1e755f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by expanding edge patterns.
    
    This function identifies non-black patterns on the edges of the grid and
    expands them to fill the largest possible rectangle. The expansion respects
    existing non-black cells and grid boundaries. Both vertical and horizontal
    expansions are handled.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with expanded patterns.
    """
    grid = input_grid.deep_copy()
    edge_patterns = find_edge_patterns(grid)
    for edge, start, pattern in edge_patterns:
        expand_pattern(grid, edge, start, pattern)
    return grid

def find_edge_patterns(grid: ColoredGrid) -> List[Tuple[str, Tuple[int, int], List[int]]]:
    patterns = []
    rows, cols = grid.get_dimensions()

    # Check top and bottom edges
    for row in [0, rows - 1]:
        col = 0
        while col < cols:
            if grid.get_cell(row, col) != 0:
                start = col
                pattern = []
                while col < cols and grid.get_cell(row, col) != 0:
                    pattern.append(grid.get_cell(row, col))
                    col += 1
                edge = "top" if row == 0 else "bottom"
                patterns.append((edge, (row, start), pattern))
            else:
                col += 1

    # Check left and right edges
    for col in [0, cols - 1]:
        row = 0
        while row < rows:
            if grid.get_cell(row, col) != 0:
                start = row
                pattern = []
                while row < rows and grid.get_cell(row, col) != 0:
                    pattern.append(grid.get_cell(row, col))
                    row += 1
                edge = "left" if col == 0 else "right"
                patterns.append((edge, (start, col), pattern))
            else:
                row += 1

    return patterns

def expand_pattern(grid: ColoredGrid, edge: str, start: Tuple[int, int], pattern: List[int]):
    rows, cols = grid.get_dimensions()
    row, col = start

    if edge in ["top", "bottom"]:
        # Expand vertically
        direction = 1 if edge == "top" else -1
        max_height = 0
        for r in range(row, rows if direction == 1 else -1, direction):
            if all(grid.get_cell(r, c) == 0 for c in range(col, col + len(pattern))):
                max_height += 1
            else:
                break

        for r in range(row, row + max_height * direction, direction):
            for c in range(col, col + len(pattern)):
                grid.set_cell(r, c, pattern[c - col])

    else:  # left or right
        # Expand horizontally
        direction = 1 if edge == "left" else -1
        max_width = 0
        for c in range(col, cols if direction == 1 else -1, direction):
            if all(grid.get_cell(r, c) == 0 for r in range(row, row + len(pattern))):
                max_width += 1
            else:
                break

        for c in range(col, col + max_width * direction, direction):
            for r in range(row, row + len(pattern)):
                grid.set_cell(r, c, pattern[r - row])
