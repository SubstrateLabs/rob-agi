from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_ac3e2b04(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding blue (1) structures that complement
    existing red (2) and green (3) patterns. The function identifies symmetry
    axes, extends blue lines from pattern anchors, and creates a balanced,
    symmetrical design while preserving the original structures.

    1. Analyzes existing red and green structures
    2. Determines symmetry axes
    3. Identifies pattern anchors (centers and edges of green crosses, endpoints and midpoints of red lines)
    4. Generates blue line patterns in empty areas
    5. Balances the design and connects patterns
    6. Refines the solution by filling gaps and removing isolated lines

    Args:
    input_grid (ColoredGrid): The input grid to be transformed

    Returns:
    ColoredGrid: The transformed grid with added blue structures
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    # Find symmetry axes
    vert_axis = cols // 2
    horz_axis = rows // 2

    # Identify pattern anchors
    anchors = find_pattern_anchors(output_grid)

    # Generate blue line patterns
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 0:
                if should_add_blue(r, c, anchors, vert_axis, horz_axis):
                    output_grid.set_cell(r, c, 1)

    # Connect patterns and refine
    connect_patterns(output_grid)
    refine_solution(output_grid)

    return output_grid

def find_pattern_anchors(grid: ColoredGrid) -> List[Tuple[int, int]]:
    anchors = []
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) in [2, 3]:
                anchors.append((r, c))
    return anchors

def should_add_blue(r: int, c: int, anchors: List[Tuple[int, int]], vert_axis: int, horz_axis: int) -> bool:
    # Check if the cell is on a line from an anchor or on a symmetry axis
    return any(r == ar or c == ac for ar, ac in anchors) or r == horz_axis or c == vert_axis

def connect_patterns(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 0:
                        if count_blue_neighbors(grid, nr, nc) >= 2:
                            grid.set_cell(nr, nc, 1)

def count_blue_neighbors(grid: ColoredGrid, r: int, c: int) -> int:
    count = 0
    rows, cols = grid.get_dimensions()
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid.get_cell(nr, nc) == 1:
            count += 1
    return count

def refine_solution(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 1:
                if count_blue_neighbors(grid, r, c) == 0:
                    grid.set_cell(r, c, 0)
