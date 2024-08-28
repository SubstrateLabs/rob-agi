from typing import Set, Tuple, List
from rob_agi.colored_grid import ColoredGrid

def get_adjacent_cells(grid: ColoredGrid, row: int, col: int) -> List[Tuple[int, int]]:
    """Get orthogonally adjacent cells."""
    rows, cols = grid.get_dimensions()
    adjacent = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            adjacent.append((nr, nc))
    return adjacent

def solve_f9d67f8b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by:
    1. Identifying all brown (9) cells.
    2. Iteratively replacing brown cells with adjacent colors, prioritizing sky blue (8).
    3. If no sky blue is adjacent, use the first non-brown color found.
    4. If all adjacent cells are brown, keep the cell for the next iteration.
    5. Repeat until all brown cells are replaced.

    This approach ensures the expansion of existing patterns, particularly sky blue areas,
    while preserving the original structure by only modifying brown cells. It processes
    the grid in layers from the outside in, ensuring a consistent transformation.

    Args:
        input_grid (ColoredGrid): The input grid to transform.

    Returns:
        ColoredGrid: The transformed grid with brown cells replaced.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()
    brown_cells = set((r, c) for r in range(rows) for c in range(cols) if new_grid.get_cell(r, c) == 9)

    while brown_cells:
        next_brown_cells = set()
        for r, c in brown_cells:
            adjacent_cells = get_adjacent_cells(new_grid, r, c)
            adjacent_colors = [new_grid.get_cell(nr, nc) for nr, nc in adjacent_cells]
            
            if 8 in adjacent_colors:
                new_grid.set_cell(r, c, 8)
            elif any(color != 9 for color in adjacent_colors):
                new_color = next(color for color in adjacent_colors if color != 9)
                new_grid.set_cell(r, c, new_color)
            else:
                next_brown_cells.add((r, c))
        
        brown_cells = next_brown_cells

    return new_grid
