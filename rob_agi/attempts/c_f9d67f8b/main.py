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
    4. Repeat until all brown cells are replaced.

    This approach ensures the expansion of existing patterns, particularly sky blue areas,
    while preserving the original structure by only modifying brown cells.

    Args:
        input_grid (ColoredGrid): The input grid to transform.

    Returns:
        ColoredGrid: The transformed grid with brown cells replaced.
    """
    new_grid = input_grid.deep_copy()
    rows, cols = new_grid.get_dimensions()
    brown_cells = set((r, c) for r in range(rows) for c in range(cols) if new_grid.get_cell(r, c) == 9)

    if not brown_cells:
        return new_grid  # Return unchanged if no brown cells

    while brown_cells:
        update_list = []
        for r, c in brown_cells:
            adjacent_cells = get_adjacent_cells(new_grid, r, c)
            new_color = next((new_grid.get_cell(nr, nc) for nr, nc in adjacent_cells if new_grid.get_cell(nr, nc) == 8), None)
            if new_color is None:
                new_color = next((new_grid.get_cell(nr, nc) for nr, nc in adjacent_cells if new_grid.get_cell(nr, nc) != 9), None)
            if new_color is not None:
                update_list.append((r, c, new_color))
        
        for r, c, new_color in update_list:
            new_grid.set_cell(r, c, new_color)
            brown_cells.remove((r, c))

    return new_grid
