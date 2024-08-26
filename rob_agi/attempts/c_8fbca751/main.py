from rob_agi.colored_grid import ColoredGrid
from typing import Set, Tuple, List
from collections import deque

def solve_8fbca751(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by outlining all blue shapes with red.
    
    This function identifies all blue (8) regions in the input grid, groups them into
    logical shapes (including nearby disconnected cells within a two-cell distance),
    and outlines each shape with red (2) cells. The outline includes cells immediately
    adjacent to blue cells (including diagonally). The outline does not extend beyond
    the grid boundaries or overwrite existing non-black cells. Each logical blue shape
    is enclosed in its own unified outline, even if it consists of nearby but not
    directly connected blue cells.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid with all blue shapes outlined in red.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    blue_cells = set()

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_adjacent_cells(r: int, c: int, distance: int = 1) -> Set[Tuple[int, int]]:
        adjacent = set()
        for dr in range(-distance, distance + 1):
            for dc in range(-distance, distance + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc):
                    adjacent.add((nr, nc))
        return adjacent

    def find_shape(start: Tuple[int, int]) -> Set[Tuple[int, int]]:
        shape = set()
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            if cell in blue_cells:
                shape.add(cell)
                blue_cells.remove(cell)
                for adjacent in get_adjacent_cells(*cell, distance=2):
                    if adjacent in blue_cells and adjacent not in shape:
                        queue.append(adjacent)
                    elif grid.values[adjacent[0]][adjacent[1]] == 8 and adjacent not in shape:
                        queue.append(adjacent)
        return shape

    # Identify all blue cells
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8:
                blue_cells.add((r, c))

    # Group blue cells into logical shapes
    shapes = []
    while blue_cells:
        start = next(iter(blue_cells))
        shapes.append(find_shape(start))

    # Create and apply outlines
    for shape in shapes:
        outline = set()
        for r, c in shape:
            for nr, nc in get_adjacent_cells(r, c):
                if (nr, nc) not in shape and grid.values[nr][nc] == 0:
                    outline.add((nr, nc))
        
        for or_, oc in outline:
            grid.values[or_][oc] = 2

    return grid
