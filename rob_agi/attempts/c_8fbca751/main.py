from rob_agi.colored_grid import ColoredGrid
from typing import Set, Tuple, List

from typing import Set, Tuple, List

def solve_8fbca751(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by outlining all blue shapes with red.
    
    This function identifies all blue (8) regions in the input grid, including both
    contiguous areas and nearby disconnected cells, then outlines each logical shape
    with red (2) cells. The outline includes cells adjacent to blue cells (including
    diagonally) within a certain distance threshold. The outline does not extend
    beyond the grid boundaries or overwrite existing non-black cells. Each logical
    blue shape is enclosed in its own unified outline.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid with all blue shapes outlined in red.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    blue_cells = set()
    distance_threshold = 2

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def distance(cell1: Tuple[int, int], cell2: Tuple[int, int]) -> int:
        return max(abs(cell1[0] - cell2[0]), abs(cell1[1] - cell2[1]))

    def get_nearby_cells(r: int, c: int) -> Set[Tuple[int, int]]:
        nearby = set()
        for dr in range(-distance_threshold, distance_threshold + 1):
            for dc in range(-distance_threshold, distance_threshold + 1):
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc):
                    nearby.add((nr, nc))
        return nearby

    def find_shape(start: Tuple[int, int]) -> Set[Tuple[int, int]]:
        shape = set()
        queue = [start]
        while queue:
            cell = queue.pop(0)
            if cell in blue_cells:
                shape.add(cell)
                blue_cells.remove(cell)
                for nearby in get_nearby_cells(*cell):
                    if nearby in blue_cells and nearby not in shape:
                        queue.append(nearby)
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
            for nr, nc in get_nearby_cells(r, c):
                if (nr, nc) not in shape and grid.values[nr][nc] == 0:
                    outline.add((nr, nc))
        
        for or_, oc in outline:
            grid.values[or_][oc] = 2

    return grid
