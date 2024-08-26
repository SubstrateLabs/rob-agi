from rob_agi.colored_grid import ColoredGrid
from typing import Set, Tuple, List

def solve_8fbca751(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by outlining all blue shapes with red.
    
    This function identifies all contiguous blue (8) regions in the input grid,
    then outlines each region separately with red (2) cells. The outline includes
    diagonally adjacent cells but does not extend beyond the grid boundaries or
    overwrite existing non-black cells. Each blue shape is enclosed in its own outline.
    
    Args:
        input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
        ColoredGrid: The transformed grid with all blue shapes outlined in red.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    visited = set()

    def flood_fill(r: int, c: int) -> Set[Tuple[int, int]]:
        stack = [(r, c)]
        region = set()
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and 0 <= r < rows and 0 <= c < cols and grid.values[r][c] == 8:
                visited.add((r, c))
                region.add((r, c))
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        stack.append((r + dr, c + dc))
        return region

    def get_outline(region: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        outline = set()
        for r, c in region:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in region and 0 <= nr < rows and 0 <= nc < cols:
                        outline.add((nr, nc))
        return outline

    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8 and (r, c) not in visited:
                blue_region = flood_fill(r, c)
                outline = get_outline(blue_region)
                for or_, oc in outline:
                    if grid.values[or_][oc] == 0:  # Only change black cells to red
                        grid.values[or_][oc] = 2

    return grid
