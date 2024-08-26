from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_12eac192(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a green (3) path that connects blue (1) cells,
    incorporating gray (5) and sky (8) cells when convenient. The path aims to be simple,
    continuous, and snake-like, while preserving much of the original grid structure.

    1. Identifies blue cells and chooses a starting point near edges or corners.
    2. Creates a green path connecting blue cells, prioritizing corners and edges.
    3. Incorporates gray and sky cells when convenient.
    4. Ensures the path is continuous and simple, avoiding unnecessary complexity.
    5. Preserves orange (7) cells and maintains the overall structure of the original grid.
    6. Adapts behavior based on grid size, with more complete coverage for smaller grids.
    7. Ensures all blue cells are connected, even if it means creating a less snake-like path.
    8. Preserves blue cells that are not part of the main path.

    Args:
        input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
        ColoredGrid: The transformed grid with the green path.
    """
    def is_valid_cell(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if is_valid_cell(r+dr, c+dc)]

    def get_cell_priority(color: int) -> int:
        return {1: 5, 5: 4, 8: 3, 0: 2}.get(color, 1)

    def create_path(start: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [start]
        visited = set([start])
        current = start
        corners = set([(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)])

        while blue_cells - visited:
            neighbors = get_neighbors(*current)
            next_cell = max(
                neighbors,
                key=lambda x: (
                    x in blue_cells - visited,
                    x in corners,
                    get_cell_priority(grid[x[0]][x[1]]),
                    x not in visited
                )
            )
            if next_cell in visited:
                # If stuck, find the nearest unvisited blue cell
                unvisited_blue = blue_cells - visited
                if unvisited_blue:
                    next_cell = min(unvisited_blue, key=lambda x: abs(x[0]-current[0]) + abs(x[1]-current[1]))
                else:
                    break
            path.append(next_cell)
            visited.add(next_cell)
            current = next_cell
            corners.discard(next_cell)

        return path

    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()
    blue_cells = set((r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1)

    if not blue_cells:
        return grid

    start = min(blue_cells, key=lambda x: (-(x[0] in (0, rows-1) or x[1] in (0, cols-1)), x[0], x[1]))
    path = create_path(start)

    for r, c in path:
        if grid[r][c] not in [1, 7]:  # Preserve blue and orange cells
            grid.set_cell(r, c, 3)

    # Ensure all blue cells are connected
    for br, bc in blue_cells:
        if grid[br][bc] == 1:
            nearest_green = min((abs(br-r) + abs(bc-c), (r, c)) for r, c in path if grid[r][c] == 3)
            r, c = nearest_green[1]
            while (r, c) != (br, bc):
                if grid[r][c] not in [1, 3, 7]:
                    grid.set_cell(r, c, 3)
                r += 1 if br > r else -1 if br < r else 0
                c += 1 if bc > c else -1 if bc < c else 0

    return grid
