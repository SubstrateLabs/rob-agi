from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, Set, List
import heapq

def solve_96a8c0cd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a minimal tree-like structure
    of red (2) cells connecting all non-black colored cells. The algorithm follows these steps:

    1. Analyze the input grid to find all colored cells.
    2. Create a new grid, copying all non-black colored cells from the input.
    3. Implement Prim's algorithm to create a minimal spanning tree connecting all colored cells.
    4. Optimize the red structure by removing unnecessary red cells.
    5. Extend the red structure to the grid edges when necessary.
    6. Perform a final verification.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the minimal red network connecting colored cells.
    """
    rows, cols = input_grid.get_dimensions()
    grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    def is_colored(r: int, c: int) -> bool:
        return input_grid.get_cell(r, c) > 0

    # Step 1: Analyze the input grid
    colored_cells = [(r, c) for r in range(rows) for c in range(cols) if is_colored(r, c)]
    if not colored_cells:
        return input_grid

    # Step 2: Copy non-black colored cells
    for r, c in colored_cells:
        grid.set_cell(r, c, input_grid.get_cell(r, c))

    # Step 3: Implement Prim's algorithm
    def manhattan_distance(cell1: Tuple[int, int], cell2: Tuple[int, int]) -> int:
        return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    start = colored_cells[0]
    visited = set([start])
    edges = [(manhattan_distance(start, neighbor), start, neighbor)
             for neighbor in get_neighbors(*start)]
    heapq.heapify(edges)

    while edges:
        _, parent, current = heapq.heappop(edges)
        if current not in visited:
            visited.add(current)
            if not is_colored(*current):
                grid.set_cell(*current, 2)  # Set to red
            for neighbor in get_neighbors(*current):
                if neighbor not in visited:
                    heapq.heappush(edges, (manhattan_distance(current, neighbor), current, neighbor))

    # Step 4: Optimize the red structure
    def is_critical(r: int, c: int) -> bool:
        if grid.get_cell(r, c) != 2:
            return False
        neighbors = [grid.get_cell(*n) for n in get_neighbors(r, c)]
        return sum(1 for n in neighbors if n > 0) > 2

    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2 and not is_critical(r, c):
                grid.set_cell(r, c, 0)

    # Step 5: Extend to edges
    for r, c in colored_cells:
        if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
            if r > 0 and grid.get_cell(r-1, c) == 2:
                for rr in range(r-1, -1, -1):
                    grid.set_cell(rr, c, 2)
            if r < rows - 1 and grid.get_cell(r+1, c) == 2:
                for rr in range(r+1, rows):
                    grid.set_cell(rr, c, 2)
            if c > 0 and grid.get_cell(r, c-1) == 2:
                for cc in range(c-1, -1, -1):
                    grid.set_cell(r, cc, 2)
            if c < cols - 1 and grid.get_cell(r, c+1) == 2:
                for cc in range(c+1, cols):
                    grid.set_cell(r, cc, 2)

    # Step 6: Final verification
    for r, c in colored_cells:
        assert grid.get_cell(r, c) == input_grid.get_cell(r, c), "Original colored cell not preserved"

    return grid
