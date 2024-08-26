from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, Set, List
import heapq

def solve_96a8c0cd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a minimal tree-like structure
    of red (2) cells connecting all non-black colored cells and extending to the grid edges.

    The algorithm follows these steps:
    1. Analyze the input grid to find all colored cells.
    2. Create a new grid, copying all non-black colored cells from the input.
    3. Implement a modified Prim's algorithm to create a minimal spanning tree connecting all colored cells.
    4. Extend the red structure to all four grid edges.
    5. Optimize the red structure by removing unnecessary red cells while maintaining connectivity.
    6. Perform a final verification to ensure all requirements are met.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the minimal red network connecting colored cells and extending to all edges.
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

    # Step 3: Implement modified Prim's algorithm
    def manhattan_distance(cell1: Tuple[int, int], cell2: Tuple[int, int]) -> int:
        return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])

    def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
        return [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                if 0 <= r+dr < rows and 0 <= c+dc < cols]

    start = colored_cells[0]
    visited = set([start])
    edges = [(manhattan_distance(start, neighbor), start, neighbor)
             for neighbor in colored_cells[1:]]
    heapq.heapify(edges)

    while edges:
        _, parent, current = heapq.heappop(edges)
        if current not in visited:
            visited.add(current)
            r, c = current
            while (r, c) != parent:
                if r < parent[0]: r += 1
                elif r > parent[0]: r -= 1
                elif c < parent[1]: c += 1
                elif c > parent[1]: c -= 1
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, 2)  # Set to red
            for neighbor in colored_cells:
                if neighbor not in visited:
                    heapq.heappush(edges, (manhattan_distance(current, neighbor), current, neighbor))

    # Step 4: Extend to all four edges
    def extend_to_edge(r: int, c: int, dr: int, dc: int):
        while 0 <= r < rows and 0 <= c < cols:
            if grid.get_cell(r, c) == 0:
                grid.set_cell(r, c, 2)
            r += dr
            c += dc

    for r in range(rows):
        extend_to_edge(r, 0, 0, 1)  # Left to right
        extend_to_edge(r, cols-1, 0, -1)  # Right to left
    for c in range(cols):
        extend_to_edge(0, c, 1, 0)  # Top to bottom
        extend_to_edge(rows-1, c, -1, 0)  # Bottom to top

    # Step 5: Optimize the red structure
    def is_critical(r: int, c: int) -> bool:
        if grid.get_cell(r, c) != 2:
            return False
        neighbors = [grid.get_cell(*n) for n in get_neighbors(r, c)]
        return sum(1 for n in neighbors if n > 0) > 2 or r in [0, rows-1] or c in [0, cols-1]

    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2 and not is_critical(r, c):
                grid.set_cell(r, c, 0)

    # Step 6: Final verification
    assert all(grid.get_cell(r, c) == input_grid.get_cell(r, c) for r, c in colored_cells), "Original colored cells not preserved"
    assert all(any(grid.get_cell(r, c) > 0 for r in range(rows)) for c in range(cols)), "Not all columns have colored cells"
    assert all(any(grid.get_cell(r, c) > 0 for c in range(cols)) for r in range(rows)), "Not all rows have colored cells"

    return grid
