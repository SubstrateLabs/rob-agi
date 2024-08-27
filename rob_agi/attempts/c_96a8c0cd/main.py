from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, Set, List
import heapq

def solve_96a8c0cd(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a minimal tree-like structure
    of red (2) cells connecting all non-black colored cells and extending to the grid edges.

    The algorithm follows these steps:
    1. Initialize the output grid by copying non-black cells from the input.
    2. Identify key points (colored cells and corners).
    3. Create a minimal spanning tree connecting all key points.
    4. Extend the red structure to all grid edges.
    5. Connect corners if not already connected.
    6. Optimize the red structure by removing unnecessary cells.
    7. Verify the solution meets all requirements.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with the minimal red network connecting colored cells and extending to all edges.
    """
    rows, cols = input_grid.get_dimensions()
    grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    # Step 1: Initialize output grid
    for r in range(rows):
        for c in range(cols):
            if input_grid.get_cell(r, c) > 0:
                grid.set_cell(r, c, input_grid.get_cell(r, c))

    # Step 2: Identify key points
    key_points = [(r, c) for r in range(rows) for c in range(cols) if grid.get_cell(r, c) > 0]
    key_points += [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
    key_points = list(set(key_points))  # Remove duplicates

    # Step 3: Create minimal spanning tree
    def manhattan_distance(cell1: Tuple[int, int], cell2: Tuple[int, int]) -> int:
        return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])

    def connect_points(start: Tuple[int, int], end: Tuple[int, int]):
        r, c = start
        while (r, c) != end:
            if grid.get_cell(r, c) == 0:
                grid.set_cell(r, c, 2)  # Set to red
            if r < end[0]: r += 1
            elif r > end[0]: r -= 1
            elif c < end[1]: c += 1
            elif c > end[1]: c -= 1

    connected = set([key_points[0]])
    while len(connected) < len(key_points):
        best_distance = float('inf')
        best_connection = None
        for point in connected:
            for candidate in key_points:
                if candidate not in connected:
                    distance = manhattan_distance(point, candidate)
                    if distance < best_distance:
                        best_distance = distance
                        best_connection = (point, candidate)
        connect_points(*best_connection)
        connected.add(best_connection[1])

    # Step 4: Extend to edges (already done in step 3)

    # Step 5: Connect corners (already done in step 3)

    # Step 6: Optimize red structure
    def is_critical(r: int, c: int) -> bool:
        if grid.get_cell(r, c) != 2:
            return False
        neighbors = [(r+dr, c+dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                     if 0 <= r+dr < rows and 0 <= c+dc < cols]
        colored_neighbors = sum(1 for nr, nc in neighbors if grid.get_cell(nr, nc) > 0)
        return colored_neighbors > 2 or (r, c) in [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]

    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 2 and not is_critical(r, c):
                grid.set_cell(r, c, 0)

    # Step 7: Verify solution
    assert all(grid.get_cell(r, c) == input_grid.get_cell(r, c) for r in range(rows) for c in range(cols) if input_grid.get_cell(r, c) > 0), "Original colored cells not preserved"
    assert all(any(grid.get_cell(r, c) > 0 for r in range(rows)) for c in range(cols)), "Not all columns have colored cells"
    assert all(any(grid.get_cell(r, c) > 0 for c in range(cols)) for r in range(rows)), "Not all rows have colored cells"
    assert all(grid.get_cell(0, c) > 0 for c in range(cols) if any(grid.get_cell(r, c) > 0 for r in range(rows))), "Not extended to top edge"
    assert all(grid.get_cell(rows-1, c) > 0 for c in range(cols) if any(grid.get_cell(r, c) > 0 for r in range(rows))), "Not extended to bottom edge"
    assert all(grid.get_cell(r, 0) > 0 for r in range(rows) if any(grid.get_cell(r, c) > 0 for c in range(cols))), "Not extended to left edge"
    assert all(grid.get_cell(r, cols-1) > 0 for r in range(rows) if any(grid.get_cell(r, c) > 0 for c in range(cols))), "Not extended to right edge"

    return grid
