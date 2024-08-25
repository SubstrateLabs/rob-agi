from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_af22c60d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling in black (0) areas with patterns
    extended from surrounding non-black cells.

    The solution follows these steps:
    1. Create a deep copy of the input grid.
    2. Identify all black cells and add them to a queue.
    3. Process the queue, filling black cells based on their non-black neighbors:
       - Prioritize orthogonal neighbors over diagonal ones.
       - If no neighbors are found, re-queue the cell for later processing.
    4. Repeat until all black cells are filled or no more changes can be made.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with black areas filled in.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def get_orthogonal_neighbors(r, c):
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                color = grid.get_cell(nr, nc)
                if color != 0:
                    neighbors.append(color)
        return neighbors

    def get_diagonal_neighbors(r, c):
        neighbors = []
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                color = grid.get_cell(nr, nc)
                if color != 0:
                    neighbors.append(color)
        return neighbors

    queue = deque()
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 0:
                queue.append((r, c))

    max_iterations = rows * cols * 2
    iterations = 0

    while queue and iterations < max_iterations:
        r, c = queue.popleft()
        if grid.get_cell(r, c) != 0:
            continue

        orthogonal_neighbors = get_orthogonal_neighbors(r, c)
        if orthogonal_neighbors:
            grid.set_cell(r, c, orthogonal_neighbors[0])
        else:
            diagonal_neighbors = get_diagonal_neighbors(r, c)
            if diagonal_neighbors:
                grid.set_cell(r, c, diagonal_neighbors[0])
            else:
                queue.append((r, c))

        iterations += 1

    # Fill any remaining black cells with the most common color
    if iterations == max_iterations:
        all_colors = [grid.get_cell(r, c) for r in range(rows) for c in range(cols) if grid.get_cell(r, c) != 0]
        most_common_color = max(set(all_colors), key=all_colors.count)
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, most_common_color)

    return grid
