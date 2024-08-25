from rob_agi.colored_grid import ColoredGrid
from collections import deque
import random

def solve_f9d67f8b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by:
    1. Identifying and removing the largest edge-touching area.
    2. Expanding surrounding patterns into the removed area.
    3. Performing subtle redistributions and smoothing.

    Args:
        input_grid (ColoredGrid): The input grid to transform.

    Returns:
        ColoredGrid: The transformed grid with large edge-touching areas removed,
                     surrounding patterns extended, and subtle redistributions applied.
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = input_grid.deep_copy()
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    def flood_fill(r, c, color):
        area = []
        queue = deque([(r, c)])
        while queue:
            cr, cc = queue.popleft()
            if not visited[cr][cc] and new_grid.get_cell(cr, cc) == color:
                visited[cr][cc] = True
                area.append((cr, cc))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        queue.append((nr, nc))
        return area

    # Find the largest edge-touching area
    largest_area = []
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and not visited[r][c]:
                area = flood_fill(r, c, new_grid.get_cell(r, c))
                if len(area) > len(largest_area):
                    largest_area = area

    # If the largest area is significant, remove it
    if len(largest_area) > 0.1 * rows * cols:
        for r, c in largest_area:
            new_grid.set_cell(r, c, -1)  # Mark as empty

        # Expand surrounding patterns
        empty_cells = set(largest_area)
        while empty_cells:
            for r, c in list(empty_cells):
                neighbors = []
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and new_grid.get_cell(nr, nc) != -1:
                        neighbors.append(new_grid.get_cell(nr, nc))
                if neighbors:
                    new_grid.set_cell(r, c, random.choice(neighbors))
                    empty_cells.remove((r, c))

    # Perform subtle redistribution
    for r in range(rows):
        for c in range(cols):
            if random.random() < 0.05:  # 5% chance of change
                neighbors = []
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors.append(new_grid.get_cell(nr, nc))
                if neighbors:
                    new_grid.set_cell(r, c, random.choice(neighbors))

    # Final smoothing
    for r in range(rows):
        for c in range(cols):
            neighbors = []
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append(new_grid.get_cell(nr, nc))
            if len(set(neighbors)) == 1 and new_grid.get_cell(r, c) != neighbors[0]:
                new_grid.set_cell(r, c, neighbors[0])

    return new_grid
