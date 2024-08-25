from rob_agi.colored_grid import ColoredGrid
from collections import deque, Counter

def solve_f9d67f8b(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by removing large edge-touching areas
    and extending the surrounding pattern.

    1. Identify colors to be removed based on edge presence.
    2. Mark cells for removal using a flood fill algorithm.
    3. Fill removed areas by extending surrounding patterns.

    Args:
        input_grid (ColoredGrid): The input grid to transform.

    Returns:
        ColoredGrid: The transformed grid with large edge-touching areas removed
                     and surrounding patterns extended.
    """
    rows, cols = input_grid.get_dimensions()

    def get_edge_colors():
        edge_colors = set()
        for r in range(rows):
            edge_colors.add(input_grid.get_cell(r, 0))
            edge_colors.add(input_grid.get_cell(r, cols - 1))
        for c in range(cols):
            edge_colors.add(input_grid.get_cell(0, c))
            edge_colors.add(input_grid.get_cell(rows - 1, c))
        return edge_colors

    def count_edge_occurrences(color):
        count = 0
        for r in range(rows):
            if input_grid.get_cell(r, 0) == color:
                count += 1
            if input_grid.get_cell(r, cols - 1) == color:
                count += 1
        for c in range(cols):
            if input_grid.get_cell(0, c) == color:
                count += 1
            if input_grid.get_cell(rows - 1, c) == color:
                count += 1
        return count

    edge_colors = get_edge_colors()
    color_counts = {color: count_edge_occurrences(color) for color in edge_colors}
    max_count = max(color_counts.values())
    colors_to_remove = {color for color, count in color_counts.items() if count == max_count}

    to_remove = [[False for _ in range(cols)] for _ in range(rows)]
    queue = deque()

    # Mark edge cells for removal
    for r in range(rows):
        if input_grid.get_cell(r, 0) in colors_to_remove:
            to_remove[r][0] = True
            queue.append((r, 0))
        if input_grid.get_cell(r, cols - 1) in colors_to_remove:
            to_remove[r][cols - 1] = True
            queue.append((r, cols - 1))
    for c in range(cols):
        if input_grid.get_cell(0, c) in colors_to_remove:
            to_remove[0][c] = True
            queue.append((0, c))
        if input_grid.get_cell(rows - 1, c) in colors_to_remove:
            to_remove[rows - 1][c] = True
            queue.append((rows - 1, c))

    # Flood fill to mark all connected cells for removal
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not to_remove[nr][nc]:
                if input_grid.get_cell(nr, nc) in colors_to_remove:
                    to_remove[nr][nc] = True
                    queue.append((nr, nc))

    # Create a new grid and fill removed areas
    new_grid = input_grid.deep_copy()
    for r in range(rows):
        for c in range(cols):
            if to_remove[r][c]:
                fill_queue = deque([(r, c)])
                while fill_queue:
                    fr, fc = fill_queue.popleft()
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = fr + dr, fc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if not to_remove[nr][nc]:
                                new_grid.set_cell(fr, fc, new_grid.get_cell(nr, nc))
                                break
                            elif to_remove[nr][nc]:
                                fill_queue.append((nr, nc))

    return new_grid
