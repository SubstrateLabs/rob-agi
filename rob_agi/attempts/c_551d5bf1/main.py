from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_551d5bf1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a sky blue (8) network that connects
    all blue (1) structures, fills enclosed areas with sky blue, and preserves
    the original blue structures.

    The function creates an initial sky blue network through the middle of the grid,
    connects all blue frames to this network, fills enclosed areas, and ensures
    the original blue structures are preserved.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    middle_row = rows // 2

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def extend_sky_blue(r, c, dr, dc):
        while is_valid(r + dr, c + dc) and output_grid.values[r + dr][c + dc] != 1:
            r += dr
            c += dc
            output_grid.values[r][c] = 8
        return r, c

    def connect_to_network(r, c):
        queue = deque([(r, c)])
        visited = set()
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            if output_grid.values[r][c] == 8:
                return
            output_grid.values[r][c] = 8
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc):
                    queue.append((nr, nc))

    def flood_fill(r, c):
        if output_grid.values[r][c] != 0:
            return
        queue = deque([(r, c)])
        while queue:
            r, c = queue.popleft()
            if not is_valid(r, c) or output_grid.values[r][c] != 0:
                continue
            output_grid.values[r][c] = 8
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                queue.append((r + dr, c + dc))

    # Create initial sky blue network
    for col in range(cols):
        output_grid.values[middle_row][col] = 8
        extend_sky_blue(middle_row, col, -1, 0)
        extend_sky_blue(middle_row, col, 1, 0)

    # Connect blue frames to the network and fill enclosed areas
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 1:
                connect_to_network(r, c)
            elif output_grid.values[r][c] == 8:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    flood_fill(r + dr, c + dc)

    # Preserve original blue structures
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 1:
                output_grid.values[r][c] = 1

    return output_grid
