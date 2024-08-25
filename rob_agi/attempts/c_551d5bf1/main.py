from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_551d5bf1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a sky blue (8) network that connects
    all blue (1) structures, fills enclosed areas with sky blue, and preserves
    the original blue structures.

    The function creates a minimal spanning tree connecting all blue structures,
    extends the network to the right and bottom edges, fills enclosed areas with
    sky blue, and ensures the original blue structures are preserved.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def find_nearest_tree_point(r, c, tree):
        queue = deque([(r, c)])
        visited = set()
        while queue:
            curr_r, curr_c = queue.popleft()
            if (curr_r, curr_c) in tree:
                return curr_r, curr_c
            visited.add((curr_r, curr_c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if is_valid(nr, nc) and (nr, nc) not in visited:
                    queue.append((nr, nc))
        return None

    def create_path(start_r, start_c, end_r, end_c):
        path = []
        r, c = start_r, start_c
        while (r, c) != (end_r, end_c):
            path.append((r, c))
            if r < end_r:
                r += 1
            elif r > end_r:
                r -= 1
            elif c < end_c:
                c += 1
            elif c > end_c:
                c -= 1
        path.append((end_r, end_c))
        return path

    def flood_fill(r, c):
        if not is_valid(r, c) or output_grid.values[r][c] != 0:
            return
        queue = deque([(r, c)])
        while queue:
            curr_r, curr_c = queue.popleft()
            if output_grid.values[curr_r][curr_c] != 0:
                continue
            output_grid.values[curr_r][curr_c] = 8
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if is_valid(nr, nc):
                    queue.append((nr, nc))

    # Identify blue structures
    blue_cells = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 1]
    blue_cells.sort()  # Sort from top to bottom, then left to right

    # Create minimal spanning tree
    tree = set([blue_cells[0]])
    for cell in blue_cells[1:]:
        nearest = find_nearest_tree_point(cell[0], cell[1], tree)
        path = create_path(nearest[0], nearest[1], cell[0], cell[1])
        for r, c in path:
            output_grid.values[r][c] = 8
            tree.add((r, c))

    # Extend to right edge
    rightmost = max(c for _, c in tree)
    for r in range(rows):
        for c in range(rightmost, cols):
            output_grid.values[r][c] = 8

    # Extend to bottom edge
    bottommost = max(r for r, _ in tree)
    for r in range(bottommost, rows):
        for c in range(cols):
            output_grid.values[r][c] = 8

    # Fill enclosed areas
    for r in range(rows):
        for c in range(cols):
            if output_grid.values[r][c] == 1:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    flood_fill(r + dr, c + dc)

    # Restore original blue cells
    for r, c in blue_cells:
        output_grid.values[r][c] = 1

    return output_grid
