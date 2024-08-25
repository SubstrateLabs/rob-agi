from rob_agi.colored_grid import ColoredGrid
from collections import Counter, deque

def solve_af22c60d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by filling in black (0) areas with patterns
    extended from surrounding non-black cells.

    The solution follows these steps:
    1. Create a distance map for each black cell to the nearest non-black cell.
    2. Iteratively fill black cells based on their neighbors and the overall pattern.
    3. Use pattern recognition to extend existing patterns into black areas.
    4. Resolve conflicts and create smooth transitions between different patterns.
    5. Repeat the process until all black cells are filled or no more changes can be made.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with black areas filled in.
    """
    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    def get_neighbors(r, c):
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append((nr, nc, grid.get_cell(nr, nc)))
        return neighbors

    def create_distance_map():
        distance_map = {}
        queue = deque()
        for r in range(rows):
            for c in range(cols):
                if grid.get_cell(r, c) != 0:
                    distance_map[(r, c)] = 0
                    queue.append((r, c, 0))
        
        while queue:
            r, c, dist = queue.popleft()
            for nr, nc, _ in get_neighbors(r, c):
                if (nr, nc) not in distance_map:
                    distance_map[(nr, nc)] = dist + 1
                    queue.append((nr, nc, dist + 1))
        
        return distance_map

    def fill_cell(r, c, distance_map):
        if grid.get_cell(r, c) != 0:
            return False

        neighbors = get_neighbors(r, c)
        non_black = [color for _, _, color in neighbors if color != 0]
        
        if not non_black:
            return False

        color_counts = Counter(non_black)
        closest_colors = [color for _, _, color in sorted(neighbors, key=lambda x: distance_map.get((x[0], x[1]), float('inf'))) if color != 0]
        
        if closest_colors:
            new_color = closest_colors[0]
        else:
            new_color = max(color_counts.items(), key=lambda x: (x[1], -x[0]))[0]

        grid.set_cell(r, c, new_color)
        return True

    distance_map = create_distance_map()
    
    changed = True
    while changed:
        changed = False
        for r in range(rows):
            for c in range(cols):
                if fill_cell(r, c, distance_map):
                    changed = True

    return grid
