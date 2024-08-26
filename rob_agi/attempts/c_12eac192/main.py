from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set
from collections import deque

def solve_12eac192(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating a green (3) path that connects blue (1) cells,
    incorporating gray (5) and sky (8) cells when convenient. The path aims to be simple,
    continuous, and snake-like, while preserving much of the original grid structure.

    1. Identifies blue cells and chooses a starting point near edges or corners.
    2. Uses a modified BFS to create a green path, prioritizing blue > gray > sky > black cells.
    3. Favors continuing in the current direction and reaching towards edges and corners.
    4. Limits path complexity and length to maintain simplicity.
    5. Connects remaining blue cells if they are close to the main path.
    6. Ensures the final green path is continuous and orange (7) cells remain unchanged.
    7. Maintains the overall structure of the original grid.

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
        return {1: 3, 5: 2, 8: 1, 0: 0}.get(color, -1)

    def bfs(start: Tuple[int, int]) -> List[Tuple[int, int]]:
        queue = deque([start])
        path = []
        visited = set()
        current_direction = None
        complexity = 0
        max_complexity = min(rows, cols) * 2

        while queue and complexity < max_complexity:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            path.append((r, c))

            neighbors = get_neighbors(r, c)
            neighbors.sort(key=lambda x: (
                -get_cell_priority(grid[x[0]][x[1]]),
                0 if current_direction and (x[0]-r, x[1]-c) == current_direction else 1,
                -(x[0] in (0, rows-1) or x[1] in (0, cols-1))
            ))

            for nr, nc in neighbors:
                if (nr, nc) not in visited and grid[nr][nc] != 7:
                    queue.append((nr, nc))
                    if current_direction and (nr-r, nc-c) != current_direction:
                        complexity += 1
                    current_direction = (nr-r, nc-c)
                    break

        return path

    grid = input_grid.deep_copy()
    rows, cols = grid.get_dimensions()

    blue_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    if not blue_cells:
        return grid

    # Choose starting point
    start = max(blue_cells, key=lambda x: (x[0] in (0, rows-1) or x[1] in (0, cols-1), 
                                           sum(1 for nr, nc in get_neighbors(*x) if grid[nr][nc] not in [1, 7])))

    # Create main path
    main_path = bfs(start)

    # Convert main path to green
    for r, c in main_path:
        if grid[r][c] in [0, 1, 5, 8]:
            grid.set_cell(r, c, 3)

    # Connect remaining blue cells
    for br, bc in blue_cells:
        if grid[br][bc] == 1:
            nearest_green = min((abs(br-r) + abs(bc-c), (r, c)) for r, c in main_path if grid[r][c] == 3)
            if nearest_green[0] <= 3:
                r, c = nearest_green[1]
                while (r, c) != (br, bc):
                    if grid[r][c] in [0, 1, 5, 8]:
                        grid.set_cell(r, c, 3)
                    r += 1 if br > r else -1 if br < r else 0
                    c += 1 if bc > c else -1 if bc < c else 0

    return grid
