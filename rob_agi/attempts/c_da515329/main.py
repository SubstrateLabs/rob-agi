from rob_agi.colored_grid import ColoredGrid
import random
from collections import deque

def solve_da515329(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the da515329 challenge by transforming the input grid into an asymmetrical maze-like pattern.
    
    The solution involves the following steps:
    1. Analyze the input grid to find the central structure
    2. Create the outer frame
    3. Implement a growth algorithm to generate an asymmetrical pattern
    4. Ensure connectivity of all sky-colored pixels
    5. Balance the pattern across quadrants
    6. Add structural elements to large empty areas
    7. Perform final adjustments and connectivity checks
    
    Args:
    input_grid (ColoredGrid): The input grid containing a central structure

    Returns:
    ColoredGrid: The transformed grid with an asymmetrical maze-like pattern
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    center, bounding_box = analyze_input(input_grid)
    create_outer_frame(new_grid)
    copy_central_structure(new_grid, input_grid)
    growth_algorithm(new_grid, center, bounding_box)
    ensure_connectivity(new_grid, center)
    balance_pattern(new_grid)
    add_structural_elements(new_grid)
    final_adjustments(new_grid)
    
    return new_grid

def analyze_input(grid: ColoredGrid) -> tuple:
    rows, cols = grid.get_dimensions()
    sky_pixels = [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 8]
    center = (sum(r for r, _ in sky_pixels) // len(sky_pixels), sum(c for _, c in sky_pixels) // len(sky_pixels))
    min_r = min(r for r, _ in sky_pixels)
    max_r = max(r for r, _ in sky_pixels)
    min_c = min(c for _, c in sky_pixels)
    max_c = max(c for _, c in sky_pixels)
    return center, (min_r, min_c, max_r, max_c)

def create_outer_frame(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        grid.values[r][0] = grid.values[r][cols-1] = 8
    for c in range(1, cols-1):
        grid.values[0][c] = grid.values[rows-1][c] = 8
    grid.values[0][0] = 0  # Keep top-left corner black

def copy_central_structure(new_grid: ColoredGrid, input_grid: ColoredGrid):
    rows, cols = new_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 8:
                new_grid.values[r][c] = 8

def growth_algorithm(grid: ColoredGrid, center: tuple, bounding_box: tuple):
    rows, cols = grid.get_dimensions()
    queue = deque([(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 8])
    visited = set(queue)
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 < nr < rows-1 and 0 < nc < cols-1 and (nr, nc) not in visited:
                distance_factor = 1 - (((nr - center[0])**2 + (nc - center[1])**2)**0.5) / (rows + cols)
                density_factor = 1 - sum(grid.values[nr+i][nc+j] == 8 for i in [-1, 0, 1] for j in [-1, 0, 1]) / 9
                random_factor = random.random()
                if random_factor < 0.3 * distance_factor + 0.3 * density_factor + 0.4:
                    grid.values[nr][nc] = 8
                    queue.append((nr, nc))
                visited.add((nr, nc))

def ensure_connectivity(grid: ColoredGrid, center: tuple):
    rows, cols = grid.get_dimensions()
    visited = set()

    def dfs(r, c):
        if not (0 <= r < rows and 0 <= c < cols) or grid.values[r][c] == 0 or (r, c) in visited:
            return
        visited.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(r + dr, c + dc)

    dfs(center[0], center[1])

    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8 and (r, c) not in visited:
                path = find_path_to_connected(grid, (r, c), visited)
                for pr, pc in path:
                    grid.values[pr][pc] = 8
                    dfs(pr, pc)

def find_path_to_connected(grid: ColoredGrid, start: tuple, connected: set) -> list:
    rows, cols = grid.get_dimensions()
    queue = deque([(start, [])])
    visited = set()

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) in connected:
            return path
        visited.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                queue.append(((nr, nc), path + [(nr, nc)]))
    return []

def balance_pattern(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    quadrants = [
        (0, 0, rows//2, cols//2),
        (0, cols//2, rows//2, cols),
        (rows//2, 0, rows, cols//2),
        (rows//2, cols//2, rows, cols)
    ]
    densities = [sum(grid.values[r][c] == 8 for r in range(q[0], q[2]) for c in range(q[1], q[3])) / ((q[2]-q[0])*(q[3]-q[1])) for q in quadrants]
    avg_density = sum(densities) / 4

    for i, (top, left, bottom, right) in enumerate(quadrants):
        if densities[i] < avg_density * 0.8:
            for _ in range(int((avg_density - densities[i]) * (bottom-top) * (right-left) * 0.5)):
                r, c = random.randint(top+1, bottom-2), random.randint(left+1, right-2)
                if grid.values[r][c] == 0:
                    grid.values[r][c] = 8

def add_structural_elements(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for _ in range(rows * cols // 100):  # Add about 1% of grid size as structural elements
        r, c = random.randint(1, rows-3), random.randint(1, cols-3)
        if all(grid.values[r+i][c+j] == 0 for i in range(2) for j in range(2)):
            for i in range(2):
                for j in range(2):
                    grid.values[r+i][c+j] = 8

def final_adjustments(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    grid.values[0][0] = 0  # Ensure top-left corner is black
    ensure_connectivity(grid, (rows//2, cols//2))  # Final connectivity check
