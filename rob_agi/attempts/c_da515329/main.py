from rob_agi.colored_grid import ColoredGrid
import random
from collections import deque

def solve_da515329(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the da515329 challenge by transforming the input grid into an asymmetrical maze-like pattern.
    
    The solution involves the following steps:
    1. Analyze the input grid to find the central structure
    2. Create a new grid with a border frame
    3. Copy the central structure from the input grid
    4. Implement a growth algorithm to generate an asymmetrical pattern
    5. Ensure connectivity of all sky-colored pixels
    6. Balance the pattern across quadrants
    7. Add structural elements to large empty areas
    8. Perform final adjustments
    
    Args:
    input_grid (ColoredGrid): The input grid containing a central structure

    Returns:
    ColoredGrid: The transformed grid with an asymmetrical maze-like pattern
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    center = analyze_input(input_grid)
    create_border_frame(new_grid)
    copy_central_structure(new_grid, input_grid)
    growth_algorithm(new_grid, center)
    ensure_connectivity(new_grid, center)
    balance_pattern(new_grid)
    add_structural_elements(new_grid)
    final_adjustments(new_grid)
    
    return new_grid

def analyze_input(grid: ColoredGrid) -> tuple:
    rows, cols = grid.get_dimensions()
    sky_pixels = [(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 8]
    return (sum(r for r, _ in sky_pixels) // len(sky_pixels), sum(c for _, c in sky_pixels) // len(sky_pixels))

def create_border_frame(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        grid.values[r][-1] = 8  # Right border
    for c in range(cols):
        grid.values[-1][c] = 8  # Bottom border
    for c in range(1, cols):
        grid.values[0][c] = 8  # Top border (except top-left corner)

def copy_central_structure(new_grid: ColoredGrid, input_grid: ColoredGrid):
    rows, cols = new_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 8:
                new_grid.values[r][c] = 8

def growth_algorithm(grid: ColoredGrid, center: tuple):
    rows, cols = grid.get_dimensions()
    queue = deque([(r, c) for r in range(rows) for c in range(cols) if grid.values[r][c] == 8])
    visited = set(queue)
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                distance_factor = 1 - (((nr - center[0])**2 + (nc - center[1])**2)**0.5) / (rows + cols)
                density_factor = 1 - sum(grid.values[nr+i][nc+j] == 8 for i in [-1, 0, 1] for j in [-1, 0, 1] if 0 <= nr+i < rows and 0 <= nc+j < cols) / 9
                direction_bias = 0.1 if dr > 0 or dc > 0 else 0  # Slight bias towards bottom and right
                if random.random() < 0.3 * distance_factor + 0.3 * density_factor + 0.3 + direction_bias:
                    grid.values[nr][nc] = 8
                    queue.append((nr, nc))
                visited.add((nr, nc))

def ensure_connectivity(grid: ColoredGrid, center: tuple):
    rows, cols = grid.get_dimensions()
    visited = set()
    stack = [center]
    
    while stack:
        r, c = stack.pop()
        if (r, c) not in visited and grid.values[r][c] == 8:
            visited.add((r, c))
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    stack.append((nr, nc))
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8 and (r, c) not in visited:
                path = find_path_to_connected(grid, (r, c), visited)
                for pr, pc in path:
                    grid.values[pr][pc] = 8
                    visited.add((pr, pc))

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
    grid.values[0][0] = 0  # Ensure top-left corner is black
    rows, cols = grid.get_dimensions()
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if grid.values[r][c] == 0 and sum(grid.values[r+i][c+j] == 8 for i in [-1, 0, 1] for j in [-1, 0, 1]) >= 7:
                grid.values[r][c] = 8
