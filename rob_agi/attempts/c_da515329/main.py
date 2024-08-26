from rob_agi.colored_grid import ColoredGrid

def solve_da515329(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the da515329 challenge by transforming the input grid into a structured pattern.
    
    The solution involves the following steps:
    1. Analyze the input grid to find the central structure
    2. Initialize the output grid
    3. Generate a spiral pattern with openings
    4. Place and connect the central structure
    5. Apply edge and corner patterns
    6. Ensure connectivity of all sky-colored pixels
    7. Make final adjustments based on grid size
    
    Args:
    input_grid (ColoredGrid): The input grid containing a central structure

    Returns:
    ColoredGrid: The transformed grid with a structured spiral pattern
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    center, structure_width, structure_height = analyze_input(input_grid)
    generate_spiral(new_grid, center, structure_width, structure_height)
    place_central_structure(new_grid, input_grid, center)
    apply_edge_patterns(new_grid)
    ensure_connectivity(new_grid)
    final_adjustments(new_grid, input_grid)
    
    return new_grid

def analyze_input(grid: ColoredGrid) -> tuple:
    rows, cols = grid.get_dimensions()
    center = None
    structure_width, structure_height = 0, 0
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8:
                if center is None:
                    center = (r, c)
                structure_height = max(structure_height, abs(r - center[0]) * 2 + 1)
                structure_width = max(structure_width, abs(c - center[1]) * 2 + 1)
    
    return center, structure_width, structure_height

def generate_spiral(grid: ColoredGrid, center: tuple, structure_width: int, structure_height: int):
    rows, cols = grid.get_dimensions()
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    r, c = 1, 1  # Start from (1,1) to keep (0,0) black
    dir_index = 0
    steps = 0
    max_steps = min(rows, cols) - 2

    while steps < max_steps:
        grid.values[r][c] = 8
        
        # Check if we need to change direction
        next_r, next_c = r + directions[dir_index][0], c + directions[dir_index][1]
        if (next_r < 1 or next_r >= rows - 1 or next_c < 1 or next_c >= cols - 1 or
            grid.values[next_r][next_c] == 8 or
            (abs(next_r - center[0]) < structure_height // 2 and abs(next_c - center[1]) < structure_width // 2)):
            dir_index = (dir_index + 1) % 4
            steps += 1
        
        r, c = r + directions[dir_index][0], c + directions[dir_index][1]
        
        # Create openings
        if steps % 3 == 0:
            grid.values[r][c] = 0

def place_central_structure(new_grid: ColoredGrid, input_grid: ColoredGrid, center: tuple):
    rows, cols = new_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 8:
                new_grid.values[r][c] = 8
    
    # Connect central structure to spiral
    r, c = center
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        while 0 <= nr < rows and 0 <= nc < cols:
            if new_grid.values[nr][nc] == 8:
                break
            new_grid.values[nr][nc] = 8
            nr, nc = nr + dr, nc + dc

def apply_edge_patterns(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    
    # Top and bottom edges
    for c in range(2, cols - 2, 3):
        grid.values[0][c] = grid.values[rows-1][c] = 8
    
    # Left and right edges
    for r in range(2, rows - 2, 3):
        grid.values[r][0] = grid.values[r][cols-1] = 8
    
    # Corners
    grid.values[0][0] = grid.values[0][1] = grid.values[1][0] = 0
    grid.values[0][cols-1] = grid.values[0][cols-2] = grid.values[1][cols-1] = 8
    grid.values[rows-1][0] = grid.values[rows-2][0] = grid.values[rows-1][1] = 8
    grid.values[rows-1][cols-1] = 8

def ensure_connectivity(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    visited = set()

    def dfs(r, c):
        if not (0 <= r < rows and 0 <= c < cols) or grid.values[r][c] == 0 or (r, c) in visited:
            return
        visited.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(r + dr, c + dc)

    # Start DFS from the center
    center = (rows // 2, cols // 2)
    dfs(center[0], center[1])

    # Remove unconnected sky blue pixels
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8 and (r, c) not in visited:
                grid.values[r][c] = 0

def final_adjustments(new_grid: ColoredGrid, input_grid: ColoredGrid):
    rows, cols = new_grid.get_dimensions()
    
    # Ensure the original structure is preserved
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 8:
                new_grid.values[r][c] = 8

    # Add isolated pixels if needed
    isolated_pixels = [
        (1, cols // 4), (2, cols // 3),
        (rows // 4, 1), (rows // 3, 2),
        (rows - 3, cols - 3), (rows - 4, cols - 4)
    ]
    for r, c in isolated_pixels:
        if 0 <= r < rows and 0 <= c < cols:
            new_grid.values[r][c] = 8

    # Ensure connectivity after adding isolated pixels
    ensure_connectivity(new_grid)
