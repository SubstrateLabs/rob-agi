from rob_agi.colored_grid import ColoredGrid

def solve_da515329(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the da515329 challenge by transforming the input grid into a structured pattern.
    
    The solution involves the following steps:
    1. Analyze the input grid to find the central structure
    2. Create the outermost frame
    3. Generate concentric frames with gaps
    4. Integrate the central structure
    5. Apply specific patterns (corners, edges)
    6. Ensure connectivity of all sky-colored pixels
    7. Balance and symmetry check
    8. Final adjustments
    
    Args:
    input_grid (ColoredGrid): The input grid containing a central structure

    Returns:
    ColoredGrid: The transformed grid with a structured concentric pattern
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    center, structure_width, structure_height = analyze_input(input_grid)
    create_outermost_frame(new_grid)
    generate_concentric_frames(new_grid, center, structure_width, structure_height)
    integrate_central_structure(new_grid, input_grid, center)
    apply_specific_patterns(new_grid)
    ensure_connectivity(new_grid)
    balance_and_symmetry_check(new_grid)
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

def create_outermost_frame(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        grid.values[r][0] = grid.values[r][cols-1] = 8
    for c in range(1, cols-1):
        grid.values[0][c] = grid.values[rows-1][c] = 8
    grid.values[0][0] = 0  # Keep top-left corner black

def generate_concentric_frames(grid: ColoredGrid, center: tuple, structure_width: int, structure_height: int):
    rows, cols = grid.get_dimensions()
    frame_count = min((rows - structure_height) // 2, (cols - structure_width) // 2)
    
    for frame in range(1, frame_count):
        for r in range(frame, rows-frame):
            if r == frame or r == rows-frame-1:
                for c in range(frame, cols-frame):
                    if (r + c) % 3 != 0:  # Create gaps
                        grid.values[r][c] = 8
            else:
                grid.values[r][frame] = grid.values[r][cols-frame-1] = 8
        
        # Ensure corners are always filled
        grid.values[frame][frame] = grid.values[frame][cols-frame-1] = 8
        grid.values[rows-frame-1][frame] = grid.values[rows-frame-1][cols-frame-1] = 8

def integrate_central_structure(new_grid: ColoredGrid, input_grid: ColoredGrid, center: tuple):
    rows, cols = new_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 8:
                new_grid.values[r][c] = 8
    
    # Connect central structure to innermost frame
    r, c = center
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        while 0 <= nr < rows and 0 <= nc < cols:
            if new_grid.values[nr][nc] == 8:
                break
            new_grid.values[nr][nc] = 8
            nr, nc = nr + dr, nc + dc

def apply_specific_patterns(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    
    # Corner patterns
    grid.values[1][1] = grid.values[1][2] = grid.values[2][1] = 8
    grid.values[1][cols-2] = grid.values[1][cols-3] = grid.values[2][cols-2] = 8
    grid.values[rows-2][1] = grid.values[rows-3][1] = grid.values[rows-2][2] = 8
    grid.values[rows-2][cols-2] = grid.values[rows-3][cols-2] = grid.values[rows-2][cols-3] = 8
    
    # Edge patterns
    for i in range(4, rows-4, 4):
        grid.values[i][0] = grid.values[i][cols-1] = 8
    for j in range(4, cols-4, 4):
        grid.values[0][j] = grid.values[rows-1][j] = 8

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

def balance_and_symmetry_check(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    
    # Horizontal symmetry check and fix
    for r in range(rows // 2):
        for c in range(cols):
            if grid.values[r][c] != grid.values[rows-1-r][c]:
                grid.values[r][c] = grid.values[rows-1-r][c] = max(grid.values[r][c], grid.values[rows-1-r][c])
    
    # Vertical symmetry check and fix
    for c in range(cols // 2):
        for r in range(rows):
            if grid.values[r][c] != grid.values[r][cols-1-c]:
                grid.values[r][c] = grid.values[r][cols-1-c] = max(grid.values[r][c], grid.values[r][cols-1-c])

def final_adjustments(new_grid: ColoredGrid, input_grid: ColoredGrid):
    rows, cols = new_grid.get_dimensions()
    
    # Ensure the original structure is preserved
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 8:
                new_grid.values[r][c] = 8

    # Ensure top-left corner is black
    new_grid.values[0][0] = 0

    # Ensure connectivity after all adjustments
    ensure_connectivity(new_grid)
