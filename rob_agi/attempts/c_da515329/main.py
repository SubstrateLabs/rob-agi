from rob_agi.colored_grid import ColoredGrid
import random

def solve_da515329(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the da515329 challenge by transforming the input grid into a maze-like structure.
    
    The solution involves the following steps:
    1. Analyze the input grid to find the plus sign center and dimensions
    2. Initialize the output grid with the original plus sign
    3. Expand the central structure while maintaining symmetry
    4. Implement recursive division for maze generation
    5. Create a frame with characteristic gaps and protrusions
    6. Ensure connectivity and fill isolated areas
    7. Add final details and optimize based on grid size
    
    Args:
    input_grid (ColoredGrid): The input grid containing a plus sign

    Returns:
    ColoredGrid: The transformed grid with a maze-like structure
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    center, plus_width, plus_height = analyze_input(input_grid)
    copy_plus_sign(input_grid, new_grid)
    expand_central_structure(new_grid, center, plus_width, plus_height)
    recursive_division(new_grid, 0, 0, rows, cols, center, plus_width, plus_height)
    create_frame(new_grid)
    ensure_connectivity(new_grid)
    add_final_details(new_grid)
    
    return new_grid

def analyze_input(grid: ColoredGrid) -> tuple:
    rows, cols = grid.get_dimensions()
    center = None
    plus_width, plus_height = 0, 0
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8:
                if center is None:
                    center = (r, c)
                plus_height = max(plus_height, r - center[0] + 1)
                plus_width = max(plus_width, c - center[1] + 1)
    
    return center, plus_width * 2 - 1, plus_height * 2 - 1

def copy_plus_sign(input_grid: ColoredGrid, new_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 8:
                new_grid.values[r][c] = 8

def expand_central_structure(grid: ColoredGrid, center: tuple, width: int, height: int):
    rows, cols = grid.get_dimensions()
    for r in range(max(0, center[0] - height), min(rows, center[0] + height + 1)):
        for c in range(max(0, center[1] - width), min(cols, center[1] + width + 1)):
            if random.random() < 0.7:  # 70% chance to expand
                grid.values[r][c] = 8

def recursive_division(grid: ColoredGrid, x: int, y: int, width: int, height: int, center: tuple, plus_width: int, plus_height: int):
    if width < 4 or height < 4:
        return
    
    horizontal = random.choice([True, False]) if width != height else width < height
    
    if horizontal:
        divide_horizontally(grid, x, y, width, height, center, plus_width, plus_height)
    else:
        divide_vertically(grid, x, y, width, height, center, plus_width, plus_height)

def divide_horizontally(grid: ColoredGrid, x: int, y: int, width: int, height: int, center: tuple, plus_width: int, plus_height: int):
    divide_y = random.randint(y + 1, y + height - 2)
    passage = random.randint(x, x + width - 1)
    
    for i in range(x, x + width):
        if i != passage and not is_in_plus(i, divide_y, center, plus_width, plus_height):
            grid.values[divide_y][i] = 8
    
    recursive_division(grid, x, y, width, divide_y - y, center, plus_width, plus_height)
    recursive_division(grid, x, divide_y + 1, width, y + height - divide_y - 1, center, plus_width, plus_height)

def divide_vertically(grid: ColoredGrid, x: int, y: int, width: int, height: int, center: tuple, plus_width: int, plus_height: int):
    divide_x = random.randint(x + 1, x + width - 2)
    passage = random.randint(y, y + height - 1)
    
    for i in range(y, y + height):
        if i != passage and not is_in_plus(divide_x, i, center, plus_width, plus_height):
            grid.values[i][divide_x] = 8
    
    recursive_division(grid, x, y, divide_x - x, height, center, plus_width, plus_height)
    recursive_division(grid, divide_x + 1, y, x + width - divide_x - 1, height, center, plus_width, plus_height)

def is_in_plus(x: int, y: int, center: tuple, plus_width: int, plus_height: int) -> bool:
    return (center[0] - plus_height // 2 <= y <= center[0] + plus_height // 2 and
            center[1] - plus_width // 2 <= x <= center[1] + plus_width // 2)

def create_frame(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                if random.random() < 0.9:  # 90% chance to be part of the frame
                    grid.values[r][c] = 8
    
    # Add characteristic gaps
    for _ in range(4):
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            grid.values[0][random.randint(1, cols - 2)] = 0
        elif side == 'bottom':
            grid.values[rows - 1][random.randint(1, cols - 2)] = 0
        elif side == 'left':
            grid.values[random.randint(1, rows - 2)][0] = 0
        else:  # right
            grid.values[random.randint(1, rows - 2)][cols - 1] = 0

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
    
    # Fill in unvisited sky blue cells
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8 and (r, c) not in visited:
                grid.values[r][c] = 0

def add_final_details(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    # Ensure corners are empty
    grid.values[0][0] = grid.values[0][cols-1] = grid.values[rows-1][0] = grid.values[rows-1][cols-1] = 0
    
    # Add some random paths for larger grids
    if rows > 15 and cols > 15:
        for _ in range(rows * cols // 50):  # Add paths proportional to grid size
            r, c = random.randint(1, rows - 2), random.randint(1, cols - 2)
            if grid.values[r][c] == 0:
                grid.values[r][c] = 8
