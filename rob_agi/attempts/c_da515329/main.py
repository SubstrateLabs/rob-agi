from rob_agi.colored_grid import ColoredGrid
import random

def solve_da515329(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the da515329 challenge by transforming the input grid into a maze-like structure.
    
    The solution involves the following steps:
    1. Find the center of the plus sign in the input grid
    2. Create a new grid and copy the original plus sign
    3. Generate a maze starting from the center of the plus sign
    4. Create an outer frame with gaps
    5. Connect the maze to the outer frame
    6. Fill some dead ends and create loops
    7. Make final adjustments for symmetry and connectivity
    
    Args:
    input_grid (ColoredGrid): The input grid containing a plus sign

    Returns:
    ColoredGrid: The transformed grid with a maze-like structure
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    # Find the center of the plus sign
    center = find_center(input_grid)
    
    # Copy the original plus sign
    copy_plus_sign(input_grid, new_grid)
    
    # Generate the maze
    generate_maze(new_grid, center)
    
    # Create outer frame
    create_outer_frame(new_grid)
    
    # Connect maze to frame
    connect_maze_to_frame(new_grid)
    
    # Fill dead ends and create loops
    fill_dead_ends(new_grid)
    
    # Final adjustments
    make_final_adjustments(new_grid)
    
    return new_grid

def find_center(grid: ColoredGrid) -> tuple:
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8:
                if (r > 0 and grid.values[r-1][c] == 8 and
                    r < rows-1 and grid.values[r+1][c] == 8 and
                    c > 0 and grid.values[r][c-1] == 8 and
                    c < cols-1 and grid.values[r][c+1] == 8):
                    return (r, c)
    return (rows // 2, cols // 2)  # Fallback to grid center

def copy_plus_sign(input_grid: ColoredGrid, new_grid: ColoredGrid):
    rows, cols = input_grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 8:
                new_grid.values[r][c] = 8

def generate_maze(grid: ColoredGrid, start: tuple):
    def is_valid(r, c):
        return 0 <= r < len(grid.values) and 0 <= c < len(grid.values[0])

    def get_neighbors(r, c):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        return [(r + dr, c + dc) for dr, dc in directions if is_valid(r + dr, c + dc)]

    visited = set()
    stack = [start]

    while stack:
        r, c = stack.pop()
        if (r, c) not in visited:
            visited.add((r, c))
            grid.values[r][c] = 8
            neighbors = get_neighbors(r, c)
            random.shuffle(neighbors)
            for nr, nc in neighbors:
                if (nr, nc) not in visited and grid.values[nr][nc] != 8:
                    stack.append((nr, nc))

def create_outer_frame(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                if random.random() < 0.9:  # 90% chance to be part of the frame
                    grid.values[r][c] = 8

def connect_maze_to_frame(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for _ in range(4):  # Ensure at least one connection on each side
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            c = random.randint(1, cols - 2)
            grid.values[0][c] = 8
            grid.values[1][c] = 8
        elif side == 'bottom':
            c = random.randint(1, cols - 2)
            grid.values[rows-1][c] = 8
            grid.values[rows-2][c] = 8
        elif side == 'left':
            r = random.randint(1, rows - 2)
            grid.values[r][0] = 8
            grid.values[r][1] = 8
        else:  # right
            r = random.randint(1, rows - 2)
            grid.values[r][cols-1] = 8
            grid.values[r][cols-2] = 8

def fill_dead_ends(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid.values[r][c] == 8:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if grid.values[r+dr][c+dc] == 8)
                if neighbors == 1 and random.random() < 0.5:  # 50% chance to extend dead end
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    random.shuffle(directions)
                    for dr, dc in directions:
                        if 0 <= r+dr < rows and 0 <= c+dc < cols and grid.values[r+dr][c+dc] == 0:
                            grid.values[r+dr][c+dc] = 8
                            break

def make_final_adjustments(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    # Ensure corners are empty
    grid.values[0][0] = grid.values[0][cols-1] = grid.values[rows-1][0] = grid.values[rows-1][cols-1] = 0
    
    # Add some random empty spaces for larger grids
    if rows > 15 and cols > 15:
        for _ in range(rows * cols // 100):  # Add empty spaces proportional to grid size
            r, c = random.randint(1, rows - 2), random.randint(1, cols - 2)
            grid.values[r][c] = 0
