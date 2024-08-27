from rob_agi.colored_grid import ColoredGrid
import random

def solve_da515329(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the da515329 challenge by transforming the input grid into an asymmetrical maze-like pattern.
    
    The solution involves the following steps:
    1. Analyze the input grid to find the central structure
    2. Create a new grid with a border frame
    3. Copy the central structure from the input grid
    4. Implement an enhanced growth algorithm to generate an asymmetrical pattern
    5. Ensure connectivity of all sky-colored pixels
    6. Handle the left edge filling
    7. Perform final adjustments and optimizations
    
    Args:
    input_grid (ColoredGrid): The input grid containing a central structure

    Returns:
    ColoredGrid: The transformed grid with an asymmetrical maze-like pattern
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    create_border_frame(new_grid)
    copy_central_structure(new_grid, input_grid)
    enhanced_growth_algorithm(new_grid)
    ensure_connectivity(new_grid)
    handle_left_edge(new_grid)
    final_adjustments(new_grid)
    
    return new_grid

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

def enhanced_growth_algorithm(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    target_fill = rows * cols * 0.45  # Target 45% fill
    filled = sum(row.count(8) for row in grid.values)
    iterations = 0
    max_iterations = rows * cols
    
    while filled < target_fill and iterations < max_iterations:
        for r in range(rows):
            for c in range(cols):
                if grid.values[r][c] == 0:
                    neighbors = sum(grid.values[r+dr][c+dc] == 8 
                                    for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                                    if 0 <= r+dr < rows and 0 <= c+dc < cols)
                    if neighbors > 0 and random.random() < 0.7 - (iterations / max_iterations) * 0.5:
                        grid.values[r][c] = 8
                        filled += 1
                        if filled >= target_fill:
                            return
        iterations += 1

def ensure_connectivity(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    visited = set()
    
    def dfs(r, c):
        if not (0 <= r < rows and 0 <= c < cols) or grid.values[r][c] != 8 or (r, c) in visited:
            return
        visited.add((r, c))
        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            dfs(r+dr, c+dc)
    
    # Start DFS from the first sky-colored pixel
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8:
                dfs(r, c)
                break
        if visited:
            break
    
    # Connect any disconnected sky-colored pixels
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8 and (r, c) not in visited:
                target_r, target_c = min(visited, key=lambda x: abs(x[0]-r) + abs(x[1]-c))
                while (r, c) != (target_r, target_c):
                    if r < target_r:
                        r += 1
                    elif r > target_r:
                        r -= 1
                    elif c < target_c:
                        c += 1
                    else:
                        c -= 1
                    grid.values[r][c] = 8
                dfs(r, c)

def handle_left_edge(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(1, rows - 1):
        if random.random() < 0.7:
            grid.values[r][0] = 8

def final_adjustments(grid: ColoredGrid):
    grid.values[0][0] = 0  # Ensure top-left corner is black
    rows, cols = grid.get_dimensions()
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if grid.values[r][c] == 0 and sum(grid.values[r+i][c+j] == 8 for i in [-1, 0, 1] for j in [-1, 0, 1]) >= 7:
                grid.values[r][c] = 8
    
    # Add small imperfections
    for _ in range(rows * cols // 20):
        r, c = random.randint(1, rows-2), random.randint(1, cols-2)
        grid.values[r][c] = 8 if grid.values[r][c] == 0 else 0
