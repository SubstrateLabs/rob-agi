from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_712bf12e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by creating a minimal network of red lines.
    
    The function performs the following steps:
    1. Analyzes the input grid to identify red squares and gray squares.
    2. Creates vertical paths from the bottom red squares, avoiding gray squares when possible.
    3. Identifies the topmost connection row and creates primary horizontal connections.
    4. Ensures full connectivity by adding additional horizontal connections if needed.
    5. Minimizes the network by removing unnecessary red squares.
    6. Performs final checks and cleanup.
    
    The original gray squares are preserved throughout the process.
    
    Args:
    input_grid (ColoredGrid): The input grid with initial red and gray squares.
    
    Returns:
    ColoredGrid: The transformed grid with the minimal network of red lines added.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols
    
    def get_color(r, c):
        return output_grid.get_cell(r, c) if is_valid(r, c) else None
    
    def set_color(r, c, color):
        if is_valid(r, c):
            output_grid.set_cell(r, c, color)
    
    # Find starting points and create vertical paths
    start_points = [c for c in range(cols) if get_color(rows-1, c) == 2]
    for c in start_points:
        r = rows - 1
        while r >= 0:
            if get_color(r, c) != 5:
                set_color(r, c, 2)
                r -= 1
            else:
                if get_color(r, c+1) != 5:
                    c += 1
                elif get_color(r, c-1) != 5:
                    c -= 1
                else:
                    break
    
    # Find topmost connection row
    topmost_row = 0
    for r in range(rows):
        if all(any(get_color(r, c) == 2 for c in range(cols)) for sp in start_points):
            topmost_row = r
            break
    
    # Create primary horizontal connections
    for r in range(topmost_row, rows):
        red_squares = [c for c in range(cols) if get_color(r, c) == 2]
        for i in range(len(red_squares) - 1):
            start, end = red_squares[i], red_squares[i+1]
            for c in range(start+1, end):
                if get_color(r, c) == 0:
                    set_color(r, c, 2)
                elif get_color(r, c) == 5:
                    if get_color(r-1, c) == 0:
                        set_color(r-1, c, 2)
                    elif get_color(r+1, c) == 0:
                        set_color(r+1, c, 2)
    
    # Ensure full connectivity
    def is_connected():
        visited = set()
        stack = [(rows-1, start_points[0])]
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and get_color(r, c) == 2:
                visited.add((r, c))
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if is_valid(nr, nc):
                        stack.append((nr, nc))
        return len(visited) == sum(1 for r in range(rows) for c in range(cols) if get_color(r, c) == 2)
    
    while not is_connected():
        for r in range(rows):
            red_squares = [c for c in range(cols) if get_color(r, c) == 2]
            for i in range(len(red_squares) - 1):
                start, end = red_squares[i], red_squares[i+1]
                for c in range(start+1, end):
                    if get_color(r, c) == 0:
                        set_color(r, c, 2)
                    elif get_color(r, c) == 5:
                        if get_color(r-1, c) == 0:
                            set_color(r-1, c, 2)
                        elif get_color(r+1, c) == 0:
                            set_color(r+1, c, 2)
    
    # Minimize the network
    for r in range(rows):
        for c in range(cols):
            if get_color(r, c) == 2:
                set_color(r, c, 0)
                if not is_connected():
                    set_color(r, c, 2)
    
    # Final cleanup
    for r in range(rows):
        for c in range(cols):
            if get_color(r, c) == 2:
                neighbors = sum(1 for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)] if get_color(r+dr, c+dc) == 2)
                if neighbors <= 1 and (r != rows-1 or c not in start_points):
                    set_color(r, c, 0)
    
    return output_grid
