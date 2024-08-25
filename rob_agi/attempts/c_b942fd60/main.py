from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b942fd60(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting non-black squares with red lines.
    
    The function creates a minimal network of red lines that:
    1. Connects all non-black squares horizontally and vertically
    2. Preserves the original positions and colors of non-black squares
    3. Ensures red lines don't extend beyond the last colored square in any direction
    4. Handles both simple and complex grid configurations
    5. Cleans up any unnecessary or stray red lines
    
    Steps:
    1. Initialize by creating a deep copy and identifying non-black squares
    2. Connect non-black squares horizontally and vertically
    3. Connect isolated squares to the nearest part of the network
    4. Clean up unnecessary red lines
    5. Verify and fix connectivity
    6. Return the modified grid
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    non_black = [(r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) != 0]
    
    # Process rows
    for r in range(rows):
        row_squares = [c for c in range(cols) if (r, c) in non_black]
        if len(row_squares) >= 2:
            for c in range(row_squares[0], row_squares[-1] + 1):
                if output_grid.get_cell(r, c) == 0:
                    output_grid.set_cell(r, c, 2)
    
    # Process columns
    for c in range(cols):
        col_squares = [r for r in range(rows) if (r, c) in non_black]
        if len(col_squares) >= 2:
            for r in range(col_squares[0], col_squares[-1] + 1):
                if output_grid.get_cell(r, c) == 0:
                    output_grid.set_cell(r, c, 2)
    
    # Connect isolated squares
    for r, c in non_black:
        if all(output_grid.get_cell(r + dr, c + dc) == 0 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)] if 0 <= r + dr < rows and 0 <= c + dc < cols):
            # Find nearest red line or non-black square
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                while 0 <= nr < rows and 0 <= nc < cols:
                    if output_grid.get_cell(nr, nc) != 0:
                        break
                    output_grid.set_cell(nr, nc, 2)
                    nr, nc = nr + dr, nc + dc
    
    # Clean up unnecessary red lines
    def count_non_black_neighbors(r, c):
        return sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                   if 0 <= r + dr < rows and 0 <= c + dc < cols and output_grid.get_cell(r + dr, c + dc) != 0)
    
    changes = True
    while changes:
        changes = False
        for r in range(rows):
            for c in range(cols):
                if output_grid.get_cell(r, c) == 2 and count_non_black_neighbors(r, c) < 2:
                    output_grid.set_cell(r, c, 0)
                    changes = True
    
    # Trim extending red lines
    for r in range(rows):
        left = min((c for c in range(cols) if output_grid.get_cell(r, c) != 0 and output_grid.get_cell(r, c) != 2), default=-1)
        right = max((c for c in range(cols) if output_grid.get_cell(r, c) != 0 and output_grid.get_cell(r, c) != 2), default=-1)
        if left != -1 and right != -1:
            for c in range(cols):
                if c < left or c > right:
                    output_grid.set_cell(r, c, 0)
    
    for c in range(cols):
        top = min((r for r in range(rows) if output_grid.get_cell(r, c) != 0 and output_grid.get_cell(r, c) != 2), default=-1)
        bottom = max((r for r in range(rows) if output_grid.get_cell(r, c) != 0 and output_grid.get_cell(r, c) != 2), default=-1)
        if top != -1 and bottom != -1:
            for r in range(rows):
                if r < top or r > bottom:
                    output_grid.set_cell(r, c, 0)
    
    # Final check to ensure all non-black squares are connected
    def dfs(r, c):
        stack = [(r, c)]
        visited = set()
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited:
                visited.add((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and output_grid.get_cell(nr, nc) != 0:
                        stack.append((nr, nc))
        return visited
    
    connected = dfs(*non_black[0])
    if len(connected) != len(non_black):
        # If not all non-black squares are connected, add necessary connections
        for r, c in non_black:
            if (r, c) not in connected:
                nearest = min(connected, key=lambda x: abs(x[0] - r) + abs(x[1] - c))
                r_step = 1 if nearest[0] > r else -1 if nearest[0] < r else 0
                c_step = 1 if nearest[1] > c else -1 if nearest[1] < c else 0
                while (r, c) != nearest:
                    if output_grid.get_cell(r, c) == 0:
                        output_grid.set_cell(r, c, 2)
                    r += r_step
                    c += c_step
    
    return output_grid
