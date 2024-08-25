from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b942fd60(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting non-black squares with red lines.
    
    The function creates a minimal network of red lines that:
    1. Creates a "backbone" structure with optimal vertical and horizontal lines
    2. Connects all non-black squares to the backbone
    3. Preserves the original positions and colors of non-black squares
    4. Ensures red lines don't extend beyond the last colored square in any direction
    5. Cleans up any unnecessary or stray red lines
    
    Steps:
    1. Create a deep copy of the input grid
    2. Identify all non-black squares
    3. Find optimal vertical and horizontal lines for the backbone
    4. Create the backbone structure
    5. Connect remaining non-black squares to the backbone
    6. Clean up unnecessary extensions
    7. Verify connectivity and add necessary connections
    8. Remove isolated red squares
    9. Return the modified grid
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Identify non-black squares
    non_black = [(r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) != 0]
    
    if not non_black:
        return output_grid  # Return original grid if no non-black squares
    
    # Find optimal vertical line
    col_counts = [sum(1 for r in range(rows) if output_grid.get_cell(r, c) != 0) for c in range(cols)]
    optimal_col = col_counts.index(max(col_counts))
    
    # Find optimal horizontal line
    row_counts = [sum(1 for c in range(cols) if output_grid.get_cell(r, c) != 0) for r in range(rows)]
    optimal_row = row_counts.index(max(row_counts))
    
    # Adjust optimal_row if it doesn't intersect with any non-black square
    if all(output_grid.get_cell(optimal_row, c) == 0 for c in range(cols)):
        optimal_row = min(non_black, key=lambda x: abs(x[0] - optimal_row))[0]
    
    # Create backbone structure
    for r in range(rows):
        output_grid.set_cell(r, optimal_col, 2)
    for c in range(cols):
        output_grid.set_cell(optimal_row, c, 2)
    
    # Connect remaining non-black squares to backbone
    for r, c in non_black:
        if output_grid.get_cell(r, c) != 2:
            if abs(r - optimal_row) <= abs(c - optimal_col):
                for rr in range(min(r, optimal_row), max(r, optimal_row) + 1):
                    if output_grid.get_cell(rr, c) == 0:
                        output_grid.set_cell(rr, c, 2)
            else:
                for cc in range(min(c, optimal_col), max(c, optimal_col) + 1):
                    if output_grid.get_cell(r, cc) == 0:
                        output_grid.set_cell(r, cc, 2)
    
    # Clean up unnecessary extensions
    for r in range(rows):
        left = min((c for c in range(cols) if output_grid.get_cell(r, c) != 0), default=cols)
        right = max((c for c in range(cols) if output_grid.get_cell(r, c) != 0), default=-1)
        for c in range(cols):
            if c < left or c > right:
                output_grid.set_cell(r, c, 0)
    
    for c in range(cols):
        top = min((r for r in range(rows) if output_grid.get_cell(r, c) != 0), default=rows)
        bottom = max((r for r in range(rows) if output_grid.get_cell(r, c) != 0), default=-1)
        for r in range(rows):
            if r < top or r > bottom:
                output_grid.set_cell(r, c, 0)
    
    # Verify connectivity and add necessary connections
    def dfs(start_r, start_c):
        stack = [(start_r, start_c)]
        visited = set()
        while stack:
            r, c = stack.pop()
            if (r, c) not in visited and output_grid.get_cell(r, c) != 0:
                visited.add((r, c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return visited
    
    connected = dfs(*non_black[0])
    if len(connected) != len(non_black):
        for r, c in non_black:
            if (r, c) not in connected:
                nearest = min(connected, key=lambda x: abs(x[0] - r) + abs(x[1] - c))
                while (r, c) != nearest:
                    if r != nearest[0]:
                        r += 1 if nearest[0] > r else -1
                    elif c != nearest[1]:
                        c += 1 if nearest[1] > c else -1
                    if output_grid.get_cell(r, c) == 0:
                        output_grid.set_cell(r, c, 2)
    
    # Remove isolated red squares
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 2:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= r + dr < rows and 0 <= c + dc < cols and output_grid.get_cell(r + dr, c + dc) != 0)
                if neighbors < 2:
                    output_grid.set_cell(r, c, 0)
    
    return output_grid
