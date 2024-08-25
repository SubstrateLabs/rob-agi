from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_b942fd60(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by connecting non-black squares with red lines.
    
    The function creates a minimal network of red lines that:
    1. Creates one or two vertical lines based on the distribution of colored squares
    2. Creates horizontal lines to connect colored squares to the vertical line(s)
    3. Preserves the original positions and colors of non-black squares
    4. Ensures red lines don't extend beyond the last colored square in any direction
    5. Removes any unnecessary or isolated red squares
    
    Steps:
    1. Create a deep copy of the input grid
    2. Identify all non-black squares
    3. Determine the optimal vertical line position(s)
    4. Create the basic structure with vertical and horizontal lines
    5. Connect any remaining unconnected squares
    6. Clean up unnecessary extensions
    7. Remove isolated red squares
    8. Perform a final connectivity check
    9. Return the modified grid
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()
    
    # Identify non-black squares
    non_black = [(r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) != 0]
    
    if not non_black:
        return output_grid  # Return original grid if no non-black squares
    
    # Find optimal vertical line(s)
    col_counts = [sum(1 for r in range(rows) if output_grid.get_cell(r, c) != 0) for c in range(cols)]
    left_col = min(c for c, count in enumerate(col_counts) if count > 0)
    right_col = max(c for c, count in enumerate(col_counts) if count > 0)
    
    if left_col == right_col or max(col_counts) > 1:
        # Use a single vertical line
        optimal_cols = [col_counts.index(max(col_counts))]
    else:
        # Use two vertical lines
        left_optimal = max(range(left_col, (left_col + right_col) // 2 + 1), key=lambda c: col_counts[c])
        right_optimal = max(range((left_col + right_col) // 2 + 1, right_col + 1), key=lambda c: col_counts[c])
        optimal_cols = [left_optimal, right_optimal]
    
    # Find top and bottom rows with colored squares
    top_row = min(r for r, c in non_black)
    bottom_row = max(r for r, c in non_black)
    
    # Create basic structure
    for col in optimal_cols:
        for r in range(top_row, bottom_row + 1):
            output_grid.set_cell(r, col, 2)
    
    for r in [top_row, bottom_row]:
        for c in range(min(optimal_cols), max(optimal_cols) + 1):
            output_grid.set_cell(r, c, 2)
    
    # Connect remaining non-black squares
    for r, c in non_black:
        if output_grid.get_cell(r, c) != 2:
            nearest_col = min(optimal_cols, key=lambda col: abs(col - c))
            for cc in range(min(c, nearest_col), max(c, nearest_col) + 1):
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
    
    # Remove isolated red squares
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 2:
                neighbors = sum(1 for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                                if 0 <= r + dr < rows and 0 <= c + dc < cols and output_grid.get_cell(r + dr, c + dc) != 0)
                if neighbors < 2:
                    output_grid.set_cell(r, c, 0)
    
    # Final connectivity check
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
    
    return output_grid
