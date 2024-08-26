from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_d017b73f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by compressing it horizontally while preserving color groups and vertical structure.
    
    The function identifies color groups, sorts them based on their topmost row and leftmost column,
    and then places them in a new grid as compactly as possible while maintaining their relative
    vertical order and left-to-right precedence. The resulting grid is compressed horizontally
    by removing columns that contain only black cells, while preserving the vertical alignment,
    appropriate horizontal spacing of color groups, and the original number of rows.
    """
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify color groups
    color_groups = []
    visited = set()
    
    def dfs(r: int, c: int, color: int) -> List[Tuple[int, int]]:
        group = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and input_grid.values[curr_r][curr_c] == color:
                visited.add((curr_r, curr_c))
                group.append((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return group
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and input_grid.values[r][c] != 0:
                group = dfs(r, c, input_grid.values[r][c])
                color_groups.append((group, input_grid.values[r][c]))
    
    # Step 2: Sort color groups
    color_groups.sort(key=lambda x: (min(r for r, _ in x[0]), min(c for _, c in x[0])))
    
    # Step 3: Initialize new grid
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Step 4: Place color groups
    rightmost_col = [0] * rows
    for group, color in color_groups:
        min_r = min(r for r, _ in group)
        max_r = max(r for r, _ in group)
        min_c = max(rightmost_col[min_r:max_r+1]) + 1
        shape = [(r - min_r, c - min(c for _, c in group)) for r, c in group]
        for r, c in group:
            new_grid[r][min_c + c - min(c for _, c in group)] = color
        for r in range(min_r, max_r + 1):
            rightmost_col[r] = max(rightmost_col[r], min_c + max(c for _, c in shape))
    
    # Step 5: Compress horizontally
    def compress_horizontally(grid):
        return [list(filter(lambda x: x is not None, row)) for row in zip(*[col for col in zip(*grid) if any(cell != 0 for cell in col)])]
    
    compressed_grid = compress_horizontally(new_grid)
    
    # Step 6: Preserve vertical structure
    final_grid = []
    for input_row, compressed_row in zip(input_grid.values, compressed_grid):
        if all(cell == 0 for cell in input_row):
            final_grid.append([0] * len(compressed_row))
        else:
            final_grid.append(compressed_row)
    
    # Step 7: Final cleanup
    final_grid = compress_horizontally(final_grid)
    
    # Step 8: Create and return the output
    return ColoredGrid(values=final_grid)
