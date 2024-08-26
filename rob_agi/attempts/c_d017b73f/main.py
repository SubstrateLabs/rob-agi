from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_d017b73f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by compressing it both vertically and horizontally while preserving color groups.
    
    The function identifies color groups, sorts them based on their topmost row and leftmost column,
    and then places them in a new grid as compactly as possible while maintaining their relative
    vertical order and left-to-right precedence. The resulting grid is compressed by removing
    any rows or columns that contain only black cells.
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
    
    # Step 3 & 4: Initialize new grid and place color groups
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for group, color in color_groups:
        min_r = min(r for r, _ in group)
        min_c = min(c for _, c in group)
        shape = [(r - min_r, c - min_c) for r, c in group]
        
        # Find the topmost-leftmost position to place the group
        for new_r in range(rows):
            for new_c in range(cols):
                if all(new_grid[new_r + dr][new_c + dc] == 0 for dr, dc in shape):
                    for dr, dc in shape:
                        new_grid[new_r + dr][new_c + dc] = color
                    break
            else:
                continue
            break
    
    # Step 5: Compress the grid
    def compress(grid):
        # Remove empty rows and columns
        grid = [row for row in grid if any(cell != 0 for cell in row)]
        grid = list(map(list, zip(*grid)))  # Transpose
        grid = [col for col in grid if any(cell != 0 for cell in col)]
        return list(map(list, zip(*grid)))  # Transpose back
    
    compressed_grid = compress(new_grid)
    
    # Step 6 & 7: Create and return the output
    return ColoredGrid(values=compressed_grid)
