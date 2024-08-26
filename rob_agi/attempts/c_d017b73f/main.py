from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict

def solve_d017b73f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by compressing it both vertically and horizontally while preserving color groups.
    
    The function identifies color groups, sorts them based on their topmost row and leftmost column,
    and then places them in a new grid as compactly as possible while maintaining their relative
    vertical order and left-to-right precedence. The resulting grid is compressed by removing
    any rows or columns that contain only black cells, while preserving the vertical alignment
    and appropriate horizontal spacing of color groups.
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
    row_groups: Dict[int, List] = {}
    for group, color in color_groups:
        min_r = min(r for r, _ in group)
        if min_r not in row_groups:
            row_groups[min_r] = []
        row_groups[min_r].append((group, color))
    
    current_col = 0
    for row in range(rows):
        if row in row_groups:
            for group, color in row_groups[row]:
                shape = [(r - row, c - min(c for _, c in group)) for r, c in group]
                for dr, dc in shape:
                    new_grid[row + dr][current_col + dc] = color
                current_col += max(dc for _, dc in shape) + 2  # Add spacing
        elif row > 0:
            new_grid[row] = new_grid[row - 1].copy()
    
    # Step 5: Compress the grid
    def compress(grid):
        # Remove empty rows and columns
        grid = [row for row in grid if any(cell != 0 for cell in row)]
        grid = list(map(list, zip(*grid)))  # Transpose
        grid = [col for col in col if any(cell != 0 for cell in col)]
        return list(map(list, zip(*grid)))  # Transpose back
    
    compressed_grid = compress(new_grid)
    
    # Step 6: Create and return the output
    return ColoredGrid(values=compressed_grid)
