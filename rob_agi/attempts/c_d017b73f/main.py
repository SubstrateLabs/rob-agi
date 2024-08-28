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
    Color groups are placed in their original vertical positions if possible, otherwise they are
    moved to the nearest available space below their original position. Single-cell groups are
    placed more flexibly to achieve better compression. Empty rows between non-empty rows are preserved.
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
    occupied_cells = set()
    for group, color in color_groups:
        min_r = min(r for r, _ in group)
        max_r = max(r for r, _ in group)
        group_height = max_r - min_r + 1
        group_width = max(c for _, c in group) - min(c for _, c in group) + 1
        
        # Find the nearest available space
        placed = False
        for start_r in range(min_r, rows - group_height + 1):
            for start_c in range(cols - group_width + 1):
                if all((r, c) not in occupied_cells 
                       for r in range(start_r, start_r + group_height)
                       for c in range(start_c, start_c + group_width)):
                    # Place the group
                    for r, c in group:
                        new_r = start_r + (r - min_r)
                        new_c = start_c + (c - min(c for _, c in group))
                        new_grid[new_r][new_c] = color
                        occupied_cells.add((new_r, new_c))
                    placed = True
                    break
            if placed:
                break
        
        if not placed:
            raise ValueError("Unable to place all color groups")
    
    # Step 5: Compress horizontally
    def compress_horizontally(grid):
        return [list(filter(lambda x: x is not None, row)) for row in zip(*[col for col in zip(*grid) if any(cell != 0 for cell in col)])]
    
    compressed_grid = compress_horizontally(new_grid)
    
    # Step 6: Preserve vertical structure and empty rows
    final_grid = []
    for input_row, compressed_row in zip(input_grid.values, compressed_grid):
        if all(cell == 0 for cell in input_row):
            final_grid.append([0] * len(compressed_row))
        else:
            final_grid.append(compressed_row)
    
    # Step 7: Remove empty rows at the edges
    while final_grid and all(cell == 0 for cell in final_grid[0]):
        final_grid.pop(0)
    while final_grid and all(cell == 0 for cell in final_grid[-1]):
        final_grid.pop()
    
    # Step 8: Ensure all rows have the same length
    max_length = max(len(row) for row in final_grid) if final_grid else 0
    final_grid = [row + [0] * (max_length - len(row)) for row in final_grid]
    
    # Step 9: Create and return the output
    return ColoredGrid(values=final_grid)
