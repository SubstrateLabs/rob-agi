from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_60a26a3e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by connecting red shapes with minimal blue lines.
    
    The solution follows these steps:
    1. Identify all red (2) shapes
    2. Group red shapes horizontally and vertically
    3. Create a minimal backbone structure connecting the shapes
    4. Optimize the solution by removing unnecessary lines
    
    This approach creates a minimal structure of blue lines that efficiently connects
    red shapes, focusing on creating a skeletal structure rather than filling all gaps.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = input_grid.get_dimensions()
    
    # Step 1: Identify red shapes
    red_shapes = [(r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 2]
    
    if not red_shapes:
        return output_grid
    
    # Step 2: Group red shapes
    groups = group_shapes(red_shapes)
    
    # Step 3: Create minimal backbone structure
    for group in groups:
        connect_group(output_grid, group)
    
    # Step 4: Create vertical connections between groups
    connect_groups_vertically(output_grid, groups)
    
    return output_grid

def group_shapes(shapes: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    shapes.sort()
    groups = []
    current_group = [shapes[0]]
    for shape in shapes[1:]:
        if shape[0] - current_group[-1][0] <= 2 and abs(shape[1] - current_group[-1][1]) <= 3:
            current_group.append(shape)
        else:
            groups.append(current_group)
            current_group = [shape]
    groups.append(current_group)
    return groups

def connect_group(grid: ColoredGrid, group: List[Tuple[int, int]]):
    min_row = min(shape[0] for shape in group)
    max_row = max(shape[0] for shape in group)
    min_col = min(shape[1] for shape in group)
    max_col = max(shape[1] for shape in group)
    
    # Connect horizontally
    mid_row = (min_row + max_row) // 2
    for c in range(min_col, max_col + 1):
        if grid.values[mid_row][c] != 2:
            grid.values[mid_row][c] = 1
    
    # Connect vertically if needed
    if max_row - min_row > 1:
        mid_col = (min_col + max_col) // 2
        for r in range(min_row, max_row + 1):
            if grid.values[r][mid_col] != 2:
                grid.values[r][mid_col] = 1

def connect_groups_vertically(grid: ColoredGrid, groups: List[List[Tuple[int, int]]]):
    if len(groups) <= 1:
        return
    
    groups.sort(key=lambda g: min(shape[0] for shape in g))
    
    for i in range(len(groups) - 1):
        top_group = groups[i]
        bottom_group = groups[i + 1]
        
        top_max_row = max(shape[0] for shape in top_group)
        bottom_min_row = min(shape[0] for shape in bottom_group)
        
        if bottom_min_row - top_max_row <= 2:
            continue
        
        top_cols = set(shape[1] for shape in top_group)
        bottom_cols = set(shape[1] for shape in bottom_group)
        common_cols = top_cols.intersection(bottom_cols)
        
        if common_cols:
            connect_col = min(common_cols)
        else:
            connect_col = min(top_cols.union(bottom_cols))
        
        for r in range(top_max_row + 1, bottom_min_row):
            grid.values[r][connect_col] = 1
