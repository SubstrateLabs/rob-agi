from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_0a2355a6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying contiguous shapes,
    grouping similar shapes, prioritizing groups, and assigning colors.
    
    1. Identify contiguous shapes of 8s in the input grid.
    2. Group similar shapes based on structure and dimensions.
    3. Prioritize groups by total area, position, and number of shapes.
    4. Assign colors (1: blue, 2: red, 3: green, 4: yellow) to groups.
    5. Create and return a new grid with transformed colors.
    """
    # Step 1: Identify contiguous shapes
    shapes = find_contiguous_shapes(input_grid)
    
    # Step 2: Group similar shapes
    groups = group_similar_shapes(shapes)
    
    # Step 3: Prioritize groups
    prioritized_groups = prioritize_groups(groups)
    
    # Step 4 & 5: Assign colors and create output grid
    output_grid = create_output_grid(input_grid, prioritized_groups)
    
    return output_grid

def find_contiguous_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    shapes = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    def dfs(r: int, c: int) -> List[Tuple[int, int]]:
        shape = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.get_cell(curr_r, curr_c) == 8:
                visited.add((curr_r, curr_c))
                shape.append((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return shape
    
    for r in range(rows):
        for c in range(cols):
            if grid.get_cell(r, c) == 8 and (r, c) not in visited:
                shapes.append(dfs(r, c))
    
    return shapes

def group_similar_shapes(shapes: List[List[Tuple[int, int]]]) -> List[List[List[Tuple[int, int]]]]:
    groups = []
    for shape in shapes:
        added = False
        for group in groups:
            if are_shapes_similar(shape, group[0]):
                group.append(shape)
                added = True
                break
        if not added:
            groups.append([shape])
    return groups

def are_shapes_similar(shape1: List[Tuple[int, int]], shape2: List[Tuple[int, int]]) -> bool:
    # Compare shapes based on dimensions and outline
    min_r1, min_c1 = min(shape1)
    max_r1, max_c1 = max(shape1)
    min_r2, min_c2 = min(shape2)
    max_r2, max_c2 = max(shape2)
    
    height1, width1 = max_r1 - min_r1 + 1, max_c1 - min_c1 + 1
    height2, width2 = max_r2 - min_r2 + 1, max_c2 - min_c2 + 1
    
    if (height1, width1) != (height2, width2):
        return False
    
    outline1 = set((r - min_r1, c - min_c1) for r, c in shape1)
    outline2 = set((r - min_r2, c - min_c2) for r, c in shape2)
    
    return outline1 == outline2

def prioritize_groups(groups: List[List[List[Tuple[int, int]]]]) -> List[List[List[Tuple[int, int]]]]:
    return sorted(groups, key=lambda g: (-sum(len(s) for s in g), min(min(s) for s in g), -len(g)))

def create_output_grid(input_grid: ColoredGrid, prioritized_groups: List[List[List[Tuple[int, int]]]]) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    colors = [1, 2, 3, 4]  # blue, red, green, yellow
    color_index = 0
    
    for group in prioritized_groups:
        color = colors[color_index]
        for shape in group:
            for r, c in shape:
                output_grid.set_cell(r, c, color)
        color_index = (color_index + 1) % len(colors)
    
    return output_grid
