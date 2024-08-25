from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_0a2355a6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by identifying distinct shapes,
    classifying them, and assigning colors based on their complexity and position.
    
    1. Identify distinct contiguous shapes of 8s in the input grid.
    2. Classify shapes based on size, complexity, hollowness, and regularity.
    3. Group similar shapes and determine the color palette.
    4. Assign colors (1: blue, 2: red, 3: green, 4: yellow) to shapes based on complexity and position.
    5. Create and return a new grid with transformed colors.
    """
    # Step 1: Identify distinct shapes
    shapes = find_contiguous_shapes(input_grid)
    
    # Step 2 & 3: Classify shapes and group similar ones
    classified_shapes = classify_shapes(shapes)
    shape_groups = group_similar_shapes(classified_shapes)
    
    # Step 4: Determine color palette and assign colors
    color_palette = determine_color_palette(len(shape_groups))
    colored_shapes = assign_colors(shape_groups, color_palette)
    
    # Step 5: Create output grid
    output_grid = create_output_grid(input_grid, colored_shapes)
    
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

def classify_shapes(shapes: List[List[Tuple[int, int]]]) -> List[Tuple[List[Tuple[int, int]], Dict]]:
    classified = []
    for shape in shapes:
        size = len(shape)
        min_r, min_c = min(shape)
        max_r, max_c = max(shape)
        bounding_box_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        perimeter = sum(1 for r, c in shape if any((r+dr, c+dc) not in shape for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]))
        
        attributes = {
            'size': size,
            'complexity': perimeter / size,
            'hollowness': 1 - (size / bounding_box_area),
            'regularity': size / bounding_box_area
        }
        classified.append((shape, attributes))
    return classified

def group_similar_shapes(classified_shapes: List[Tuple[List[Tuple[int, int]], Dict]]) -> List[List[Tuple[List[Tuple[int, int]], Dict]]]:
    groups = []
    for shape, attributes in classified_shapes:
        added = False
        for group in groups:
            if are_shapes_similar(attributes, group[0][1]):
                group.append((shape, attributes))
                added = True
                break
        if not added:
            groups.append([(shape, attributes)])
    return groups

def are_shapes_similar(attr1: Dict, attr2: Dict) -> bool:
    return all(abs(attr1[k] - attr2[k]) < 0.1 for k in attr1)

def determine_color_palette(num_groups: int) -> List[int]:
    if num_groups <= 2:
        return [1, 3]  # blue, green
    elif num_groups == 3:
        return [1, 2, 3]  # blue, red, green
    else:
        return [1, 2, 3, 4]  # blue, red, green, yellow

def assign_colors(shape_groups: List[List[Tuple[List[Tuple[int, int]], Dict]]], color_palette: List[int]) -> List[Tuple[List[Tuple[int, int]], int]]:
    colored_shapes = []
    shape_groups.sort(key=lambda g: g[0][1]['complexity'])
    
    for i, group in enumerate(shape_groups):
        color = color_palette[i % len(color_palette)]
        group.sort(key=lambda s: (min(s[0])[0], min(s[0])[1]))  # Sort by top-left position
        for shape, _ in group:
            colored_shapes.append((shape, color))
    
    return colored_shapes

def create_output_grid(input_grid: ColoredGrid, colored_shapes: List[Tuple[List[Tuple[int, int]], int]]) -> ColoredGrid:
    output_grid = input_grid.deep_copy()
    for shape, color in colored_shapes:
        for r, c in shape:
            output_grid.set_cell(r, c, color)
    return output_grid
