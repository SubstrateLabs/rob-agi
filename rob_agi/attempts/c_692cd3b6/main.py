from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_692cd3b6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by connecting two 'C' shapes with yellow, extending to grid edges when necessary.
    
    1. Identify the two 'C' shapes (red color 2 with gray color 5 inside).
    2. Determine the bounding box for the yellow connecting path.
    3. Fill the connecting area with yellow, including the openings of the C-shapes.
    4. Extend yellow to grid edges if the connecting area touches an edge.
    5. Clean up any unnecessary yellow areas.
    6. Preserve the original 'C' shapes.
    
    This approach ensures the most direct yellow path between the C-shapes
    while extending to the grid edges only when necessary.
    """
    # Step 1: Identify 'C' shapes
    c_shapes = find_c_shapes(input_grid)
    
    # Step 2 & 3: Determine bounding box and fill connecting area
    new_grid = input_grid.deep_copy()
    fill_connecting_area(new_grid, c_shapes)
    
    # Step 4: Extend yellow to grid edges if necessary
    extend_to_edges(new_grid)
    
    # Step 5: Clean up unnecessary yellow
    clean_up_yellow(new_grid)
    
    # Step 6: Preserve original 'C' shapes
    preserve_c_shapes(new_grid, c_shapes)
    
    return new_grid

def find_c_shapes(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
    shapes = []
    visited = set()
    rows, cols = grid.get_dimensions()
    
    def dfs(r: int, c: int) -> List[Tuple[int, int]]:
        shape = []
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] == 2:
                visited.add((curr_r, curr_c))
                shape.append((curr_r, curr_c))
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return shape
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 2 and (r, c) not in visited:
                shapes.append(dfs(r, c))
    
    return shapes

def get_occupied_rows_cols(shapes: List[List[Tuple[int, int]]]) -> Tuple[Set[int], Set[int]]:
    occupied_rows = set()
    occupied_cols = set()
    for shape in shapes:
        for r, c in shape:
            occupied_rows.add(r)
            occupied_cols.add(c)
    return occupied_rows, occupied_cols

def fill_connecting_area(grid: ColoredGrid, c_shapes: List[List[Tuple[int, int]]]):
    if len(c_shapes) != 2:
        raise ValueError("Expected exactly two C-shapes")
    
    # Find bounding boxes for each C-shape
    bbox1 = get_bounding_box(c_shapes[0])
    bbox2 = get_bounding_box(c_shapes[1])
    
    # Determine the connecting area
    connecting_bbox = get_connecting_bbox(bbox1, bbox2)
    
    # Fill the connecting area
    for r in range(connecting_bbox[0], connecting_bbox[2] + 1):
        for c in range(connecting_bbox[1], connecting_bbox[3] + 1):
            if grid.values[r][c] == 0:
                grid.values[r][c] = 4
    
    # Fill C-shape openings
    fill_c_shape_openings(grid, c_shapes)

def get_bounding_box(shape: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    max_c = max(c for _, c in shape)
    return (min_r, min_c, max_r, max_c)

def get_connecting_bbox(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    min_r = min(bbox1[0], bbox2[0])
    min_c = min(bbox1[1], bbox2[1])
    max_r = max(bbox1[2], bbox2[2])
    max_c = max(bbox1[3], bbox2[3])
    return (min_r, min_c, max_r, max_c)

def fill_c_shape_openings(grid: ColoredGrid, c_shapes: List[List[Tuple[int, int]]]):
    for shape in c_shapes:
        bbox = get_bounding_box(shape)
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                if grid.values[r][c] == 0 and (r, c) != center:
                    grid.values[r][c] = 4

def extend_to_edges(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    
    # Check top and bottom edges
    for c in range(cols):
        if grid.values[0][c] == 4:
            for r in range(rows):
                if grid.values[r][c] == 0:
                    grid.values[r][c] = 4
        if grid.values[rows-1][c] == 4:
            for r in range(rows-1, -1, -1):
                if grid.values[r][c] == 0:
                    grid.values[r][c] = 4
    
    # Check left and right edges
    for r in range(rows):
        if grid.values[r][0] == 4:
            for c in range(cols):
                if grid.values[r][c] == 0:
                    grid.values[r][c] = 4
        if grid.values[r][cols-1] == 4:
            for c in range(cols-1, -1, -1):
                if grid.values[r][c] == 0:
                    grid.values[r][c] = 4

def clean_up_yellow(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    
    def clean_edge(edge_cells):
        for r, c in edge_cells:
            if grid.values[r][c] == 4:
                grid.values[r][c] = 0
            else:
                break

    # Clean top and bottom edges
    for c in range(cols):
        clean_edge([(r, c) for r in range(rows)])
        clean_edge([(r, c) for r in range(rows-1, -1, -1)])
    
    # Clean left and right edges
    for r in range(rows):
        clean_edge([(r, c) for c in range(cols)])
        clean_edge([(r, c) for c in range(cols-1, -1, -1)])

def preserve_c_shapes(grid: ColoredGrid, c_shapes: List[List[Tuple[int, int]]]):
    for shape in c_shapes:
        for r, c in shape:
            if grid.values[r][c] == 4:
                grid.values[r][c] = 2
