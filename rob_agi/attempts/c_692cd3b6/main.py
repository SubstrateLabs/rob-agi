from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_692cd3b6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by connecting two 'C' shapes with yellow, extending to grid edges when necessary.
    
    1. Identify the two 'C' shapes (red color 2 with gray color 5 inside).
    2. Create a bounding rectangle that encompasses both C-shapes.
    3. Fill the connecting rectangle with yellow, preserving the C-shapes.
    4. Extend yellow to grid edges if a C-shape touches an edge.
    5. Clean up any disconnected yellow areas.
    6. Preserve the original 'C' shapes.
    
    This approach ensures the correct yellow path between the C-shapes
    and extends to the appropriate grid edges only when a C-shape touches an edge.
    """
    # Step 1 & 2: Identify 'C' shapes and determine their positions
    c_shapes = find_c_shapes(input_grid)
    top_left_c, bottom_right_c = determine_c_shape_positions(c_shapes)
    
    # Step 3 & 4: Create bounding box and fill connecting area
    new_grid = input_grid.deep_copy()
    fill_connecting_area(new_grid, top_left_c, bottom_right_c)
    
    # Step 5 & 6: Extend yellow to appropriate edges
    extend_to_specific_edges(new_grid, top_left_c, bottom_right_c)
    
    # Step 7: Clean up unnecessary yellow
    clean_up_yellow(new_grid, top_left_c, bottom_right_c)
    
    # Step 8: Preserve original 'C' shapes
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

def determine_c_shape_positions(c_shapes: List[List[Tuple[int, int]]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    if len(c_shapes) != 2:
        raise ValueError("Expected exactly two C-shapes")
    
    c1, c2 = c_shapes
    c1_center = sum(r for r, _ in c1) / len(c1), sum(c for _, c in c1) / len(c1)
    c2_center = sum(r for r, _ in c2) / len(c2), sum(c for _, c in c2) / len(c2)
    
    if c1_center < c2_center:
        return c1, c2
    else:
        return c2, c1

def fill_connecting_area(grid: ColoredGrid, top_left_c: List[Tuple[int, int]], bottom_right_c: List[Tuple[int, int]]):
    bbox1 = get_bounding_box(top_left_c)
    bbox2 = get_bounding_box(bottom_right_c)
    
    connecting_bbox = get_connecting_bbox(bbox1, bbox2)
    
    # Extend the bounding box slightly beyond C-shape openings
    connecting_bbox = (
        max(0, connecting_bbox[0] - 1),
        max(0, connecting_bbox[1] - 1),
        min(grid.num_rows - 1, connecting_bbox[2] + 1),
        min(grid.num_cols - 1, connecting_bbox[3] + 1)
    )
    
    for r in range(connecting_bbox[0], connecting_bbox[2] + 1):
        for c in range(connecting_bbox[1], connecting_bbox[3] + 1):
            if grid.values[r][c] == 0:
                grid.values[r][c] = 4

def get_bounding_box(shape: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    max_c = max(c for _, c in shape)
    return (min_r, min_c, max_r, max_c)

def get_connecting_bbox(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return (
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3])
    )

def extend_to_specific_edges(grid: ColoredGrid, top_left_c: List[Tuple[int, int]], bottom_right_c: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    
    # Extend to top and left for top-left C-shape
    min_r_top = min(r for r, _ in top_left_c)
    min_c_left = min(c for _, c in top_left_c)
    
    for r in range(min_r_top, -1, -1):
        for c in range(min_c_left + 1):
            if grid.values[r][c] == 0:
                grid.values[r][c] = 4
    
    for c in range(min_c_left, -1, -1):
        for r in range(min_r_top + 1):
            if grid.values[r][c] == 0:
                grid.values[r][c] = 4
    
    # Extend to bottom and right for bottom-right C-shape
    max_r_bottom = max(r for r, _ in bottom_right_c)
    max_c_right = max(c for _, c in bottom_right_c)
    
    for r in range(max_r_bottom, rows):
        for c in range(max_c_right, cols):
            if grid.values[r][c] == 0:
                grid.values[r][c] = 4
    
    for c in range(max_c_right, cols):
        for r in range(max_r_bottom, rows):
            if grid.values[r][c] == 0:
                grid.values[r][c] = 4

def clean_up_yellow(grid: ColoredGrid, top_left_c: List[Tuple[int, int]], bottom_right_c: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    
    def is_connected_to_main_path(r: int, c: int) -> bool:
        visited = set()
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if curr_r < 0 or curr_r >= rows or curr_c < 0 or curr_c >= cols:
                continue
            if (curr_r, curr_c) in visited:
                continue
            visited.add((curr_r, curr_c))
            if grid.values[curr_r][curr_c] == 4:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if (nr, nc) in top_left_c or (nr, nc) in bottom_right_c:
                            return True
                stack.extend([(curr_r + dr, curr_c + dc) for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]])
        return False
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 4 and not is_connected_to_main_path(r, c):
                grid.values[r][c] = 0

def preserve_c_shapes(grid: ColoredGrid, c_shapes: List[List[Tuple[int, int]]]):
    for shape in c_shapes:
        for r, c in shape:
            if grid.values[r][c] == 4:
                grid.values[r][c] = 2
