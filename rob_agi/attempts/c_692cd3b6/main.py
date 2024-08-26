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
    # Step 1: Identify 'C' shapes
    c_shapes = find_c_shapes(input_grid)
    if len(c_shapes) != 2:
        raise ValueError("Expected exactly two C-shapes")
    
    # Step 2 & 3: Create connecting rectangle and fill with yellow
    new_grid = input_grid.deep_copy()
    connecting_rectangle = create_connecting_rectangle(c_shapes[0], c_shapes[1])
    fill_yellow(new_grid, *connecting_rectangle)
    
    # Step 4: Extend to edges if necessary
    extend_to_edges(new_grid, c_shapes[0], c_shapes[1])
    
    # Step 5: Clean up disconnected yellow areas
    clean_up_yellow(new_grid, connecting_rectangle)
    
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
            if (curr_r, curr_c) not in visited and grid.values[curr_r][curr_c] in [2, 5]:
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

def get_bounding_box(shape: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_r = min(r for r, _ in shape)
    max_r = max(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    max_c = max(c for _, c in shape)
    return (min_r, min_c, max_r, max_c)

def create_connecting_rectangle(shape1: List[Tuple[int, int]], shape2: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    bbox1 = get_bounding_box(shape1)
    bbox2 = get_bounding_box(shape2)
    return (
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3])
    )

def fill_yellow(grid: ColoredGrid, left: int, top: int, right: int, bottom: int):
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            if grid.values[r][c] == 0:
                grid.values[r][c] = 4

def extend_to_edges(grid: ColoredGrid, shape1: List[Tuple[int, int]], shape2: List[Tuple[int, int]]):
    rows, cols = grid.get_dimensions()
    bbox1 = get_bounding_box(shape1)
    bbox2 = get_bounding_box(shape2)
    
    # Extend to left edge if any shape touches it
    if bbox1[1] == 0 or bbox2[1] == 0:
        for r in range(rows):
            if grid.values[r][0] == 0:
                grid.values[r][0] = 4
    
    # Extend to right edge if any shape touches it
    if bbox1[3] == cols - 1 or bbox2[3] == cols - 1:
        for r in range(rows):
            if grid.values[r][cols - 1] == 0:
                grid.values[r][cols - 1] = 4
    
    # Extend to top edge if any shape touches it
    if bbox1[0] == 0 or bbox2[0] == 0:
        for c in range(cols):
            if grid.values[0][c] == 0:
                grid.values[0][c] = 4
    
    # Extend to bottom edge if any shape touches it
    if bbox1[2] == rows - 1 or bbox2[2] == rows - 1:
        for c in range(cols):
            if grid.values[rows - 1][c] == 0:
                grid.values[rows - 1][c] = 4

def clean_up_yellow(grid: ColoredGrid, connecting_rectangle: Tuple[int, int, int, int]):
    rows, cols = grid.get_dimensions()
    top, left, bottom, right = connecting_rectangle
    
    def is_connected(r: int, c: int) -> bool:
        visited = set()
        stack = [(r, c)]
        while stack:
            curr_r, curr_c = stack.pop()
            if curr_r < top or curr_r > bottom or curr_c < left or curr_c > right:
                return True
            if (curr_r, curr_c) in visited:
                continue
            visited.add((curr_r, curr_c))
            if grid.values[curr_r][curr_c] == 4:
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        stack.append((nr, nc))
        return False
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 4 and not is_connected(r, c):
                grid.values[r][c] = 0

def preserve_c_shapes(grid: ColoredGrid, c_shapes: List[List[Tuple[int, int]]]):
    for shape in c_shapes:
        for r, c in shape:
            if grid.values[r][c] in [2, 5]:
                grid.values[r][c] = grid.values[r][c]  # Preserve original color
