from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Set

def solve_692cd3b6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by connecting two 'C' shapes with yellow, extending to grid edges only when necessary.
    
    1. Identify the two 'C' shapes (red color 2 with gray color 5 inside).
    2. Create a minimal bounding rectangle that encompasses both C-shapes.
    3. Fill the connecting rectangle with yellow, preserving the C-shapes.
    4. Extend yellow to edges if a C-shape touches an edge within the bounding rectangle.
    5. Fill open parts of C-shapes with yellow.
    6. Clean up any disconnected yellow areas.
    7. Preserve the original layout outside the connecting area.
    8. Ensure yellow forms a single connected region.
    
    This approach creates the correct yellow path between the C-shapes,
    extends to appropriate grid edges when necessary, and preserves
    the original layout outside the connecting area.
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
    extend_to_edges(new_grid, c_shapes[0], c_shapes[1], connecting_rectangle)
    
    # Step 5: Fill open parts of C-shapes
    fill_open_parts(new_grid, c_shapes)
    
    # Step 6 & 8: Clean up disconnected yellow areas and ensure single connected region
    clean_up_yellow(new_grid, connecting_rectangle)
    
    # Step 7: Preserve original layout outside connecting area
    preserve_original_layout(new_grid, input_grid, connecting_rectangle)
    
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

def extend_to_edges(grid: ColoredGrid, shape1: List[Tuple[int, int]], shape2: List[Tuple[int, int]], connecting_rectangle: Tuple[int, int, int, int]):
    rows, cols = grid.get_dimensions()
    left, top, right, bottom = connecting_rectangle
    
    # Extend to left edge if any shape touches it within the connecting rectangle
    if any(c == left and left == 0 for _, c in shape1 + shape2):
        for r in range(top, bottom + 1):
            grid.values[r][0] = 4
    
    # Extend to right edge if any shape touches it within the connecting rectangle
    if any(c == right and right == cols - 1 for _, c in shape1 + shape2):
        for r in range(top, bottom + 1):
            grid.values[r][cols - 1] = 4
    
    # Extend to top edge if any shape touches it within the connecting rectangle
    if any(r == top and top == 0 for r, _ in shape1 + shape2):
        for c in range(left, right + 1):
            grid.values[0][c] = 4
    
    # Extend to bottom edge if any shape touches it within the connecting rectangle
    if any(r == bottom and bottom == rows - 1 for r, _ in shape1 + shape2):
        for c in range(left, right + 1):
            grid.values[rows - 1][c] = 4

def clean_up_yellow(grid: ColoredGrid, connecting_rectangle: Tuple[int, int, int, int]):
    rows, cols = grid.get_dimensions()
    
    def flood_fill(r: int, c: int, target_color: int, replacement_color: int):
        if target_color == replacement_color:
            return
        if grid.values[r][c] != target_color:
            return
        
        grid.values[r][c] = replacement_color
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                flood_fill(nr, nc, target_color, replacement_color)
    
    # Find a yellow cell within the connecting rectangle
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 4:
                flood_fill(r, c, 4, 5)  # Temporarily mark connected yellow as 5
                break
        else:
            continue
        break
    
    # Clean up disconnected yellow and revert connected yellow back to 4
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 4:
                grid.values[r][c] = 0
            elif grid.values[r][c] == 5:
                grid.values[r][c] = 4

def preserve_c_shapes(grid: ColoredGrid, c_shapes: List[List[Tuple[int, int]]]):
    for shape in c_shapes:
        for r, c in shape:
            if grid.values[r][c] in [2, 5]:
                grid.values[r][c] = grid.values[r][c]  # Preserve original color
def preserve_original_layout(new_grid: ColoredGrid, input_grid: ColoredGrid, connecting_rectangle: Tuple[int, int, int, int]):
    left, top, right, bottom = connecting_rectangle
    rows, cols = new_grid.get_dimensions()
    
    for r in range(rows):
        for c in range(cols):
            if not (top <= r <= bottom and left <= c <= right):
                new_grid.values[r][c] = input_grid.values[r][c]
def fill_open_parts(grid: ColoredGrid, c_shapes: List[List[Tuple[int, int]]]):
    for shape in c_shapes:
        # Find the open part of the C-shape
        open_cell = find_open_part(grid, shape)
        if open_cell:
            r, c = open_cell
            grid.values[r][c] = 4  # Fill with yellow

def find_open_part(grid: ColoredGrid, shape: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    rows, cols = grid.get_dimensions()
    for r, c in shape:
        if grid.values[r][c] == 2:  # Red cell
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in shape and grid.values[nr][nc] == 0:
                    return nr, nc
    return None
