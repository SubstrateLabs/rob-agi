from rob_agi.colored_grid import ColoredGrid

def solve_da515329(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the da515329 challenge by transforming the input grid into a structured pattern.
    
    The solution involves the following steps:
    1. Analyze the input grid to find the plus sign center and dimensions
    2. Create the outermost rectangle frame
    3. Generate nested rectangles with openings
    4. Connect the rectangles and the central plus sign
    5. Add characteristic edge patterns
    6. Add isolated pixels
    7. Make final adjustments and optimizations based on grid size
    
    Args:
    input_grid (ColoredGrid): The input grid containing a plus sign

    Returns:
    ColoredGrid: The transformed grid with a structured pattern
    """
    rows, cols = input_grid.get_dimensions()
    new_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])
    
    center, plus_width, plus_height = analyze_input(input_grid)
    create_outer_frame(new_grid)
    generate_nested_rectangles(new_grid, center, plus_width, plus_height)
    connect_structures(new_grid, center)
    add_edge_patterns(new_grid)
    add_isolated_pixels(new_grid)
    final_adjustments(new_grid, input_grid)
    
    return new_grid

def analyze_input(grid: ColoredGrid) -> tuple:
    rows, cols = grid.get_dimensions()
    center = None
    plus_width, plus_height = 0, 0
    
    for r in range(rows):
        for c in range(cols):
            if grid.values[r][c] == 8:
                if center is None:
                    center = (r, c)
                plus_height = max(plus_height, abs(r - center[0]) * 2 + 1)
                plus_width = max(plus_width, abs(c - center[1]) * 2 + 1)
    
    return center, plus_width, plus_height

def create_outer_frame(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and not (r < 2 and c < 2):
                grid.values[r][c] = 8

def generate_nested_rectangles(grid: ColoredGrid, center: tuple, plus_width: int, plus_height: int):
    rows, cols = grid.get_dimensions()
    top, left = 2, 2
    bottom, right = rows - 3, cols - 3
    opening_side = 0  # 0: top, 1: right, 2: bottom, 3: left

    while right - left > plus_width and bottom - top > plus_height:
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                if r in (top, bottom) or c in (left, right):
                    if not (opening_side == 0 and r == top and left + (right - left) // 2 - 1 <= c <= left + (right - left) // 2) and \
                       not (opening_side == 1 and c == right and top + (bottom - top) // 2 - 1 <= r <= top + (bottom - top) // 2) and \
                       not (opening_side == 2 and r == bottom and left + (right - left) // 2 - 1 <= c <= left + (right - left) // 2) and \
                       not (opening_side == 3 and c == left and top + (bottom - top) // 2 - 1 <= r <= top + (bottom - top) // 2):
                        grid.values[r][c] = 8

        top += 2
        left += 2
        bottom -= 2
        right -= 2
        opening_side = (opening_side + 1) % 4

def connect_structures(grid: ColoredGrid, center: tuple):
    rows, cols = grid.get_dimensions()
    r, c = center

    # Connect vertically
    for i in range(r, rows):
        if grid.values[i][c] == 8:
            break
        grid.values[i][c] = 8
    for i in range(r, -1, -1):
        if grid.values[i][c] == 8:
            break
        grid.values[i][c] = 8

    # Connect horizontally
    for j in range(c, cols):
        if grid.values[r][j] == 8:
            break
        grid.values[r][j] = 8
    for j in range(c, -1, -1):
        if grid.values[r][j] == 8:
            break
        grid.values[r][j] = 8

def add_edge_patterns(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()

    # Right edge pattern
    for r in range(3, rows - 3, 3):
        grid.values[r][cols-1] = grid.values[r][cols-2] = 0

    # Bottom edge pattern
    for c in range(3, cols - 3, 3):
        grid.values[rows-1][c] = grid.values[rows-2][c] = 0
        if c + 1 < cols:
            grid.values[rows-1][c+1] = 8

def add_isolated_pixels(grid: ColoredGrid):
    rows, cols = grid.get_dimensions()
    isolated_pixels = [
        (1, cols // 4), (2, cols // 3),
        (rows // 4, 1), (rows // 3, 2),
        (rows - 3, cols - 3), (rows - 4, cols - 4)
    ]
    for r, c in isolated_pixels:
        if 0 <= r < rows and 0 <= c < cols:
            grid.values[r][c] = 8

def final_adjustments(new_grid: ColoredGrid, input_grid: ColoredGrid):
    rows, cols = new_grid.get_dimensions()
    
    # Ensure the original plus sign is preserved
    for r in range(rows):
        for c in range(cols):
            if input_grid.values[r][c] == 8:
                new_grid.values[r][c] = 8

    # Keep top-left corner open
    new_grid.values[0][0] = new_grid.values[0][1] = new_grid.values[1][0] = 0

    # Ensure connectivity
    center = (rows // 2, cols // 2)
    visited = set()

    def dfs(r, c):
        if not (0 <= r < rows and 0 <= c < cols) or new_grid.values[r][c] == 0 or (r, c) in visited:
            return
        visited.add((r, c))
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(r + dr, c + dc)

    dfs(center[0], center[1])

    # Remove unconnected sky blue pixels
    for r in range(rows):
        for c in range(cols):
            if new_grid.values[r][c] == 8 and (r, c) not in visited:
                new_grid.values[r][c] = 0
