from rob_agi.colored_grid import ColoredGrid
from collections import deque

def solve_c0f76784(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the c0f76784 challenge by identifying shapes made of 5s and filling their interiors.
    
    The solution follows these steps:
    1. Identify connected regions of 5s (shapes) using flood fill.
    2. For each shape:
       - Calculate its bounding box to determine width and height.
       - Fill inner cells with 8 for large shapes (width or height >= 5) or 7 for small shapes.
       - For odd-sized shapes, place a 6 in the center if it's not a border cell.
    3. Process all shapes in the grid independently.
    4. Restore the original border cells (5s) after processing.

    Args:
    input_grid (ColoredGrid): The input grid to be processed.

    Returns:
    ColoredGrid: The processed grid with filled shapes.
    """
    def flood_fill(grid, start_row, start_col):
        rows, cols = len(grid), len(grid[0])
        queue = deque([(start_row, start_col)])
        shape = set()
        while queue:
            r, c = queue.popleft()
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 5 and (r, c) not in shape:
                shape.add((r, c))
                queue.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
        return shape

    def get_bounding_box(shape):
        min_row = min(r for r, c in shape)
        max_row = max(r for r, c in shape)
        min_col = min(c for r, c in shape)
        max_col = max(c for r, c in shape)
        return min_row, max_row, min_col, max_col

    def process_shape(grid, shape):
        min_row, max_row, min_col, max_col = get_bounding_box(shape)
        height = max_row - min_row + 1
        width = max_col - min_col + 1

        fill_value = 8 if height >= 5 or width >= 5 else 7

        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                if (r, c) not in shape:
                    grid[r][c] = fill_value

        # Restore the original border cells (5s)
        for r, c in shape:
            grid[r][c] = 5

        if height % 2 == 1 and width % 2 == 1:
            center_r, center_c = (min_row + max_row) // 2, (min_col + max_col) // 2
            if (center_r, center_c) not in shape:
                grid[center_r][center_c] = 6 if height == 3 and width == 3 else fill_value

    output = input_grid.deep_copy()
    grid = output.values

    shapes = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 5 and not any((r, c) in shape for shape in shapes):
                shape = flood_fill(grid, r, c)
                if shape:
                    shapes.append(shape)
                    process_shape(grid, shape)

    return output
