from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_136b0064(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing shapes and reorganizing them into a 7-column grid.
    
    1. Splits the grid into left and right sections based on the yellow vertical line.
    2. Processes each section by identifying connected regions, compressing them horizontally.
    3. Places compressed shapes in a new 7-column grid, maintaining vertical order.
    4. Handles special cases like gray cells and optimizes the layout.
    5. Fills remaining space with black (empty) cells.
    
    Returns a new ColoredGrid with the transformed layout.
    """
    rows, cols = input_grid.get_dimensions()
    yellow_col = next(c for c in range(cols) if input_grid.values[0][c] == 4)
    
    def process_section(start_col: int, end_col: int) -> List[Tuple[int, List[List[int]]]]:
        shapes = []
        for color in range(1, 7):  # Exclude black (0) and yellow (4)
            regions = input_grid.find_connected_regions(color)
            for region in regions:
                if any(start_col <= c < end_col for _, c in region):
                    compressed = compress_shape(region, color)
                    shapes.append((min(r for r, _ in region), compressed))
        return sorted(shapes)
    
    def compress_shape(region: List[Tuple[int, int]], color: int) -> List[List[int]]:
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = max(min(c for _, c in region) - yellow_col, 0)
        max_c = min(max(c for _, c in region) - yellow_col, 2)
        shape = [[0 for _ in range(max_c - min_c + 1)] for _ in range(max_r - min_r + 1)]
        for r, c in region:
            shape[r - min_r][c - yellow_col - min_c] = color
        return shape if len(shape[0]) > len(shape) else list(map(list, zip(*shape[::-1])))  # Rotate if vertical
    
    left_shapes = process_section(0, yellow_col)
    right_shapes = process_section(yellow_col + 1, cols)
    
    output = [[0 for _ in range(7)] for _ in range(rows)]
    
    def place_shape(shape: List[List[int]], start_col: int, direction: int):
        for r, row in enumerate(shape):
            for c, color in enumerate(row):
                if color != 0:
                    output[r][start_col + c * direction] = color
    
    left_col, right_col = 0, 6
    for _, shape in left_shapes:
        place_shape(shape, left_col, 1)
        left_col += len(shape[0])
    for _, shape in right_shapes:
        place_shape(shape, right_col, -1)
        right_col -= len(shape[0])
    
    # Handle gray cell
    gray_pos = next(((r, c) for r in range(rows) for c in range(cols) if input_grid.values[r][c] == 5), None)
    if gray_pos:
        output[0][3] = 5  # Place gray in the middle of the top row
    
    # Optimize layout
    for c in range(7):
        if all(output[r][c] == 0 for r in range(rows)):
            for r in range(rows):
                output[r] = output[r][:c] + output[r][c+1:] + [0]
    
    return ColoredGrid(values=[row[:7] for row in output])  # Ensure 7 columns
