from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import defaultdict

def solve_e78887d1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms an input grid into a 3-row output grid by compressing, idealizing, and arranging color patterns.
    
    The solution involves:
    1. Analyzing color groups and their patterns in the input grid.
    2. Compressing vertical repetitions to fit within 3 rows.
    3. Idealizing shapes while maintaining their essential characteristics.
    4. Arranging idealized shapes horizontally, preserving relative positions.
    5. Optimizing space usage and ensuring proper separation between shapes.
    6. Aligning elements vertically and completing partial patterns where appropriate.
    7. Refining the output to ensure symmetry and completeness of shapes.
    
    This approach preserves the essence of input patterns while creating a consistent, idealized 3-row output.
    """
    rows, cols = input_grid.get_dimensions()
    color_groups = identify_color_groups(input_grid)
    
    compressed_shapes = []
    for color, positions in color_groups.items():
        bounding_box = get_bounding_box(positions)
        pattern = analyze_pattern(positions, bounding_box)
        prominence = len(positions) / ((bounding_box[2] - bounding_box[0] + 1) * (bounding_box[3] - bounding_box[1] + 1))
        compressed = compress_vertically(positions, bounding_box)
        idealized = idealize_shape(compressed, pattern, prominence)
        compressed_shapes.append((color, idealized, bounding_box, prominence))
    
    output_grid = arrange_shapes(compressed_shapes, cols)
    optimize_space(output_grid)
    align_vertically(output_grid)
    refine_output(output_grid)
    
    return output_grid

def identify_color_groups(grid: ColoredGrid) -> Dict[int, List[Tuple[int, int]]]:
    color_groups = defaultdict(list)
    rows, cols = grid.get_dimensions()
    for r in range(rows):
        for c in range(cols):
            color = grid.values[r][c]
            if color != 0:
                color_groups[color].append((r, c))
    return dict(color_groups)

def analyze_pattern(positions: List[Tuple[int, int]], rows: int, cols: int) -> str:
    unique_rows = len(set(r for r, _ in positions))
    unique_cols = len(set(c for _, c in positions))
    
    if unique_rows == 1:
        return "horizontal"
    if unique_cols == 1:
        return "vertical"
    if len(positions) >= rows * cols / 4:
        return "block"
    if unique_rows == 2 and unique_cols == 2:
        return "corner"
    return "scattered"

def create_idealized_representation(color: int, pattern: str, prominence: float) -> List[List[int]]:
    if pattern == "vertical":
        return [[color, 0, color], [color, 0, color], [color, 0, color]]
    if pattern == "horizontal":
        return [[0, color, 0], [color, color, color], [0, color, 0]]
    if pattern == "block":
        return [[color, color, 0], [color, color, color], [0, color, color]]
    if pattern == "corner":
        return [[color, color, 0], [color, 0, 0], [color, color, 0]]
    # Scattered pattern
    if prominence > 0.5:
        return [[color, 0, color], [0, color, 0], [color, 0, color]]
    else:
        return [[color, 0, 0], [0, color, 0], [0, 0, color]]

def arrange_shapes(compressed_shapes: List[Tuple[int, List[List[int]], Tuple[int, int, int, int], float]], cols: int) -> ColoredGrid:
    output = [[0 for _ in range(cols)] for _ in range(3)]
    for color, shape, (top, left, bottom, right), prominence in compressed_shapes:
        shape_width = right - left + 1
        for r in range(3):
            for c in range(shape_width):
                if shape[r][c] != 0 and output[r][left + c] == 0:
                    output[r][left + c] = color
    return ColoredGrid(values=output)

def identify_pattern(positions: List[Tuple[int, int]], rows: int, cols: int) -> str:
    if len(set(r for r, _ in positions)) == 1:
        return "horizontal"
    if len(set(c for _, c in positions)) == 1:
        return "vertical"
    if len(positions) >= rows * cols / 4:
        return "block"
    return "scattered"

def create_representation(color: int, pattern: str, cols: int) -> List[List[int]]:
    if pattern == "vertical":
        return [[color, 0, color] for _ in range(3)]
    if pattern == "horizontal":
        return [[0, 0, 0], [color] * 3, [0, 0, 0]]
    if pattern == "block":
        return [[color, color, 0], [color, color, color], [0, color, color]]
    return [[color, 0, 0], [0, color, 0], [0, 0, color]]  # scattered

def merge_representation(output_grid: ColoredGrid, representation: List[List[int]]):
    for r in range(3):
        for c in range(len(representation[0])):
            if representation[r][c] != 0:
                output_grid.values[r][output_grid.values[r].index(0)] = representation[r][c]

def refine_output(grid: ColoredGrid):
    for col in range(grid.get_dimensions()[1]):
        colors = [grid.values[r][col] for r in range(3)]
        if len(set(colors)) == 1 and colors[0] != 0:
            grid.values[1][col] = 0
        elif colors.count(0) == 2:
            non_zero = next(color for color in colors if color != 0)
            for r in range(3):
                if grid.values[r][col] == 0:
                    grid.values[r][col] = non_zero
                    break

def complete_patterns(grid: ColoredGrid):
    """Completes patterns in the grid by filling in missing parts of shapes and ensuring consistency."""
    for col in range(grid.get_dimensions()[1]):
        if is_partial_vertical_line(grid, col):
            complete_vertical_line(grid, col)
        if is_partial_horizontal_line(grid, col):
            complete_horizontal_line(grid, col)
    
    # Ensure vertical alignment
    for col in range(grid.get_dimensions()[1]):
        align_vertically(grid, col)

def align_vertically(grid: ColoredGrid, col: int):
    """Aligns colors vertically in a column, moving non-zero values to the top."""
    colors = [grid.values[row][col] for row in range(3) if grid.values[row][col] != 0]
    for row in range(len(colors)):
        grid.values[row][col] = colors[row]
    for row in range(len(colors), 3):
        grid.values[row][col] = 0

def is_partial_vertical_line(grid: ColoredGrid, col: int) -> bool:
    """Checks if there's a partial vertical line in the given column."""
    colors = [grid.values[row][col] for row in range(3)]
    return colors.count(0) == 1 and len(set(colors) - {0}) == 1

def complete_vertical_line(grid: ColoredGrid, col: int):
    """Completes a partial vertical line in the given column."""
    color = max(set(grid.values[row][col] for row in range(3)) - {0})
    for row in range(3):
        if grid.values[row][col] == 0:
            grid.values[row][col] = color

def is_partial_horizontal_line(grid: ColoredGrid, col: int) -> bool:
    """Checks if there's a partial horizontal line starting from the given column."""
    return any(sum(1 for c in range(col, min(col+3, grid.get_dimensions()[1])) if grid.values[row][c] == color) == 2
               for row in range(3)
               for color in set(grid.values[row][c] for c in range(col, min(col+3, grid.get_dimensions()[1]))) - {0})

def complete_horizontal_line(grid: ColoredGrid, col: int):
    """Completes a partial horizontal line starting from the given column."""
    for row in range(3):
        colors = [grid.values[row][c] for c in range(col, min(col+3, grid.get_dimensions()[1]))]
        if len(set(colors) - {0}) == 1 and colors.count(0) == 1:
            color = max(set(colors) - {0})
            for c in range(col, min(col+3, grid.get_dimensions()[1])):
                if grid.values[row][c] == 0:
                    grid.values[row][c] = color
def get_bounding_box(positions: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    top = min(r for r, _ in positions)
    left = min(c for _, c in positions)
    bottom = max(r for r, _ in positions)
    right = max(c for _, c in positions)
    return (top, left, bottom, right)

def compress_vertically(positions: List[Tuple[int, int]], bounding_box: Tuple[int, int, int, int]) -> List[List[int]]:
    top, left, bottom, right = bounding_box
    height = bottom - top + 1
    width = right - left + 1
    
    if height <= 3:
        return [[1 if (r + top, c + left) in positions else 0 for c in range(width)] for r in range(height)]
    
    compressed = [[0 for _ in range(width)] for _ in range(3)]
    for r in range(3):
        for c in range(width):
            if any((r + i * 3 + top, c + left) in positions for i in range((height + 2) // 3)):
                compressed[r][c] = 1
    return compressed

def idealize_shape(shape: List[List[int]], pattern: str, prominence: float) -> List[List[int]]:
    if pattern == "vertical":
        return [[1, 0, 1] for _ in range(3)]
    if pattern == "horizontal":
        return [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    if pattern == "block":
        return [[1, 1, 0], [1, 1, 1], [0, 1, 1]]
    if pattern == "corner":
        return [[1, 1, 0], [1, 0, 0], [1, 1, 0]]
    # For scattered patterns, use prominence to determine density
    if prominence > 0.5:
        return [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    else:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

def optimize_space(grid: ColoredGrid):
    cols = grid.get_dimensions()[1]
    for c in range(cols - 1, 0, -1):
        if all(grid.values[r][c] == 0 for r in range(3)):
            for r in range(3):
                grid.values[r][c:] = grid.values[r][c+1:] + [0]

def align_vertically(grid: ColoredGrid):
    cols = grid.get_dimensions()[1]
    for c in range(cols):
        column = [grid.values[r][c] for r in range(3)]
        non_zero = [color for color in column if color != 0]
        for r in range(len(non_zero)):
            grid.values[r][c] = non_zero[r]
        for r in range(len(non_zero), 3):
            grid.values[r][c] = 0
