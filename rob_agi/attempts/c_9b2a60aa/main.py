from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9b2a60aa(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by replicating the largest non-zero shape
    in the input grid. The replication is done vertically by default, or horizontally
    if the shape touches the bottom of the grid. The colors and positions for replication
    are determined by non-zero values in the leftmost column. The original shape and
    single cells remain unchanged. Replication stops if it would exceed grid boundaries
    or overlap with existing non-zero cells.
    """
    # Find the largest shape
    original_shape = find_largest_shape(input_grid)
    if not original_shape:
        return input_grid  # No shape to replicate

    # Determine replication direction
    direction = get_replication_direction(original_shape, input_grid.get_dimensions())

    # Get replication colors and positions
    replication_info = get_replication_colors_and_positions(input_grid)

    # Create output grid
    output_grid = input_grid.deep_copy()

    # Calculate bounding box of the original shape
    min_row = min(r for r, _ in original_shape)
    max_row = max(r for r, _ in original_shape)
    min_col = min(c for _, c in original_shape)
    max_col = max(c for _, c in original_shape)
    shape_height = max_row - min_row + 1
    shape_width = max_col - min_col + 1

    # Perform replication
    for color, position in replication_info:
        if direction == 'horizontal':
            start_row = max_row
            start_col = position
        else:
            start_row = position
            start_col = input_grid.num_cols - shape_width

        if can_place_shape(output_grid, original_shape, (start_row, start_col)):
            place_shape(output_grid, original_shape, (start_row, start_col), color)

    return output_grid

def find_largest_shape(grid: ColoredGrid) -> List[Tuple[int, int]]:
    largest_shape = []
    visited = set()

    for r in range(grid.num_rows):
        for c in range(grid.num_cols):
            if grid.get_cell(r, c) != 0 and (r, c) not in visited:
                shape = grid.find_connected_regions(grid.get_cell(r, c))[0]
                if len(shape) > len(largest_shape):
                    largest_shape = shape
                visited.update(shape)

    return largest_shape

def get_replication_direction(shape: List[Tuple[int, int]], grid_dimensions: Tuple[int, int]) -> str:
    max_row = max(r for r, _ in shape)
    return 'horizontal' if max_row == grid_dimensions[0] - 1 else 'vertical'

def get_replication_colors_and_positions(grid: ColoredGrid) -> List[Tuple[int, int]]:
    return [(grid.get_cell(r, 0), r) for r in range(grid.num_rows) if grid.get_cell(r, 0) != 0]

def can_place_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], position: Tuple[int, int]) -> bool:
    min_row = min(r for r, _ in shape)
    min_col = min(c for _, c in shape)
    for r, c in shape:
        new_r, new_c = r - min_row + position[0], c - min_col + position[1]
        if new_r < 0 or new_r >= grid.num_rows or new_c < 0 or new_c >= grid.num_cols:
            return False
        if grid.get_cell(new_r, new_c) != 0:
            return False
    return True

def place_shape(grid: ColoredGrid, shape: List[Tuple[int, int]], position: Tuple[int, int], color: int):
    min_row = min(r for r, _ in shape)
    min_col = min(c for _, c in shape)
    for r, c in shape:
        new_r, new_c = r - min_row + position[0], c - min_col + position[1]
        grid.set_cell(new_r, new_c, color)
