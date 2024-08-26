from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_9b2a60aa(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the grid transformation challenge by replicating the largest non-zero shape
    in the input grid. The replication is done vertically (top to bottom) by default,
    switching to horizontal (right to left) if vertical space runs out.
    Up to three replicas are created using colors from the leftmost column.
    The original shape and grid contents remain unchanged. Replication stops if it would
    exceed grid boundaries or overlap with existing non-zero cells.
    """
    original_shape = find_largest_shape(input_grid)
    if not original_shape:
        return input_grid  # No shape to replicate

    replication_colors = get_replication_colors(input_grid)
    output_grid = input_grid.deep_copy()

    shape_bounds = get_shape_bounds(original_shape)
    shape_height = shape_bounds[2] - shape_bounds[0] + 1
    shape_width = shape_bounds[3] - shape_bounds[1] + 1

    start_row = 0
    start_col = shape_bounds[1]
    direction = 'vertical'

    replicas_placed = 0
    for color in replication_colors:
        if direction == 'vertical':
            if start_row + shape_height > input_grid.num_rows:
                direction = 'horizontal'
                start_row = shape_bounds[0]
                start_col = input_grid.num_cols - shape_width
            
        if direction == 'horizontal':
            if start_col < 0:
                break

        transformed_shape = transform_shape(original_shape, direction)
        if can_place_shape(output_grid, transformed_shape, (start_row, start_col)):
            place_shape(output_grid, transformed_shape, (start_row, start_col), color)
            replicas_placed += 1
            
            if direction == 'vertical':
                start_row += shape_height
            else:
                start_col -= shape_width
        else:
            break

        if replicas_placed == 3:
            break

    return output_grid

def transform_shape(shape: List[Tuple[int, int]], direction: str) -> List[Tuple[int, int]]:
    """Transform the shape based on the replication direction."""
    min_row = min(r for r, _ in shape)
    min_col = min(c for _, c in shape)
    if direction == 'horizontal':
        return [(r, -c + min_col) for r, c in shape]
    else:
        return [(-r + min_row, -c + min_col) for r, c in shape]

def get_replication_colors(grid: ColoredGrid) -> List[int]:
    """Get up to three non-zero colors from the leftmost column."""
    colors = [grid.get_cell(r, 0) for r in range(grid.num_rows) if grid.get_cell(r, 0) != 0]
    return colors[:3]

def get_shape_bounds(shape: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    min_row = min(r for r, _ in shape)
    max_row = max(r for r, _ in shape)
    min_col = min(c for _, c in shape)
    max_col = max(c for _, c in shape)
    return (min_row, min_col, max_row, max_col)

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

def get_replication_colors(grid: ColoredGrid) -> List[int]:
    """Get up to three non-zero colors from the leftmost column."""
    colors = [grid.get_cell(r, 0) for r in range(grid.num_rows) if grid.get_cell(r, 0) != 0]
    return list(dict.fromkeys(colors))[:3]  # Remove duplicates and take up to 3 colors

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
