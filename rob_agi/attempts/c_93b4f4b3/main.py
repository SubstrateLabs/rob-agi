from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_93b4f4b3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by compressing it to 6 columns while preserving the border and simplifying internal shapes.
    
    1. Extracts the 'U' shaped border and internal shapes.
    2. Creates a new grid with 6 columns and the same number of rows as the input.
    3. Places the border in the output grid.
    4. Simplifies internal shapes and arranges them within the border.
    5. Fills any remaining space with the border color.
    
    Returns the transformed ColoredGrid.
    """
    # Extract border and internal shapes
    border_color = input_grid.values[0][0]
    border_region = input_grid.find_connected_regions(border_color)[0]
    internal_shapes = [region for color in range(10) if color != border_color for region in input_grid.find_connected_regions(color)]

    # Create output grid
    rows = input_grid.get_dimensions()[0]
    output_grid = ColoredGrid(values=[[0 for _ in range(6)] for _ in range(rows)])

    # Place border in output grid
    for r in range(rows):
        output_grid.values[r][0] = border_color
        output_grid.values[r][5] = border_color
    for c in range(6):
        output_grid.values[0][c] = border_color
        output_grid.values[-1][c] = border_color

    # Simplify and arrange internal shapes
    simplified_shapes = [simplify_shape(shape, color) for shape, color in internal_shapes]
    simplified_shapes.sort(key=lambda x: x[1] * x[2], reverse=True)  # Sort by area

    for shape in simplified_shapes:
        color, height, width = shape
        placed = False
        for r in range(1, rows - height):
            for c in range(1, 5 - width + 1):
                if can_place_shape(output_grid, r, c, height, width):
                    place_shape(output_grid, r, c, height, width, color)
                    placed = True
                    break
            if placed:
                break

    # Fill remaining space
    for r in range(rows):
        for c in range(6):
            if output_grid.values[r][c] == 0:
                output_grid.values[r][c] = border_color

    return output_grid

def simplify_shape(shape: List[Tuple[int, int]], color: int) -> Tuple[int, int, int]:
    """Simplifies a shape to fit within the 4-column inner width."""
    area = len(shape)
    if area <= 4:
        return color, 2, 2
    elif area <= 9:
        return color, 3, 3
    else:
        return color, 2, 4

def can_place_shape(grid: ColoredGrid, row: int, col: int, height: int, width: int) -> bool:
    """Checks if a shape can be placed at the given position."""
    for r in range(row, row + height):
        for c in range(col, col + width):
            if grid.values[r][c] != 0:
                return False
    return True

def place_shape(grid: ColoredGrid, row: int, col: int, height: int, width: int, color: int) -> None:
    """Places a shape at the given position."""
    for r in range(row, row + height):
        for c in range(col, col + width):
            grid.values[r][c] = color
