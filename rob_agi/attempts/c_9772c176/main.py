from rob_agi.colored_grid import ColoredGrid
from typing import Tuple, List

def solve_9772c176(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by adding yellow (4) paths around and through sky blue (8) shapes.
    
    The solution follows these steps:
    1. Find sky blue shapes in the grid.
    2. Outline each shape with yellow, prioritizing top and left sides.
    3. Draw a diagonal yellow line through each shape.
    4. Connect shapes with yellow paths.
    5. Add yellow extensions or "rays" from the outlines.
    6. Ensure a continuous yellow path from top-left to bottom-right of the grid.

    Args:
    input_grid (ColoredGrid): The input grid containing sky blue shapes.

    Returns:
    ColoredGrid: The transformed grid with added yellow paths.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def find_sky_blue_pixel() -> Tuple[int, int]:
        for r in range(rows):
            for c in range(cols):
                if output_grid.get_cell(r, c) == 8:
                    return r, c
        return -1, -1

    def is_valid(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def add_yellow_pixel(r: int, c: int):
        if is_valid(r, c) and output_grid.get_cell(r, c) == 0:
            output_grid.set_cell(r, c, 4)

    def outline_shape(start_r: int, start_c: int):
        stack = [(start_r, start_c)]
        visited = set()

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))

            for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc):
                    if output_grid.get_cell(nr, nc) == 8:
                        stack.append((nr, nc))
                    elif output_grid.get_cell(nr, nc) == 0:
                        add_yellow_pixel(nr, nc)

    def draw_diagonal(start_r: int, start_c: int, end_r: int, end_c: int):
        dx = abs(end_c - start_c)
        dy = abs(end_r - start_r)
        sx = 1 if start_c < end_c else -1
        sy = 1 if start_r < end_r else -1
        err = dx - dy

        while start_r != end_r or start_c != end_c:
            add_yellow_pixel(start_r, start_c)
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                start_c += sx
            if e2 < dx:
                err += dx
                start_r += sy

    def connect_shapes(shapes: List[Tuple[int, int]]):
        for i in range(len(shapes) - 1):
            start_r, start_c = shapes[i]
            end_r, end_c = shapes[i + 1]
            draw_diagonal(start_r, start_c, end_r, end_c)

    def add_rays(r: int, c: int):
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and output_grid.get_cell(nr, nc) == 0:
                add_yellow_pixel(nr, nc)

    # Find all sky blue shapes
    shapes = []
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 8 and all(output_grid.get_cell(r+dr, c+dc) != 8 for dr, dc in [(-1, 0), (0, -1)] if is_valid(r+dr, c+dc)):
                shapes.append((r, c))

    # Process each shape
    for shape_start in shapes:
        outline_shape(*shape_start)
        shape_end = max((r, c) for r in range(rows) for c in range(cols) if output_grid.get_cell(r, c) == 8)
        draw_diagonal(*shape_start, *shape_end)

    # Connect shapes
    connect_shapes(shapes)

    # Add rays and ensure path continuity
    for r in range(rows):
        for c in range(cols):
            if output_grid.get_cell(r, c) == 4:
                add_rays(r, c)

    # Ensure path from top-left to bottom-right
    draw_diagonal(0, 0, rows-1, cols-1)

    return output_grid
