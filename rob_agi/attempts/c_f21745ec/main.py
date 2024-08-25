from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_f21745ec(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Removes orange (7) shapes and small shapes (less than 5x5).
    2. Fills hollow shapes with a recursive pattern based on their outline.
    3. Leaves already filled shapes unchanged.

    The function identifies distinct shapes, processes each shape according to the rules,
    and returns the transformed grid.
    """
    output_grid = input_grid.deep_copy()
    rows, cols = output_grid.get_dimensions()

    def find_shapes() -> List[List[Tuple[int, int]]]:
        shapes = []
        visited = set()
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and output_grid.get_cell(r, c) != 0:
                    shape = output_grid.find_connected_regions(output_grid.get_cell(r, c))[0]
                    shapes.append(shape)
                    visited.update(shape)
        return shapes

    def should_remove_shape(shape: List[Tuple[int, int]], color: int) -> bool:
        return color == 7 or len(shape) < 25  # 5x5 = 25 cells

    def is_shape_filled(shape: List[Tuple[int, int]]) -> bool:
        return all(output_grid.get_cell(r, c) != 0 for r, c in shape)

    def get_shape_outline(shape: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        outline = []
        for r, c in shape:
            if any((r+dr, c+dc) not in shape for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]):
                outline.append((r, c))
        return outline

    def recursive_fill(shape: List[Tuple[int, int]], color: int):
        if len(shape) < 9:  # 3x3 minimum size for recursive fill
            return

        outline = get_shape_outline(shape)
        inner_cells = [cell for cell in shape if cell not in outline]

        if not inner_cells:
            return

        for r, c in inner_cells:
            output_grid.set_cell(r, c, color)

        new_shapes = output_grid.find_connected_regions(0)
        for new_shape in new_shapes:
            if all(cell in shape for cell in new_shape):
                recursive_fill(new_shape, color)

    shapes = find_shapes()
    for shape in shapes:
        color = output_grid.get_cell(shape[0][0], shape[0][1])
        if should_remove_shape(shape, color):
            for r, c in shape:
                output_grid.set_cell(r, c, 0)
        elif not is_shape_filled(shape):
            recursive_fill(shape, color)

    return output_grid
