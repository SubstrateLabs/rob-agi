from rob_agi.colored_grid import ColoredGrid

def solve_67a423a3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the intersection of vertical and horizontal non-zero lines,
    and applying a yellow (4) 3x3 square around it, preserving all original non-zero values.
    
    1. Find the intersection point of vertical and horizontal non-zero lines.
    2. Apply a 3x3 yellow (4) square around the intersection, only changing cells that were originally 0 (black).
    3. Return the transformed grid.
    """
    def find_intersection(grid: ColoredGrid) -> tuple[int, int]:
        rows, cols = grid.get_dimensions()
        vertical_line = next(col for col in range(cols) if any(grid.get_cell(row, col) != 0 for row in range(rows)))
        horizontal_line = next(row for row in range(rows) if any(grid.get_cell(row, col) != 0 for col in range(cols)))
        return horizontal_line, vertical_line

    def apply_transformation(grid: ColoredGrid, row: int, col: int) -> ColoredGrid:
        result = grid.deep_copy()
        rows, cols = grid.get_dimensions()
        for r in range(max(0, row - 1), min(rows, row + 2)):
            for c in range(max(0, col - 1), min(cols, col + 2)):
                if result.get_cell(r, c) == 0:
                    result.set_cell(r, c, 4)  # Set only zero (black) cells in the 3x3 area to 4 (yellow)
        return result

    intersection_row, intersection_col = find_intersection(input_grid)
    return apply_transformation(input_grid, intersection_row, intersection_col)
