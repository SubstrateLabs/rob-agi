from rob_agi.colored_grid import ColoredGrid

def solve_67a423a3(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by identifying the intersection of vertical and horizontal non-zero lines,
    and applying a yellow (4) 3x3 square around it, preserving the central vertical line and only the
    intersection cell of the horizontal line.
    
    1. Find the intersection point of the main vertical and horizontal non-zero lines.
    2. Apply a 3x3 yellow (4) square around the intersection, preserving the central vertical line
       and only the intersection cell of the horizontal line.
    3. Return the transformed grid.
    """
    def find_intersection(grid: ColoredGrid) -> tuple[int, int]:
        rows, cols = grid.get_dimensions()
        vertical_line = next(col for col in range(cols) if all(grid.get_cell(row, col) != 0 for row in range(rows)))
        horizontal_line = next(row for row in range(rows) if all(grid.get_cell(row, col) != 0 for col in range(cols)))
        return horizontal_line, vertical_line

    def apply_transformation(grid: ColoredGrid, row: int, col: int) -> ColoredGrid:
        result = grid.deep_copy()
        rows, cols = grid.get_dimensions()
        
        # Apply 3x3 yellow square, preserving the central vertical line and intersection cell
        for r in range(max(0, row - 1), min(rows, row + 2)):
            for c in range(max(0, col - 1), min(cols, col + 2)):
                if c == col:
                    # Preserve the central vertical line
                    result.set_cell(r, c, grid.get_cell(r, c))
                elif r == row and c == col:
                    # Preserve the intersection cell
                    result.set_cell(r, c, grid.get_cell(r, c))
                else:
                    # Set other cells in the 3x3 area to yellow (4)
                    result.set_cell(r, c, 4)
        
        return result

    intersection_row, intersection_col = find_intersection(input_grid)
    return apply_transformation(input_grid, intersection_row, intersection_col)
