from rob_agi.colored_grid import ColoredGrid

def solve_a5313dff(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the a5313dff challenge by filling the interior of red rectangles with blue,
    while preserving any existing non-black colors inside the rectangles.
    
    The function identifies all red rectangles in the input grid and fills their
    interiors with blue (1), except for cells that already have a non-black color.
    Multiple, possibly nested rectangles are handled correctly.
    """
    def find_rectangles(grid):
        height, width = grid.get_dimensions()
        rectangles = []
        for top in range(height):
            for left in range(width):
                if grid.get_cell(top, left) == 2:
                    for bottom in range(top, height):
                        for right in range(left, width):
                            if all(grid.get_cell(r, c) == 2 for r in (top, bottom) for c in range(left, right + 1)) and \
                               all(grid.get_cell(r, c) == 2 for c in (left, right) for r in range(top, bottom + 1)):
                                rectangles.append((top, left, bottom, right))
        return sorted(rectangles, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))

    def fill_rectangle(grid, rect):
        top, left, bottom, right = rect
        for r in range(top + 1, bottom):
            for c in range(left + 1, right):
                if grid.get_cell(r, c) == 0:
                    grid.set_cell(r, c, 1)

    result = input_grid.deep_copy()
    rectangles = find_rectangles(result)
    
    for rect in rectangles:
        fill_rectangle(result, rect)

    return result
