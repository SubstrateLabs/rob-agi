from rob_agi.colored_grid import ColoredGrid


def solve_776ffc46(input: ColoredGrid) -> ColoredGrid:
    def is_3x3_square(grid, row, col, color):
        if row + 2 >= grid.get_dimensions()[0] or col + 2 >= grid.get_dimensions()[1]:
            return False
        for i in range(3):
            for j in range(3):
                if grid.get_cell(row + i, col + j) != color:
                    return False
        return True

    def replace_3x3_squares(grid, from_color, to_color):
        output = grid.deep_copy()
        height, width = grid.get_dimensions()
        for row in range(height):
            for col in range(width):
                if is_3x3_square(grid, row, col, from_color):
                    for i in range(3):
                        for j in range(3):
                            output.set_cell(row + i, col + j, to_color)
        return output

    # First, replace 3x3 blue squares with red
    intermediate = replace_3x3_squares(input, 1, 2)
    
    # Then, replace 3x3 red squares with green
    output = replace_3x3_squares(intermediate, 2, 3)

    return output
