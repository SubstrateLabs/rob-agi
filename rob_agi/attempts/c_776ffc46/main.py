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

    output = input.deep_copy()
    height, width = input.get_dimensions()

    for row in range(height):
        for col in range(width):
            if is_3x3_square(input, row, col, 1):
                for i in range(3):
                    for j in range(3):
                        output.set_cell(row + i, col + j, 2)
            elif is_3x3_square(input, row, col, 2):
                for i in range(3):
                    for j in range(3):
                        output.set_cell(row + i, col + j, 3)
    return output
