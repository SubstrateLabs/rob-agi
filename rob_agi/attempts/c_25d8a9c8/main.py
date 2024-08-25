from rob_agi.colored_grid import ColoredGrid

def solve_25d8a9c8(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following rules:
    1. The output grid has the same dimensions as the input grid.
    2. Starting from the top, find the first row with all identical elements.
    3. This row and all rows below it become rows of 5's in the output.
    4. All rows above become rows of 0's in the output.
    """
    height, width = input_grid.get_dimensions()
    output_values = []
    
    found_identical = False
    for row in range(height):
        input_row = [input_grid.get_cell(row, col) for col in range(width)]
        if len(set(input_row)) == 1:
            found_identical = True
        if found_identical:
            output_values.append([5] * width)
        else:
            output_values.append([0] * width)
    
    return ColoredGrid(values=output_values)
