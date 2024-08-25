from rob_agi.colored_grid import ColoredGrid

def solve_d5c634a2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 3x6 output grid based on the following rules:
    1. Divides the input grid into 3 horizontal thirds.
    2. For each third:
       - Checks the left half for a horizontal line of 3+ red squares. If found, sets output[third][0] to green (3).
       - Checks the right half for a horizontal line of 3+ red squares. If found, sets output[third][1] to green (3).
       - Divides the third into 3 vertical sections and checks each (except the leftmost) for any red squares.
         If found, sets the corresponding output cell to blue (1).
    3. All other cells in the output remain black (0).
    """
    input_height, input_width = input_grid.get_dimensions()
    third_height = -(-input_height // 3)  # Ceiling division
    output = [[0 for _ in range(6)] for _ in range(3)]

    for third in range(3):
        start_row = third * third_height
        end_row = min((third + 1) * third_height, input_height)
        mid_col = input_width // 2

        # Process left half
        if has_horizontal_line(input_grid, start_row, end_row, 0, mid_col):
            output[third][0] = 3

        # Process right half
        if has_horizontal_line(input_grid, start_row, end_row, mid_col, input_width):
            output[third][1] = 3

        # Process three vertical sections
        section_width = -(-input_width // 3)  # Ceiling division
        for section in range(3):
            start_col = section * section_width
            end_col = min((section + 1) * section_width, input_width)
            
            # Skip the leftmost section (already processed as left half)
            if section > 0:
                if has_any_red(input_grid, start_row, end_row, start_col, end_col):
                    output[third][section + 1] = 1

    return ColoredGrid(values=output)

def has_horizontal_line(grid, start_row, end_row, start_col, end_col):
    for row in range(start_row, end_row):
        count = 0
        for col in range(start_col, end_col):
            if grid.values[row][col] == 2:  # Red
                count += 1
                if count >= 3:
                    return True
            else:
                count = 0
    return False

def has_any_red(grid, start_row, end_row, start_col, end_col):
    return any(grid.values[row][col] == 2 
               for row in range(start_row, end_row) 
               for col in range(start_col, end_col))
