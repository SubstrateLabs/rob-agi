from rob_agi.colored_grid import ColoredGrid

def solve_d5c634a2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 3x6 output grid based on the following rules:
    1. Divides the input grid into 3 horizontal thirds.
    2. For each third, divides it into 6 vertical sections.
    3. For each section:
       - If a horizontal red line (at least 3 consecutive red squares) is found, 
         set the corresponding output cell to green (3).
       - If no horizontal line is found, but there's any red square in the section, 
         set the output cell to blue (1).
       - Otherwise, the output cell remains black (0).
    """
    input_height, input_width = input_grid.get_dimensions()
    third_height = -(-input_height // 3)  # Ceiling division to include all rows
    section_width = -(-input_width // 6)  # Ceiling division to include all columns

    output_values = [[0 for _ in range(6)] for _ in range(3)]

    for third in range(3):
        start_row = third * third_height
        end_row = min((third + 1) * third_height, input_height)

        for section in range(6):
            start_col = section * section_width
            end_col = min((section + 1) * section_width, input_width)

            has_horizontal_line = False
            has_red = False

            for row in range(start_row, end_row):
                for col in range(start_col, end_col - 2):
                    if (col + 2 < input_width and 
                        input_grid.values[row][col] == input_grid.values[row][col+1] == input_grid.values[row][col+2] == 2):
                        has_horizontal_line = True
                        break
                if has_horizontal_line:
                    break

            if not has_horizontal_line:
                for row in range(start_row, end_row):
                    for col in range(start_col, end_col):
                        if input_grid.values[row][col] == 2:
                            has_red = True
                            break
                    if has_red:
                        break

            if has_horizontal_line:
                output_values[third][section] = 3
            elif has_red:
                output_values[third][section] = 1

    return ColoredGrid(values=output_values)
