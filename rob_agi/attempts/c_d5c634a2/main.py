from rob_agi.colored_grid import ColoredGrid

def solve_d5c634a2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 3x6 output grid based on the following rules:
    1. Divides the input grid into 18 regions (3 rows, 6 columns).
    2. For each region:
       - If a horizontal red line (at least 3 consecutive red squares) is found, 
         set the corresponding output cell to green (3).
       - If no horizontal line is found, but there's any red square in the region, 
         set the output cell to blue (1).
       - Otherwise, the output cell remains black (0).
    """
    input_height, input_width = input_grid.get_dimensions()
    region_height = input_height // 3
    region_width = input_width // 6

    output_values = [[0 for _ in range(6)] for _ in range(3)]

    for row in range(3):
        for col in range(6):
            top = row * region_height
            left = col * region_width
            subgrid = input_grid.extract_subgrid(top, left, region_height, region_width)

            # Check for horizontal red line
            horizontal_line_found = False
            for r in range(region_height):
                for c in range(region_width - 2):
                    if subgrid.values[r][c] == subgrid.values[r][c+1] == subgrid.values[r][c+2] == 2:
                        output_values[row][col] = 3
                        horizontal_line_found = True
                        break
                if horizontal_line_found:
                    break

            # If no horizontal line, check for any red square
            if not horizontal_line_found:
                if any(2 in row for row in subgrid.values):
                    output_values[row][col] = 1

    return ColoredGrid(values=output_values)
