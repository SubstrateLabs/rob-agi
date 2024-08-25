from rob_agi.colored_grid import ColoredGrid

def solve_d5c634a2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid into a 3x6 output grid based on the following rules:
    1. Divides the input grid into 18 regions (3 rows, 6 columns).
    2. For each region:
       - If a horizontal red line (at least 3 connected red squares) is found, 
         set the corresponding output cell to green (3).
       - If no horizontal line is found, but there's a red square in the 
         rightmost third of the region, set the output cell to blue (1).
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
            connected_regions = subgrid.find_connected_regions(2)
            horizontal_line_found = False
            for region in connected_regions:
                if max(c for _, c in region) - min(c for _, c in region) >= 2:  # Width of at least 3
                    output_values[row][col] = 3
                    horizontal_line_found = True
                    break

            # If no horizontal line, check rightmost third
            if not horizontal_line_found:
                rightmost_third = subgrid.extract_subgrid(0, 2*region_width//3, region_height, region_width//3)
                if any(2 in row for row in rightmost_third.values):
                    output_values[row][col] = 1

    return ColoredGrid(values=output_values)
