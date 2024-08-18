from rob_agi.colored_grid import ColoredGrid


def solve_776ffc46(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid based on the following rules:
    1. Blue (1) regions with more than one cell change to Red (2)
    2. Red (2) regions with more than one cell remain Red (2)
    3. Green (3) regions with more than one cell remain Green (3)
    Single-cell regions and other colors remain unchanged.

    Approach:
    1. Create a deep copy of the input grid to avoid modifying the original.
    2. Find connected regions for Blue (1) color.
    3. Transform Blue regions with more than one cell to Red (2).
    4. Return the transformed grid.
    """
    output_grid = input_grid.deep_copy()
    blue_regions = output_grid.find_connected_regions(1)  # Blue color

    for region in blue_regions:
        if len(region) > 1:
            for r, c in region:
                output_grid.set_cell(r, c, 2)  # Change to Red

    return output_grid
