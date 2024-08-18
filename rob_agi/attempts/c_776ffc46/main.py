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
    2. Find connected regions for Blue (1), Red (2), and Green (3) colors.
    3. Transform Blue regions with more than one cell to Red (2).
    4. Keep Red and Green regions with more than one cell unchanged.
    5. Return the transformed grid.
    """
    output_grid = input_grid.deep_copy()
    
    for color in [1, 2, 3]:  # Blue, Red, Green
        regions = output_grid.find_connected_regions(color)
        for region in regions:
            if len(region) > 1:
                new_color = 2 if color == 1 else color  # Change Blue to Red, keep others
                for r, c in region:
                    output_grid.set_cell(r, c, new_color)

    return output_grid
