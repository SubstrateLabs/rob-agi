from rob_agi.colored_grid import ColoredGrid

def solve_e21a174a(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solves the e21a174a challenge by rearranging color groups vertically.
    
    The function identifies distinct color groups in the input grid,
    calculates their vertical centers, and then rearranges them from
    top to bottom in reverse order of their original positions while
    maintaining their relative spacing.
    
    Args:
    input_grid (ColoredGrid): The input grid to be transformed.
    
    Returns:
    ColoredGrid: The transformed grid with color groups rearranged.
    """
    # Step 1: Analyze the input grid
    color_groups = {}
    for row in range(len(input_grid.values)):
        for col in range(len(input_grid.values[0])):
            color = input_grid.values[row][col]
            if color != 0:
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append((row, col))

    # Step 2: Process each color group
    processed_groups = []
    for color, cells in color_groups.items():
        min_row = min(cell[0] for cell in cells)
        max_row = max(cell[0] for cell in cells)
        min_col = min(cell[1] for cell in cells)
        max_col = max(cell[1] for cell in cells)
        vertical_center = (min_row + max_row) / 2
        processed_groups.append((color, vertical_center, (min_row, max_row, min_col, max_col), cells))

    # Step 3: Sort color groups
    processed_groups.sort(key=lambda x: x[1])

    # Step 4: Calculate new positions
    new_positions = []
    for i, group in enumerate(reversed(processed_groups)):
        if i == 0:
            new_top = group[2][0]  # Original top position
        else:
            prev_group = processed_groups[-(i)]
            gap = group[2][0] - prev_group[2][1]  # Gap from previous group
            new_top = new_positions[-1] - (group[2][1] - group[2][0]) - gap
        new_positions.append(new_top)
    new_positions.reverse()

    # Step 5: Create the output grid
    output_grid = ColoredGrid(values=[[0 for _ in range(len(input_grid.values[0]))] for _ in range(len(input_grid.values))])

    # Step 6: Place color groups in new positions
    for (color, _, (min_row, max_row, _, _), cells), new_top in zip(processed_groups, new_positions):
        for row, col in cells:
            new_row = int(new_top + (row - min_row))
            output_grid.values[new_row][col] = color

    # Step 7: Return the new ColoredGrid
    return output_grid
