from rob_agi.colored_grid import ColoredGrid

def solve_bbb1b8b6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by combining the left and right halves into a 4x4 grid.
    
    1. Extract the left half of the input grid (before the gray column).
    2. Extract the right half of the input grid (after the gray column).
    3. Prepare a 4x4 base grid from the left half, expanding or truncating if necessary.
    4. Fill in black (0) cells in the base grid with non-black colors from the right half.
    5. Return the resulting 4x4 grid as a ColoredGrid object.
    """
    # Step 1: Extract left half
    gray_index = input_grid.values[0].index(5)
    left_half = [row[:gray_index] for row in input_grid.values]

    # Step 2: Extract right half
    right_half = [row[gray_index+1:] for row in input_grid.values]

    # Step 3: Prepare 4x4 base grid
    output = []
    for i in range(4):
        if i < len(left_half):
            row = left_half[i][:4]  # Take up to 4 elements
            row += [0] * (4 - len(row))  # Pad with zeros if needed
        else:
            row = [0] * 4  # Add empty rows if left_half has fewer than 4 rows
        output.append(row)

    # Step 4: Fill in black cells with colors from the right half
    for i in range(4):
        for j in range(4):
            if output[i][j] == 0 and i < len(right_half) and j < len(right_half[i]):
                if right_half[i][j] != 0:
                    output[i][j] = right_half[i][j]

    # Step 5: Return result
    return ColoredGrid(values=output)
