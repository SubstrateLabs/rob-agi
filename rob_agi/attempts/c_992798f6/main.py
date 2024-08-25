from rob_agi.colored_grid import ColoredGrid

def solve_992798f6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by connecting two colored squares (blue and red) with a green line.
    The line starts adjacent to the top square, moves vertically when possible,
    then diagonally if needed, and ends adjacent to the bottom square.
    The path adapts based on the relative positions of the squares.
    
    1. Identify colored squares
    2. Determine top and bottom squares
    3. Generate the adaptive path
    4. Create output grid with the green line
    """
    # Step 1: Identify colored squares
    blue_pos, red_pos = None, None
    for y, row in enumerate(input_grid.values):
        for x, cell in enumerate(row):
            if cell == 1:
                blue_pos = (x, y)
            elif cell == 2:
                red_pos = (x, y)
    
    if not blue_pos or not red_pos:
        return input_grid  # Return original grid if colored squares are not found

    # Step 2: Determine top and bottom squares
    top_pos, bottom_pos = (blue_pos, red_pos) if blue_pos[1] < red_pos[1] else (red_pos, blue_pos)
    
    # Step 3: Generate the adaptive path
    path = []
    current = (top_pos[0], top_pos[1] + 1)  # Start one step below the top square
    target_col = bottom_pos[0]

    while current[1] < bottom_pos[1] - 1:  # Stop one step before the bottom square
        path.append(current)
        if current[0] == target_col:
            current = (current[0], current[1] + 1)  # Move vertically
        else:
            dx = 1 if target_col > current[0] else -1
            current = (current[0] + dx, current[1] + 1)  # Move diagonally

    # Ensure we end adjacent to the bottom square
    if current[0] != target_col:
        path.append((current[0], current[1]))
        path.append((target_col, current[1]))
    else:
        path.append(current)

    # Step 4: Create output grid with the green line
    output_grid = input_grid.deep_copy()
    for x, y in path:
        output_grid.values[y][x] = 3  # Set to green

    return output_grid
