from rob_agi.colored_grid import ColoredGrid

def solve_992798f6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by connecting two colored squares (blue and red) with a green line.
    The line starts adjacent to the top square, maximizes diagonal movement when possible,
    then moves vertically or horizontally as needed, and ends adjacent to the bottom square.
    The path adapts based on the relative positions of the squares, handling various edge cases.

    1. Identify colored squares
    2. Determine top and bottom squares
    3. Choose the starting point based on relative positions
    4. Generate the adaptive path with maximized diagonal movement
    5. Handle edge cases (adjacent squares, same column/row)
    6. Create output grid with the green line
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
    
    # Step 3: Choose the starting point
    rows, cols = input_grid.get_dimensions()
    if top_pos[0] == 0:
        start = (top_pos[0] + 1, top_pos[1])
    elif top_pos[0] == cols - 1:
        start = (top_pos[0] - 1, top_pos[1])
    else:
        start = (top_pos[0] + 1, top_pos[1] + 1) if bottom_pos[0] >= top_pos[0] else (top_pos[0] - 1, top_pos[1] + 1)
    
    # Step 4: Generate the adaptive path
    path = []
    current = start
    dx = bottom_pos[0] - start[0]
    dy = bottom_pos[1] - start[1]

    # Handle edge case for adjacent squares
    if dx == 0 and dy == 0:
        mid_x = (top_pos[0] + bottom_pos[0]) // 2
        mid_y = (top_pos[1] + bottom_pos[1]) // 2
        path.append((mid_x, mid_y))
    else:
        while abs(dx) > 0 or dy > 0:
            path.append(current)
            if abs(dx) > 0 and dy > 0:
                # Move diagonally
                current = (current[0] + (1 if dx > 0 else -1), current[1] + 1)
                dx += -1 if dx > 0 else 1
                dy -= 1
            elif dy > 0:
                # Move vertically
                current = (current[0], current[1] + 1)
                dy -= 1
            else:
                # Move horizontally
                current = (current[0] + (1 if dx > 0 else -1), current[1])
                dx += -1 if dx > 0 else 1

    # Step 5: Create output grid with the green line
    output_grid = input_grid.deep_copy()
    for x, y in path:
        output_grid.values[y][x] = 3  # Set to green

    return output_grid
