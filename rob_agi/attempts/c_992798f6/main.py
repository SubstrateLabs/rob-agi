from rob_agi.colored_grid import ColoredGrid

def solve_992798f6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by connecting two colored squares (blue and red) with a green line.
    The line starts adjacent to the higher square, moves diagonally when possible,
    and ends adjacent to the lower square. The path adapts based on the relative
    positions of the squares, allowing for both diagonal and straight movements.
    
    1. Identify colored squares
    2. Determine start and end points
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

    # Step 2: Determine start and end points
    start_pos = blue_pos if blue_pos[1] <= red_pos[1] else red_pos
    end_pos = red_pos if start_pos == blue_pos else blue_pos
    
    # Step 3: Generate the adaptive path
    path = []
    current = (start_pos[0], start_pos[1] + (1 if start_pos == blue_pos else -1))
    path.append(current)

    while current != (end_pos[0], end_pos[1] - (1 if end_pos == blue_pos else -1)):
        dx = 1 if end_pos[0] > current[0] else -1 if end_pos[0] < current[0] else 0
        dy = 1 if end_pos[1] > current[1] else -1 if end_pos[1] < current[1] else 0

        if dx != 0 and dy != 0:
            current = (current[0] + dx, current[1] + dy)  # Move diagonally
        elif dx != 0:
            current = (current[0] + dx, current[1])  # Move horizontally
        elif dy != 0:
            current = (current[0], current[1] + dy)  # Move vertically
        
        path.append(current)

    # Step 4: Create output grid with the green line
    output_grid = input_grid.deep_copy()
    for x, y in path:
        output_grid.values[y][x] = 3  # Set to green

    return output_grid
