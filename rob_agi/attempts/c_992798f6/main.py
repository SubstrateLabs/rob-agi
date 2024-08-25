from rob_agi.colored_grid import ColoredGrid
import random

def solve_992798f6(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the challenge by connecting two colored squares (blue and red) with a green line.
    The line starts and ends one square away from the colored squares, forms an L-shape with a curve,
    and follows the primary direction determined by the relative positions of the squares.
    
    1. Identify colored squares
    2. Determine start and end points
    3. Calculate primary direction
    4. Generate main segment of the path
    5. Generate curved segment
    6. Ensure correct end point
    7. Create output grid with the green line
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
    start_pos = blue_pos if blue_pos[1] > red_pos[1] else red_pos
    end_pos = red_pos if start_pos == blue_pos else blue_pos
    
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    
    start_point = (start_pos[0], start_pos[1] - 1) if abs(dy) > abs(dx) else (start_pos[0] + 1, start_pos[1])
    end_point = (end_pos[0], end_pos[1] + 1) if abs(dy) > abs(dx) else (end_pos[0] - 1, end_pos[1])

    # Step 3: Calculate primary direction
    primary_direction = 'vertical' if abs(dy) > abs(dx) else 'horizontal'

    # Step 4: Generate main segment of the path
    path = [start_point]
    current = start_point
    main_segment_length = int(max(abs(dx), abs(dy)) * 2 / 3)
    
    for _ in range(main_segment_length):
        if primary_direction == 'vertical':
            current = (current[0], current[1] + (1 if dy > 0 else -1))
        else:
            current = (current[0] + (1 if dx > 0 else -1), current[1])
        path.append(current)

    # Step 5: Generate curved segment
    while current != end_point:
        options = []
        if primary_direction == 'vertical':
            if current[0] != end_point[0]:
                options.append((current[0] + (1 if dx > 0 else -1), current[1]))
            if current[1] != end_point[1]:
                options.append((current[0], current[1] + (1 if dy > 0 else -1)))
        else:
            if current[1] != end_point[1]:
                options.append((current[0], current[1] + (1 if dy > 0 else -1)))
            if current[0] != end_point[0]:
                options.append((current[0] + (1 if dx > 0 else -1), current[1]))
        
        if random.random() < 0.2:  # 20% chance to move diagonally
            options.append((current[0] + (1 if dx > 0 else -1), current[1] + (1 if dy > 0 else -1)))
        
        current = min(options, key=lambda p: abs(p[0] - end_point[0]) + abs(p[1] - end_point[1]))
        path.append(current)

    # Step 6: Ensure correct end point
    if path[-1] != end_point:
        path.append(end_point)

    # Step 7: Create output grid with the green line
    output_grid = input_grid.deep_copy()
    for x, y in path:
        output_grid.values[y][x] = 3  # Set to green

    return output_grid
