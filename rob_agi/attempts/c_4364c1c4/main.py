from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_4364c1c4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by moving shapes based on their vertical position:
    1. Identifies the background color as the most frequent color.
    2. Finds all distinct shapes (connected regions of non-background colors).
    3. Sorts shapes from top to bottom.
    4. Moves shapes:
       - Topmost shape: 1 cell left
       - Second from top: 1 cell right (if exists)
       - Middle shapes: 1 cell left
       - Bottommost shape: 1 cell down and 2 cells right
    5. Applies movements while keeping shapes within grid bounds.
    """
    # Step 1: Identify background color
    flattened = [cell for row in input_grid.values for cell in row]
    background_color = Counter(flattened).most_common(1)[0][0]

    # Step 2: Find shapes
    shapes = []
    for color in set(flattened) - {background_color}:
        shapes.extend(input_grid.find_connected_regions(color))

    # Step 3: Sort shapes
    shapes.sort(key=lambda shape: min(cell[0] for cell in shape))

    # Step 4 & 5: Process shapes and implement movement
    new_grid = ColoredGrid(values=[[background_color for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])

    for i, shape in enumerate(shapes):
        if i == 0:  # Topmost shape
            dx, dy = -1, 0
        elif i == 1 and len(shapes) > 2:  # Second from top (if more than 2 shapes)
            dx, dy = 1, 0
        elif i == len(shapes) - 1:  # Bottommost shape
            dx, dy = 2, 1
        else:  # Middle shapes
            dx, dy = -1, 0

        for row, col in shape:
            new_row, new_col = row + dy, col + dx
            if 0 <= new_row < new_grid.num_rows and 0 <= new_col < new_grid.num_cols:
                new_grid.values[new_row][new_col] = input_grid.values[row][col]

    return new_grid
