from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_ed98d772(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 6x6 output grid with the following steps:
    1. Identify the frame color (most common) and zigzag color (second most common) from the input.
    2. Create a 6x6 grid, copying the input to all four quadrants.
    3. Draw a zigzag pattern with the zigzag color.
    4. Apply the frame pattern to separate the quadrants.
    5. Preserve the original input in the top-left quadrant.
    """
    # Analyze the input grid
    color_counts = Counter(cell for row in input_grid.values for cell in row)
    frame_color, zigzag_color = sorted(color_counts.items(), key=lambda x: (-x[1], x[0]))[:2]
    frame_color, zigzag_color = frame_color[0], zigzag_color[0]

    # Create the initial 6x6 output grid
    output_grid = [[0 for _ in range(6)] for _ in range(6)]
    for i in range(2):
        for j in range(2):
            for r in range(3):
                for c in range(3):
                    output_grid[i*3+r][j*3+c] = input_grid.values[r][c]

    # Apply the zigzag pattern
    zigzag_path = [(0,3), (0,4), (0,5), (1,5), (2,5), (2,4), (2,3), (3,3), (3,4), (3,5), (4,5), (5,5), (5,4), (5,3)]
    for r, c in zigzag_path:
        output_grid[r][c] = zigzag_color

    # Apply the frame pattern
    frame_positions = [(1,3), (2,4), (3,2), (4,3)]
    for r, c in frame_positions:
        output_grid[r][c] = frame_color

    # Preserve the original input in the top-left quadrant
    for r in range(3):
        for c in range(3):
            output_grid[r][c] = input_grid.values[r][c]

    return ColoredGrid(values=output_grid)
