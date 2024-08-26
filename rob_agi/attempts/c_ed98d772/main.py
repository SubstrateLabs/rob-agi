from rob_agi.colored_grid import ColoredGrid

def solve_ed98d772(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 6x6 output grid by:
    1. Copying the input to the top-left quadrant.
    2. Rotating the input 90 degrees clockwise for each subsequent quadrant.
    3. Creating a frame pattern around the center using the most common non-zero color.
    4. Filling the center with zeros.
    """
    def rotate_90_clockwise(grid):
        return [list(row) for row in zip(*grid[::-1])]

    # Create the initial 6x6 output grid
    output_grid = [[0 for _ in range(6)] for _ in range(6)]

    # Copy input to all quadrants with rotations
    for i in range(2):
        for j in range(2):
            rotated = input_grid.values
            for _ in range(i * 2 + j):
                rotated = rotate_90_clockwise(rotated)
            for r in range(3):
                for c in range(3):
                    output_grid[i*3+r][j*3+c] = rotated[r][c]

    # Find the most common non-zero color
    frame_color = max((color for row in input_grid.values for color in row if color != 0), 
                      key=lambda x: sum(row.count(x) for row in input_grid.values))

    # Create the frame pattern
    frame_positions = [(1,2), (1,3), (2,1), (2,4), (3,2), (3,3), (4,1), (4,4)]
    for r, c in frame_positions:
        output_grid[r][c] = frame_color

    # Fill the center with zeros
    output_grid[2][2] = output_grid[2][3] = output_grid[3][2] = output_grid[3][3] = 0

    return ColoredGrid(values=output_grid)
