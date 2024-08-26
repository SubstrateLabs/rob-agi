from rob_agi.colored_grid import ColoredGrid

def solve_8e2edd66(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 output grid by expanding each cell into a 3x3 subgrid.
    
    The transformation follows these color-specific rules:
    1. Each non-zero cell in the input expands to fill the corners of its corresponding 3x3 subgrid in the output.
    2. For 9 (brown):
       - Add horizontal and vertical connections between adjacent 9's.
       - Fill L-shaped patterns.
       - Add diagonal connections.
    3. For 8 (sky blue):
       - Add only diagonal connections between adjacent 8's.
    4. For 7 (orange):
       - No additional connections.
       - If all four corners of the input grid are 7's, fill the center cell of the output grid with 7.
    5. All other positions in the output grid remain zero (black).

    This creates a pattern that preserves the structure of the input while expanding it into a larger, more intricate grid.
    """
    # Create a new 9x9 grid filled with zeros
    output_values = [[0 for _ in range(9)] for _ in range(9)]
    
    def set_corners(row, col, value):
        for r in [row*3, row*3+2]:
            for c in [col*3, col*3+2]:
                output_values[r][c] = value

    # Expand each input cell
    for i in range(3):
        for j in range(3):
            v = input_grid.values[i][j]
            if v != 0:
                set_corners(i, j, v)

    def process_9s():
        for i in range(3):
            for j in range(3):
                if input_grid.values[i][j] == 9:
                    # Horizontal connection
                    if j < 2 and input_grid.values[i][j+1] == 9:
                        output_values[i*3+1][j*3+2] = 9
                    # Vertical connection
                    if i < 2 and input_grid.values[i+1][j] == 9:
                        output_values[i*3+2][j*3+1] = 9
                    # L-shaped pattern
                    if i < 2 and j < 2:
                        if input_grid.values[i][j+1] == 9 and input_grid.values[i+1][j] == 9:
                            output_values[i*3+1][j*3] = output_values[i*3+2][j*3] = 9
                            output_values[i*3][j*3+1] = output_values[i*3][j*3+2] = 9
                    # Diagonal connection
                    if i < 2 and j < 2 and input_grid.values[i+1][j+1] == 9:
                        output_values[i*3+2][j*3+2] = 9

    def process_8s():
        for i in range(2):
            for j in range(2):
                if input_grid.values[i][j] == 8 and input_grid.values[i+1][j+1] == 8:
                    output_values[i*3+2][j*3+2] = 8
                if input_grid.values[i][j+1] == 8 and input_grid.values[i+1][j] == 8:
                    output_values[i*3+2][(j+1)*3] = 8

    def process_7s():
        corners = [input_grid.values[0][0], input_grid.values[0][2], input_grid.values[2][0], input_grid.values[2][2]]
        if all(corner == 7 for corner in corners):
            output_values[4][4] = 7

    # Process colors in descending order
    process_9s()
    process_8s()
    process_7s()

    return ColoredGrid(values=output_values)
