from rob_agi.colored_grid import ColoredGrid

def solve_8e2edd66(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 output grid by expanding each cell into a 3x3 subgrid.
    
    The transformation follows these rules:
    1. Each non-zero cell in the input expands to fill the corners of its corresponding 3x3 subgrid in the output.
    2. The center of each 3x3 subgrid in the output always remains 0 (black).
    3. Adjacent non-zero cells in the input create connections along their shared edges in the output.
    4. L-shaped patterns of three non-zero cells in the input create filled L-shapes in the output, except for subgrid centers.
    5. Diagonal connections are created when diagonally adjacent non-zero cells are present in the input.
    6. All other positions in the output grid remain zero (black).

    This creates a pattern that preserves the structure of the input while expanding it into a larger, more intricate grid.
    """
    # Create a new 9x9 grid filled with zeros
    output_values = [[0 for _ in range(9)] for _ in range(9)]
    
    # Helper function to set corners of a 3x3 subgrid
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

    # Handle horizontal and vertical connections
    for i in range(3):
        for j in range(3):
            v = input_grid.values[i][j]
            if v != 0:
                # Right neighbor
                if j < 2 and input_grid.values[i][j+1] != 0:
                    output_values[i*3+1][j*3+2] = v
                # Bottom neighbor
                if i < 2 and input_grid.values[i+1][j] != 0:
                    output_values[i*3+2][j*3+1] = v

    # Handle L-shaped patterns and diagonal connections
    for i in range(2):
        for j in range(2):
            top_left = input_grid.values[i][j]
            top_right = input_grid.values[i][j+1]
            bottom_left = input_grid.values[i+1][j]
            bottom_right = input_grid.values[i+1][j+1]
            
            # L-shaped pattern
            if top_left != 0 and top_right != 0 and bottom_left != 0:
                output_values[i*3+1][j*3] = output_values[i*3+2][j*3] = top_left
                output_values[i*3][j*3+1] = output_values[i*3][j*3+2] = top_left
            
            # Diagonal connections
            if top_left != 0 and bottom_right != 0:
                output_values[i*3+2][j*3+2] = top_left
            if top_right != 0 and bottom_left != 0:
                output_values[i*3+2][(j+1)*3] = top_right

    # Create and return a new ColoredGrid with the output values
    return ColoredGrid(values=output_values)
