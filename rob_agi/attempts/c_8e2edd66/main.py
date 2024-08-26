from rob_agi.colored_grid import ColoredGrid

def solve_8e2edd66(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 3x3 input grid into a 9x9 output grid by expanding each cell into a 3x3 subgrid.
    
    The transformation follows these color-specific rules:
    1. Each non-zero cell in the input expands to fill the corners of its corresponding 3x3 subgrid in the output.
    2. For 9 (brown):
       - Add horizontal and vertical connections between adjacent 9's.
       - Fill L-shaped patterns.
       - Add diagonal connections in both bottom-right and top-left directions.
    3. For 8 (sky blue):
       - Add diagonal connections between adjacent 8's in both directions.
       - Connect non-adjacent 8's with minimal paths, maintaining symmetry.
    4. For 7 (orange):
       - If all four corners of the input grid are 7's, fill the center cell of the output grid with 7.
       - Create minimal paths connecting all 7's, using orthogonal or diagonal moves.
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
                            output_values[i*3+1][j*3+2] = output_values[i*3+2][j*3+1] = 9
                    # Diagonal connection (bottom-right)
                    if i < 2 and j < 2 and input_grid.values[i+1][j+1] == 9:
                        output_values[i*3+2][j*3+2] = 9
                    # Diagonal connection (top-left)
                    if i > 0 and j > 0 and input_grid.values[i-1][j-1] == 9:
                        output_values[i*3][j*3] = 9

    def process_8s():
        eight_positions = [(i, j) for i in range(3) for j in range(3) if input_grid.values[i][j] == 8]
        for i, j in eight_positions:
            # Diagonal connections
            if (i+1, j+1) in eight_positions:  # Bottom-right
                output_values[i*3+2][j*3+2] = 8
            if (i-1, j+1) in eight_positions:  # Top-right
                output_values[i*3][j*3+2] = 8
            if (i+1, j-1) in eight_positions:  # Bottom-left
                output_values[i*3+2][j*3] = 8
            if (i-1, j-1) in eight_positions:  # Top-left
                output_values[i*3][j*3] = 8
        
        # Connect non-adjacent 8's
        if len(eight_positions) > 1:
            for idx, (i1, j1) in enumerate(eight_positions):
                for i2, j2 in eight_positions[idx+1:]:
                    if abs(i1-i2) + abs(j1-j2) > 1:  # Not adjacent
                        # Calculate the middle point
                        mid_i, mid_j = (i1+i2)//2, (j1+j2)//2
                        # Connect through the edges of subgrids
                        if i1 != i2 and j1 != j2:  # Diagonal
                            output_values[mid_i*3+1][mid_j*3+1] = 8
                        elif i1 == i2:  # Same row
                            output_values[i1*3+1][min(j1,j2)*3+2] = 8
                        else:  # Same column
                            output_values[min(i1,i2)*3+2][j1*3+1] = 8

    def process_7s():
        corners = [input_grid.values[0][0], input_grid.values[0][2], input_grid.values[2][0], input_grid.values[2][2]]
        if all(corner == 7 for corner in corners):
            output_values[4][4] = 7
        
        seven_positions = [(i, j) for i in range(3) for j in range(3) if input_grid.values[i][j] == 7]
        for idx, (i1, j1) in enumerate(seven_positions):
            for i2, j2 in seven_positions[idx+1:]:
                di, dj = i2-i1, j2-j1
                steps = max(abs(di), abs(dj))
                for step in range(1, steps):
                    r = i1*3 + (di*step*3)//steps + 1
                    c = j1*3 + (dj*step*3)//steps + 1
                    output_values[r][c] = 7

    # Process colors in descending order
    process_9s()
    process_8s()
    process_7s()

    return ColoredGrid(values=output_values)
