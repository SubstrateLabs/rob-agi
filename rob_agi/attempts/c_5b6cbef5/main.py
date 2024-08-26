from rob_agi.colored_grid import ColoredGrid

def solve_5b6cbef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 4x4 input grid into a 16x16 output grid by expanding the pattern.
    
    The solution involves the following steps:
    1. Analyze the input 4x4 grid to identify key patterns and color distributions.
    2. Create a 16x16 output grid by strategically placing and expanding the original pattern.
    3. Replicate the pattern in each quadrant while preserving the overall structure.
    4. Maintain color consistency and preserve any all-zero rows or columns from the original input.
    5. Refine the expanded pattern to ensure logical connections between quadrants.
    
    This process creates a larger pattern that preserves the input's structure and characteristics
    while expanding it across the 16x16 grid in a way that maintains visual coherence and pattern integrity.
    """
    def copy_pattern(src, dest, src_row, src_col, dest_row, dest_col, size):
        for i in range(size):
            for j in range(size):
                dest[dest_row + i][dest_col + j] = src[src_row + i][src_col + j]

    def expand_pattern(src, size):
        expanded = [[0 for _ in range(size * 2)] for _ in range(size * 2)]
        for i in range(size):
            for j in range(size):
                expanded[i*2][j*2] = expanded[i*2][j*2+1] = expanded[i*2+1][j*2] = expanded[i*2+1][j*2+1] = src[i][j]
        return expanded

    # Create the 16x16 output grid
    output = [[0 for _ in range(16)] for _ in range(16)]

    # Expand the original 4x4 pattern to 8x8
    expanded_pattern = expand_pattern(input_grid.values, 4)

    # Place the expanded pattern in each quadrant
    for i in range(2):
        for j in range(2):
            copy_pattern(expanded_pattern, output, 0, 0, i*8, j*8, 8)

    # Refine the pattern by preserving all-zero rows and columns
    for i in range(4):
        if all(input_grid.values[i][j] == 0 for j in range(4)):
            for j in range(16):
                output[i*2][j] = output[i*2+1][j] = 0
        if all(input_grid.values[j][i] == 0 for j in range(4)):
            for j in range(16):
                output[j][i*2] = output[j][i*2+1] = 0

    # Ensure color consistency
    colors = set(input_grid.values[i][j] for i in range(4) for j in range(4))
    for i in range(16):
        for j in range(16):
            if output[i][j] not in colors:
                output[i][j] = 0  # Replace any new colors with black (0)

    return ColoredGrid(values=output)
