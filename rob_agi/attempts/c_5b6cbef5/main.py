from rob_agi.colored_grid import ColoredGrid

def solve_5b6cbef5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform a 4x4 input grid into a 16x16 output grid by expanding the pattern.
    
    The solution involves the following steps:
    1. Analyze the input 4x4 grid to identify key patterns, color distributions, and edge characteristics.
    2. Create a 16x16 output grid by strategically expanding the original pattern.
    3. Handle different pattern types:
       - For sparse patterns, use direct mapping.
       - For dense patterns, replicate and expand while preserving structure.
    4. Preserve edge patterns, especially for grids with distinct edge columns or rows.
    5. Maintain color consistency and preserve any all-zero rows or columns from the original input.
    6. Refine the expanded pattern to ensure logical connections and pattern continuity.
    
    This process creates a larger pattern that preserves the input's structure and characteristics
    while expanding it across the 16x16 grid in a way that maintains visual coherence and pattern integrity.
    """
    def analyze_input(grid):
        non_zero_count = sum(1 for row in grid for cell in row if cell != 0)
        edge_pattern = any(grid[i][0] != 0 or grid[i][3] != 0 or grid[0][i] != 0 or grid[3][i] != 0 for i in range(4))
        return non_zero_count, edge_pattern

    def expand_sparse(src, dest):
        for i in range(4):
            for j in range(4):
                if src[i][j] != 0:
                    dest[i*4][j*4] = dest[i*4][j*4+3] = dest[i*4+3][j*4] = dest[i*4+3][j*4+3] = src[i][j]

    def expand_dense(src, dest):
        for i in range(4):
            for j in range(4):
                for di in range(4):
                    for dj in range(4):
                        dest[i*4+di][j*4+dj] = src[i][j]

    def preserve_edges(src, dest):
        for i in range(4):
            if src[i][0] != 0:
                for di in range(4):
                    dest[i*4+di][0] = dest[i*4+di][1] = dest[i*4+di][2] = dest[i*4+di][3] = src[i][0]
            if src[i][3] != 0:
                for di in range(4):
                    dest[i*4+di][-4] = dest[i*4+di][-3] = dest[i*4+di][-2] = dest[i*4+di][-1] = src[i][3]
            if src[0][i] != 0:
                for dj in range(4):
                    dest[0][i*4+dj] = dest[1][i*4+dj] = dest[2][i*4+dj] = dest[3][i*4+dj] = src[0][i]
            if src[3][i] != 0:
                for dj in range(4):
                    dest[-4][i*4+dj] = dest[-3][i*4+dj] = dest[-2][i*4+dj] = dest[-1][i*4+dj] = src[3][i]

    # Create the 16x16 output grid
    output = [[0 for _ in range(16)] for _ in range(16)]

    non_zero_count, edge_pattern = analyze_input(input_grid.values)

    if non_zero_count <= 4:
        expand_sparse(input_grid.values, output)
    else:
        expand_dense(input_grid.values, output)

    if edge_pattern:
        preserve_edges(input_grid.values, output)

    # Preserve all-zero rows and columns
    for i in range(4):
        if all(input_grid.values[i][j] == 0 for j in range(4)):
            for j in range(16):
                output[i*4][j] = output[i*4+1][j] = output[i*4+2][j] = output[i*4+3][j] = 0
        if all(input_grid.values[j][i] == 0 for j in range(4)):
            for j in range(16):
                output[j][i*4] = output[j][i*4+1] = output[j][i*4+2] = output[j][i*4+3] = 0

    return ColoredGrid(values=output)
