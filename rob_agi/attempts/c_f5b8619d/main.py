from rob_agi.colored_grid import ColoredGrid

def solve_f5b8619d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by doubling its size, replicating non-zero values in 2x2 blocks,
    preserving zero values, filling empty spaces with sky color (8), and creating both horizontal
    and vertical symmetry.
    
    The transformation follows these steps:
    1. Initialize an output grid with dimensions twice that of the input, filled with sky color (8).
    2. Process each cell of the input grid:
       a. Replicate non-zero values in 2x2 blocks in the output.
       b. Preserve zero values in their corresponding positions.
    3. Create horizontal symmetry by mirroring the left half to the right half.
    4. Create vertical symmetry by mirroring the top half to the bottom half.
    """
    height, width = input_grid.get_dimensions()
    output = [[8 for _ in range(width*2)] for _ in range(height*2)]

    # Process each cell of the input grid
    for row in range(height):
        for col in range(width):
            value = input_grid.get_cell(row, col)
            output[row*2][col*2] = output[row*2][col*2+1] = value
            output[row*2+1][col*2] = output[row*2+1][col*2+1] = value

    # Create horizontal symmetry
    for row in range(height*2):
        for col in range(width):
            output[row][width*2-col-1] = output[row][col]
            output[row][width*2-col-2] = output[row][col+1]

    # Create vertical symmetry
    for row in range(height):
        output[height*2-row-1] = output[row].copy()
        output[height*2-row-2] = output[row+1].copy()

    return ColoredGrid(values=output)
