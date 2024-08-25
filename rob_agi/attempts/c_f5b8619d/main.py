from rob_agi.colored_grid import ColoredGrid

def solve_f5b8619d(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transform the input grid by doubling its size, replicating non-zero values in 2x2 blocks,
    filling surrounding areas with sky color (8), and creating vertical symmetry.
    
    The transformation follows these steps:
    1. Initialize an output grid with dimensions twice that of the input.
    2. Process each column of the input grid:
       a. Replicate non-zero values in 2x2 blocks in the output.
       b. Fill surrounding areas with sky color (8) until reaching another non-zero value or grid edge.
    3. Mirror the left and right edges of the output grid.
    4. Create vertical symmetry by mirroring the top half to the bottom half.
    """
    def process_column(col):
        result = [0] * (height * 2)
        sky_fill = 8
        for i in range(height):
            if input_grid.get_cell(i, col) != 0:
                result[i*2] = result[i*2+1] = input_grid.get_cell(i, col)
                # Fill sky color above
                for j in range(i*2-1, -1, -1):
                    if result[j] != 0:
                        break
                    result[j] = sky_fill
                # Fill sky color below
                for j in range(i*2+2, height*2):
                    if result[j] != 0:
                        break
                    result[j] = sky_fill
        return result

    height, width = input_grid.get_dimensions()
    output = [[0 for _ in range(width*2)] for _ in range(height*2)]

    # Process each column
    for col in range(width):
        processed = process_column(col)
        for row in range(height*2):
            output[row][col*2] = output[row][col*2+1] = processed[row]

    # Mirror left and right edges
    for row in range(height*2):
        output[row][0] = output[row][1]
        output[row][-1] = output[row][-2]

    # Create vertical symmetry
    for row in range(height, height*2):
        output[row] = output[height*2-row-1].copy()

    return ColoredGrid(values=output)
