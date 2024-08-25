from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_af24b4cc(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation challenge by:
    1. Dividing the input 10x9 grid into six 3x3 blocks, ignoring the border of 0s.
    2. For each 3x3 block, determining the most frequent non-zero color.
    3. Constructing a 5x4 output grid with the first and last rows being all 0s.
    4. Placing the most frequent colors from each 3x3 block into the 2x3 center of the output grid.
    """
    def most_frequent_nonzero(subgrid):
        colors = [cell for row in subgrid for cell in row if cell != 0]
        return Counter(colors).most_common(1)[0][0] if colors else 0

    input_values = input_grid.values
    output_values = [[0 for _ in range(5)] for _ in range(4)]

    for i in range(2):
        for j in range(3):
            subgrid = [row[1+j*3:4+j*3] for row in input_values[1+i*4:4+i*4]]
            output_values[i+1][j+1] = most_frequent_nonzero(subgrid)

    return ColoredGrid(values=output_values)
