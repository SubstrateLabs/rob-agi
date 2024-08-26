from rob_agi.colored_grid import ColoredGrid
from collections import Counter

def solve_e57337a4(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 15x15 input grid into a 3x3 output grid.
    
    The function works as follows:
    1. Determines the predominant color (background) of the input grid.
    2. Extracts the top-left 9x9 section from the input grid.
    3. Divides this 9x9 section into nine 3x3 blocks.
    4. For each 3x3 block, checks if it contains any black (0) squares.
    5. Creates a 3x3 output grid where each cell corresponds to a 3x3 block:
       - If a block contains a black square, the corresponding cell is set to black (0).
       - Otherwise, the cell is set to the background color.
    
    This process preserves the relative positions of black squares while condensing
    the information from the 15x15 grid into a 3x3 summary.
    """
    # Determine the background color
    all_colors = [color for row in input_grid.values for color in row]
    background_color = Counter(all_colors).most_common(1)[0][0]
    
    # Extract the top-left 9x9 section
    section_9x9 = [row[:9] for row in input_grid.values[:9]]
    
    # Initialize the 3x3 output grid with the background color
    output_values = [[background_color for _ in range(3)] for _ in range(3)]
    
    # Process each 3x3 block in the 9x9 section
    for i in range(3):
        for j in range(3):
            # Check if there's any black square in this 3x3 block
            for x in range(3):
                for y in range(3):
                    if section_9x9[3*i + x][3*j + y] == 0:
                        output_values[i][j] = 0
                        break
                if output_values[i][j] == 0:
                    break
    
    return ColoredGrid(values=output_values)
