from rob_agi.colored_grid import ColoredGrid

def solve_0c9aba6e(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 13x4 input grid into a 6x4 output grid based on the following rules:
    1. Only considers the top 6 rows of the input grid.
    2. Divides the top 6 rows into 2x2 blocks.
    3. For each 2x2 block:
       - If the number of red (2) squares is odd, the corresponding output cell is sky blue (8).
       - If the number of red (2) squares is even (including 0), the output cell is black (0).
    4. Returns the resulting 6x4 grid.
    """
    # Extract the top 6 rows from the input grid
    input_subgrid = input_grid.extract_subgrid(0, 0, 6, 4)
    
    # Create a new 6x4 ColoredGrid for the output, initially filled with black (0)
    output_grid = ColoredGrid(values=[[0 for _ in range(4)] for _ in range(6)])
    
    # Iterate through the 2x2 blocks in the extracted top 6 rows
    for row in range(0, 6, 2):
        for col in range(0, 4, 2):
            # Count red squares in the 2x2 block
            red_count = sum(1 for i in range(2) for j in range(2) 
                            if input_subgrid.values[row+i][col+j] == 2)
            
            # Set output cell
            output_row, output_col = row // 2, col // 2
            if red_count % 2 == 1:  # odd count
                output_grid.values[output_row][output_col] = 8
            # Even count: leave as 0 (black)
    
    return output_grid
