from rob_agi.colored_grid import ColoredGrid

def solve_e633a9e5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 5x5 output grid by expanding each cell into a 2x2 area.
    The expansion follows these rules:
    1. Initialize a 5x5 grid with placeholder values.
    2. Map each input cell to the top-left of a 2x2 area in the output grid.
    3. Fill remaining cells based on their position:
       - Top-right: minimum of left neighbor and right input cell
       - Bottom-left: minimum of top neighbor and bottom input cell
       - Bottom-right: minimum of top, left, top-left neighbors, and corresponding input cell
    4. Always choose the smallest color value when comparing.
    """
    input_values = input_grid.values
    output_values = [[None for _ in range(5)] for _ in range(5)]

    # Map input values to output grid
    for r in range(3):
        for c in range(3):
            output_values[2*r][2*c] = input_values[r][c]

    # Fill in remaining cells
    for r in range(5):
        for c in range(5):
            if output_values[r][c] is None:
                input_r, input_c = r // 2, c // 2
                
                if r % 2 == 0 and c % 2 == 1:  # Top-right
                    left = output_values[r][c-1]
                    right_input = input_values[input_r][min(input_c+1, 2)] if input_c < 2 else left
                    output_values[r][c] = min(left, right_input)
                
                elif r % 2 == 1 and c % 2 == 0:  # Bottom-left
                    top = output_values[r-1][c]
                    bottom_input = input_values[min(input_r+1, 2)][input_c] if input_r < 2 else top
                    output_values[r][c] = min(top, bottom_input)
                
                else:  # Bottom-right
                    top = output_values[r-1][c]
                    left = output_values[r][c-1]
                    top_left = output_values[r-1][c-1]
                    input_val = input_values[input_r][input_c]
                    output_values[r][c] = min(top, left, top_left, input_val)

    return ColoredGrid(values=output_values)
