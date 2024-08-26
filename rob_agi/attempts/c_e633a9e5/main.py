from rob_agi.colored_grid import ColoredGrid

def solve_e633a9e5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x3 input grid into a 5x5 output grid by expanding each cell into a 2x2 area.
    The expansion follows these rules:
    1. Initialize a 5x5 grid with placeholder values.
    2. Map each input cell to the top-left of a 2x2 area in the output grid.
    3. Fill remaining cells by comparing with neighboring cells and the corresponding input cell.
    4. Always choose the smallest (or equal) color value when comparing.
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
                relevant_values = []
                
                # Check cell above
                if r > 0 and output_values[r-1][c] is not None:
                    relevant_values.append(output_values[r-1][c])
                
                # Check cell to the left
                if c > 0 and output_values[r][c-1] is not None:
                    relevant_values.append(output_values[r][c-1])
                
                # Check cell diagonally up and left
                if r > 0 and c > 0 and output_values[r-1][c-1] is not None:
                    relevant_values.append(output_values[r-1][c-1])
                
                # Always include the corresponding input grid cell
                input_r, input_c = r // 2, c // 2
                relevant_values.append(input_values[input_r][input_c])
                
                # Set the current cell to the minimum of relevant values
                output_values[r][c] = min(relevant_values)

    return ColoredGrid(values=output_values)
