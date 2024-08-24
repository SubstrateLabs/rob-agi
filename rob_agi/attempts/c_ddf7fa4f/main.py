from rob_agi.colored_grid import ColoredGrid

def solve_ddf7fa4f(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the grid transformation puzzle by applying the following rules:
    1. Preserve the first row of the input grid as it contains the color mapping.
    2. Create a color mapping dictionary from the first row, associating column indices with non-zero values.
    3. Process rows from the second row onwards:
       a. Iterate through each cell in these rows.
       b. If a cell contains the value 5, replace it with the corresponding color from the mapping.
       c. If there's no corresponding color in the mapping, keep the 5 as is.
       d. Ensure continuous regions of 5s are replaced with the same color.
    4. Handle the leftmost column specially if it has a non-zero value in the first row.
    5. Return the modified grid.
    """
    output = input_grid.deep_copy()
    height, width = output.get_dimensions()
    
    # Create color mapping from the first row
    color_mapping = {col: val for col, val in enumerate(output.values[0]) if val != 0}
    
    def flood_fill(row, col, new_color):
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if 0 <= r < height and 0 <= c < width and output.get_cell(r, c) == 5:
                output.set_cell(r, c, new_color)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((r + dr, c + dc))
    
    for row in range(1, height):
        for col in range(width):
            if output.get_cell(row, col) == 5:
                new_color = None
                if col in color_mapping:
                    new_color = color_mapping[col]
                elif col == 0 and output.get_cell(0, 0) != 0:
                    new_color = output.get_cell(0, 0)
                
                if new_color is not None:
                    flood_fill(row, col, new_color)
    
    return output
