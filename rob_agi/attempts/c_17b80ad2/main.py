from rob_agi.colored_grid import ColoredGrid

def solve_17b80ad2(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by creating vertical lines of color based on the following rules:
    1. Non-black dots initiate vertical lines that extend downwards.
    2. Lines continue until they reach another non-black dot or the bottom of the grid.
    3. If there's a gap between colored segments in a column, it's filled with the color above.
    4. Gray dots in the bottom row are preserved regardless of lines above them.
    5. The rest of the grid remains black (0).
    """
    new_grid = input_grid.deep_copy()
    height, width = new_grid.get_dimensions()

    for col in range(width):
        line_segments = []
        for row in range(height):
            if new_grid.values[row][col] != 0:
                start_row = row
                color = new_grid.values[row][col]
                end_row = row
                while end_row < height - 1 and new_grid.values[end_row + 1][col] == 0:
                    end_row += 1
                line_segments.append((start_row, end_row, color))

        line_segments.sort(key=lambda x: x[0])

        for i, (start, end, color) in enumerate(line_segments):
            for row in range(start, end + 1):
                new_grid.values[row][col] = color
            if i < len(line_segments) - 1:
                next_start = line_segments[i + 1][0]
                if next_start > end + 1:
                    for row in range(end + 1, next_start):
                        new_grid.values[row][col] = color

        if new_grid.values[height - 1][col] == 5:
            new_grid.values[height - 1][col] = 5

    return new_grid
