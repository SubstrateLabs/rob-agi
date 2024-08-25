from rob_agi.colored_grid import ColoredGrid

def solve_1a2e2828(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the 1a2e2828 challenge by finding the color with the most complete lines.
    
    The function counts complete horizontal and vertical lines of each color,
    then returns a 1x1 grid with the color that has the most complete lines.
    If multiple colors have the same maximum count, it chooses the color with
    the highest value.
    """
    rows, cols = input_grid.get_dimensions()
    color_counts = {color: 0 for color in range(10)}

    # Count complete horizontal lines
    for row in range(rows):
        if len(set(input_grid.values[row])) == 1:
            color_counts[input_grid.values[row][0]] += 1

    # Count complete vertical lines
    for col in range(cols):
        if len(set(input_grid.values[r][col] for r in range(rows))) == 1:
            color_counts[input_grid.values[0][col]] += 1

    # Find color(s) with maximum count
    max_count = max(color_counts.values())
    max_colors = [color for color, count in color_counts.items() if count == max_count]

    # Choose color with highest value
    chosen_color = max(max_colors)

    # Create and return result
    return ColoredGrid(values=[[chosen_color]])
