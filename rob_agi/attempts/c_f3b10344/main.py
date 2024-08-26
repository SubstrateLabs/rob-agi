from rob_agi.colored_grid import ColoredGrid

def solve_f3b10344(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Solve the f3b10344 challenge by applying sky blue frames around shapes and connecting adjacent frames.
    
    The function identifies non-black shapes in the grid, surrounds them with sky blue (8) frames,
    and connects frames of adjacent shapes with the same color. The original shapes are preserved,
    and the background remains black (0) where no changes are applied.
    """
    # Check if the entire grid is black (all zeros)
    if all(cell == 0 for row in input_grid.values for cell in row):
        return input_grid

    # Create a deep copy of the input grid to work on
    grid = input_grid.deep_copy()

    # Find all non-black regions
    all_regions = []
    for color in range(1, 10):  # Exclude black (0)
        regions = grid.find_connected_regions(color)
        all_regions.extend((color, region) for region in regions)

    # Process each shape
    for color, region in all_regions:
        # Determine the bounding box
        min_row = min(r for r, _ in region)
        max_row = max(r for r, _ in region)
        min_col = min(c for _, c in region)
        max_col = max(c for _, c in region)

        # Apply sky blue frame
        for r in range(max(0, min_row - 1), min(grid.num_rows, max_row + 2)):
            for c in range(max(0, min_col - 1), min(grid.num_cols, max_col + 2)):
                if grid.values[r][c] == 0:
                    if (r == min_row - 1 or r == max_row + 1 or
                        c == min_col - 1 or c == max_col + 1):
                        grid.values[r][c] = 8

    # Connect adjacent frames
    for i, (color1, region1) in enumerate(all_regions):
        for color2, region2 in all_regions[i+1:]:
            if color1 == color2:
                min_row1, max_row1 = min(r for r, _ in region1), max(r for r, _ in region1)
                min_col1, max_col1 = min(c for _, c in region1), max(c for _, c in region1)
                min_row2, max_row2 = min(r for r, _ in region2), max(r for r, _ in region2)
                min_col2, max_col2 = min(c for _, c in region2), max(c for _, c in region2)

                # Check if regions are close horizontally
                if abs(min_col1 - max_col2) <= 3 or abs(min_col2 - max_col1) <= 3:
                    start_col = min(max_col1, max_col2) + 1
                    end_col = max(min_col1, min_col2)
                    for r in range(max(min_row1, min_row2), min(max_row1, max_row2) + 1):
                        for c in range(start_col, end_col):
                            if grid.values[r][c] == 0:
                                grid.values[r][c] = 8

                # Check if regions are close vertically
                if abs(min_row1 - max_row2) <= 3 or abs(min_row2 - max_row1) <= 3:
                    start_row = min(max_row1, max_row2) + 1
                    end_row = max(min_row1, min_row2)
                    for c in range(max(min_col1, min_col2), min(max_col1, max_col2) + 1):
                        for r in range(start_row, end_row):
                            if grid.values[r][c] == 0:
                                grid.values[r][c] = 8

    return grid
