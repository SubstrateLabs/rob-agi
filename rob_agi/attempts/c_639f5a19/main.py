from rob_agi.colored_grid import ColoredGrid

def solve_639f5a19(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms sky blue (color 8) regions in the input grid into a specific color pattern.
    
    The solution works as follows:
    1. Find all connected regions of sky blue (color 8) in the input grid.
    2. For each region:
       a. Determine the section size (6x6 or 4x4) based on the region's dimensions.
       b. Divide the region into sections and apply a color pattern to each section:
          - Top-left (2x2 or 3x3): Magenta (6)
          - Top-right (2x2 or 3x3): Blue (1)
          - Center (2x2): Yellow (4)
          - Bottom-left (2x2 or 3x3): Red (2)
          - Bottom-right (2x2 or 3x3): Green (3)
    3. The pattern adapts to regions of any size, maintaining the relative positions of colors.
    4. Non-sky blue areas in the grid are preserved.

    Args:
    input_grid (ColoredGrid): The input grid to be transformed.

    Returns:
    ColoredGrid: The transformed grid with sky blue regions replaced by the color pattern.
    """
    new_grid = input_grid.deep_copy()
    sky_blue_regions = input_grid.find_connected_regions(8)
    
    for region in sky_blue_regions:
        min_row = min(r for r, _ in region)
        min_col = min(c for _, c in region)
        max_row = max(r for r, _ in region)
        max_col = max(c for _, c in region)
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        
        section_size = 6 if width >= 6 and height >= 6 else 4
        
        for cell_row, cell_col in region:
            relative_row = cell_row - min_row
            relative_col = cell_col - min_col
            
            section_row = relative_row % section_size
            section_col = relative_col % section_size
            
            if section_row < section_size // 2 and section_col < section_size // 2:
                new_color = 6  # Magenta
            elif section_row < section_size // 2 and section_col >= section_size // 2:
                new_color = 1  # Blue
            elif section_row >= section_size // 2 and section_col < section_size // 2:
                new_color = 2  # Red
            elif section_row >= section_size // 2 and section_col >= section_size // 2:
                new_color = 3  # Green
            else:
                new_color = 4  # Yellow (center)
            
            if section_size == 6:
                if 2 <= section_row <= 3 and 2 <= section_col <= 3:
                    new_color = 4  # Yellow (center for 6x6)
            else:  # 4x4 section
                if 1 <= section_row <= 2 and 1 <= section_col <= 2:
                    new_color = 4  # Yellow (center for 4x4)
            
            new_grid.values[cell_row][cell_col] = new_color
    
    return new_grid
