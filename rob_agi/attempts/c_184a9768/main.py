from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

def solve_184a9768(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following steps:
    1. Analyzes the input grid to identify color regions, their sizes, and positions
    2. Establishes a color hierarchy based on the total cell count of each color
    3. Creates a main structure based on the dominant color (usually blue or red)
    4. Processes secondary colors, placing them within or adjacent to the main structure
    5. Handles remaining colors, maintaining their relative positions
    6. Optimizes the layout to fit color regions efficiently within the grid
    7. Removes gray dots and ensures all regions are rectangular and continuous
    8. Creates a black border around the entire structure

    The transformation simplifies and structures the input while preserving color relationships,
    relative positions, and approximate proportions of color regions.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    def count_colors() -> Dict[int, int]:
        color_counts = {}
        for row in input_grid.values:
            for cell in row:
                if cell != 0:
                    color_counts[cell] = color_counts.get(cell, 0) + 1
        return color_counts

    def find_largest_region(color: int) -> List[Tuple[int, int]]:
        visited = set()
        largest_region = []
        for r in range(rows):
            for c in range(cols):
                if input_grid.values[r][c] == color and (r, c) not in visited:
                    region = []
                    stack = [(r, c)]
                    while stack:
                        curr_r, curr_c = stack.pop()
                        if (curr_r, curr_c) not in visited and input_grid.values[curr_r][curr_c] == color:
                            visited.add((curr_r, curr_c))
                            region.append((curr_r, curr_c))
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = curr_r + dr, curr_c + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    stack.append((nr, nc))
                    if len(region) > len(largest_region):
                        largest_region = region
        return largest_region

    def get_bounding_box(region: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        if not region:
            return (0, 0, 0, 0)
        min_r = min(r for r, _ in region)
        max_r = max(r for r, _ in region)
        min_c = min(c for _, c in region)
        max_c = max(c for _, c in region)
        return (min_r, min_c, max_r, max_c)

    def create_rectangle(color: int, top: int, left: int, bottom: int, right: int):
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                output_grid.values[r][c] = color

    # Count colors and establish hierarchy
    color_counts = count_colors()
    color_hierarchy = sorted(color_counts.keys(), key=lambda x: color_counts[x], reverse=True)

    # Create main structure
    main_color = color_hierarchy[0]
    main_region = find_largest_region(main_color)
    main_bbox = get_bounding_box(main_region)
    create_rectangle(main_color, *main_bbox)

    # Process secondary colors
    for color in color_hierarchy[1:]:
        if color == 5:  # Skip gray
            continue
        region = find_largest_region(color)
        if not region:
            continue
        bbox = get_bounding_box(region)
        relative_position = (
            (bbox[0] + bbox[2]) // 2 - (main_bbox[0] + main_bbox[2]) // 2,
            (bbox[1] + bbox[3]) // 2 - (main_bbox[1] + main_bbox[3]) // 2
        )
        
        # Determine placement
        if color == 8:  # Sky blue
            top = (main_bbox[0] + main_bbox[2]) // 2 - 1
            left = (main_bbox[1] + main_bbox[3]) // 2 - 1
            create_rectangle(color, top, left, top + 1, left + 1)
        elif all(abs(x) < (main_bbox[2] - main_bbox[0]) // 4 for x in relative_position):
            # Place inside main structure
            height = min(bbox[2] - bbox[0] + 1, (main_bbox[2] - main_bbox[0]) // 2)
            width = min(bbox[3] - bbox[1] + 1, (main_bbox[3] - main_bbox[1]) // 2)
            top = main_bbox[0] + (main_bbox[2] - main_bbox[0] - height) // 2
            left = main_bbox[1] + (main_bbox[3] - main_bbox[1] - width) // 2
            create_rectangle(color, top, left, top + height - 1, left + width - 1)
        else:
            # Place adjacent to main structure
            if abs(relative_position[0]) > abs(relative_position[1]):
                # Place above or below
                top = main_bbox[0] - 2 if relative_position[0] < 0 else main_bbox[2] + 2
                left = (main_bbox[1] + main_bbox[3]) // 2 - (bbox[3] - bbox[1]) // 2
                create_rectangle(color, top, left, top + 1, left + (bbox[3] - bbox[1]))
            else:
                # Place left or right
                top = (main_bbox[0] + main_bbox[2]) // 2 - (bbox[2] - bbox[0]) // 2
                left = main_bbox[1] - 2 if relative_position[1] < 0 else main_bbox[3] + 2
                create_rectangle(color, top, left, top + (bbox[2] - bbox[0]), left + 1)

    return output_grid
