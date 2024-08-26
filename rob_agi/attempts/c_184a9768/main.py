from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple, Dict
from collections import deque

class Region:
    def __init__(self, color: int, cells: List[Tuple[int, int]]):
        self.color = color
        self.cells = cells
        self.bbox = self.calculate_bbox()
        self.size = len(cells)

    def calculate_bbox(self) -> Tuple[int, int, int, int]:
        if not self.cells:
            return (0, 0, 0, 0)
        min_r = min(r for r, _ in self.cells)
        max_r = max(r for r, _ in self.cells)
        min_c = min(c for _, c in self.cells)
        max_c = max(c for _, c in self.cells)
        return (min_r, min_c, max_r, max_c)

def solve_184a9768(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by simplifying and structuring color regions:
    1. Analyzes the input grid to identify color regions and their properties
    2. Establishes a color hierarchy based on total area
    3. Creates a simplified main structure based on the dominant color
    4. Processes secondary colors, placing them relative to the main structure
    5. Optimizes the layout to minimize empty space
    6. Removes gray dots and ensures all regions are rectangular
    7. Adds a black border around the entire structure

    The transformation maintains color relationships, relative positions,
    and approximate proportions while creating a more structured output.
    """
    rows, cols = input_grid.get_dimensions()
    output_grid = ColoredGrid(values=[[0 for _ in range(cols)] for _ in range(rows)])

    def analyze_grid() -> Dict[int, List[Region]]:
        color_regions = {}
        visited = set()

        for r in range(rows):
            for c in range(cols):
                if (r, c) not in visited and input_grid.values[r][c] not in [0, 5]:
                    color = input_grid.values[r][c]
                    region = []
                    queue = deque([(r, c)])

                    while queue:
                        curr_r, curr_c = queue.popleft()
                        if (curr_r, curr_c) not in visited and input_grid.values[curr_r][curr_c] == color:
                            visited.add((curr_r, curr_c))
                            region.append((curr_r, curr_c))
                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = curr_r + dr, curr_c + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    queue.append((nr, nc))

                    if color not in color_regions:
                        color_regions[color] = []
                    color_regions[color].append(Region(color, region))

        return color_regions

    def simplify_region(region: Region) -> Tuple[int, int, int, int]:
        top, left, bottom, right = region.bbox
        return (top, left, bottom, right)

    def create_rectangle(color: int, top: int, left: int, bottom: int, right: int):
        for r in range(top, bottom + 1):
            for c in range(left, right + 1):
                output_grid.values[r][c] = color

    color_regions = analyze_grid()
    color_hierarchy = sorted(color_regions.keys(), key=lambda x: sum(r.size for r in color_regions[x]), reverse=True)

    main_color = color_hierarchy[0]
    main_region = max(color_regions[main_color], key=lambda r: r.size)
    main_bbox = simplify_region(main_region)
    create_rectangle(main_color, *main_bbox)

    for color in color_hierarchy[1:]:
        for region in sorted(color_regions[color], key=lambda r: r.size, reverse=True):
            bbox = simplify_region(region)
            relative_position = (
                (bbox[0] + bbox[2]) // 2 - (main_bbox[0] + main_bbox[2]) // 2,
                (bbox[1] + bbox[3]) // 2 - (main_bbox[1] + main_bbox[3]) // 2
            )

            if color == 8:  # Sky blue
                top = (main_bbox[0] + main_bbox[2]) // 2 - 1
                left = (main_bbox[1] + main_bbox[3]) // 2 - 1
                create_rectangle(color, top, left, top + 1, left + 1)
            elif all(abs(x) < (main_bbox[2] - main_bbox[0]) // 3 for x in relative_position):
                # Place inside main structure
                height = min(bbox[2] - bbox[0] + 1, (main_bbox[2] - main_bbox[0]) // 3)
                width = min(bbox[3] - bbox[1] + 1, (main_bbox[3] - main_bbox[1]) // 3)
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

    # Optimize space usage
    non_zero_rows = [r for r in range(rows) if any(output_grid.values[r][c] != 0 for c in range(cols))]
    non_zero_cols = [c for c in range(cols) if any(output_grid.values[r][c] != 0 for r in range(rows))]
    
    if non_zero_rows and non_zero_cols:
        min_row, max_row = min(non_zero_rows), max(non_zero_rows)
        min_col, max_col = min(non_zero_cols), max(non_zero_cols)
        
        optimized_grid = [[0 for _ in range(max_col - min_col + 3)] for _ in range(max_row - min_row + 3)]
        
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                optimized_grid[r - min_row + 1][c - min_col + 1] = output_grid.values[r][c]
        
        output_grid = ColoredGrid(values=optimized_grid)

    return output_grid
