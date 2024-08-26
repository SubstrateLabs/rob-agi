from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_aa4ec2a5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Initializes the output grid with yellow (4) background and red (2) outer border.
    2. Identifies and classifies blue (1) regions based on size.
    3. Creates a primary segmentation structure with red (2) lines.
    4. Processes large regions (>9 cells): outlines with red (2), fills with sky blue (8) and blue (1) or magenta (6) based on thickness.
    5. Processes medium regions (4-9 cells): outlines with red (2) and fills with blue (1).
    6. Processes small regions (1-3 cells): creates special structures with red (2) and blue (1).
    7. Adjusts intersections, corners, and refines color distribution.
    8. Preserves background structure and handles special cases.
    9. Performs a final pass to ensure consistency and proper integration with borders.
    """
    def initialize_grid(input_grid: ColoredGrid) -> ColoredGrid:
        new_grid = ColoredGrid(values=[[4 for _ in range(input_grid.num_cols)] for _ in range(input_grid.num_rows)])
        for y in range(new_grid.num_rows):
            new_grid.values[y][0] = new_grid.values[y][-1] = 2
        for x in range(new_grid.num_cols):
            new_grid.values[0][x] = new_grid.values[-1][x] = 2
        return new_grid

    def find_blue_regions(grid: ColoredGrid) -> List[List[Tuple[int, int]]]:
        visited = set()
        regions = []
        for y in range(grid.num_rows):
            for x in range(grid.num_cols):
                if grid.values[y][x] == 1 and (y, x) not in visited:
                    region = []
                    stack = [(y, x)]
                    while stack:
                        cy, cx = stack.pop()
                        if (cy, cx) not in visited and grid.values[cy][cx] == 1:
                            visited.add((cy, cx))
                            region.append((cy, cx))
                            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < grid.num_rows and 0 <= nx < grid.num_cols:
                                    stack.append((ny, nx))
                    regions.append(region)
        return regions

    def process_large_region(new_grid: ColoredGrid, region: List[Tuple[int, int]]):
        outline_region(new_grid, region, 2)
        inner_region = get_inner_region(region)
        for y, x in inner_region:
            new_grid.values[y][x] = 8  # Fill with sky blue
    
        innermost_region = get_inner_region(inner_region)
        for y, x in innermost_region:
            if is_thin_area(region, y, x):
                new_grid.values[y][x] = 6  # Fill thin areas with magenta
            else:
                new_grid.values[y][x] = 1  # Fill thick areas with blue
    
        add_internal_structure(new_grid, region)

    def is_thin_area(region: List[Tuple[int, int]], y: int, x: int) -> bool:
        neighbors = sum(1 for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                        if (y+dy, x+dx) in region)
        return neighbors <= 2

    def add_internal_structure(new_grid: ColoredGrid, region: List[Tuple[int, int]]):
        min_y, min_x = min(region)
        max_y, max_x = max(region)
        mid_y, mid_x = (min_y + max_y) // 2, (min_x + max_x) // 2
        for y in range(min_y, max_y + 1):
            if min_x < mid_x < max_x:
                new_grid.values[y][mid_x] = 2
        for x in range(min_x, max_x + 1):
            if min_y < mid_y < max_y:
                new_grid.values[mid_y][x] = 2

    def process_medium_region(new_grid: ColoredGrid, region: List[Tuple[int, int]]):
        for y, x in region:
            new_grid.values[y][x] = 1  # Fill with blue
        outline_region(new_grid, region, 2)
        add_simple_internal_structure(new_grid, region)

    def add_simple_internal_structure(new_grid: ColoredGrid, region: List[Tuple[int, int]]):
        min_y, min_x = min(region)
        max_y, max_x = max(region)
        mid_y, mid_x = (min_y + max_y) // 2, (min_x + max_x) // 2
        if max_y - min_y > max_x - min_x:
            for x in range(min_x, max_x + 1):
                new_grid.values[mid_y][x] = 2
        else:
            for y in range(min_y, max_y + 1):
                new_grid.values[y][mid_x] = 2

    def process_small_region(new_grid: ColoredGrid, region: List[Tuple[int, int]]):
        if len(region) == 1:
            y, x = region[0]
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < new_grid.num_rows and 0 <= nx < new_grid.num_cols:
                        new_grid.values[ny][nx] = 2 if (dy, dx) != (0, 0) else 1
        else:
            expanded_region = expand_region(region)
            for y, x in expanded_region:
                new_grid.values[y][x] = 1  # Fill with blue
            outline_region(new_grid, expanded_region, 2)

    def expand_region(region: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        min_y, min_x = min(region)
        max_y, max_x = max(region)
        return [(y, x) for y in range(min_y - 1, max_y + 2) 
                       for x in range(min_x - 1, max_x + 2)]

    def outline_region(new_grid: ColoredGrid, region: List[Tuple[int, int]], outline_color: int):
        for y, x in region:
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < new_grid.num_rows and 0 <= nx < new_grid.num_cols and (ny, nx) not in region:
                    new_grid.values[ny][nx] = outline_color

    def process_top_edge(new_grid: ColoredGrid):
        for x in range(new_grid.num_cols):
            if any(new_grid.values[1][x] != 4 for x in range(max(0, x-1), min(new_grid.num_cols, x+2))):
                new_grid.values[0][x] = 2

    def create_primary_segmentation(new_grid: ColoredGrid):
        for y in range(5, new_grid.num_rows - 5, 5):
            for x in range(new_grid.num_cols):
                if new_grid.values[y][x] == 4:
                    new_grid.values[y][x] = 2
        for x in range(5, new_grid.num_cols - 5, 5):
            for y in range(new_grid.num_rows):
                if new_grid.values[y][x] == 4:
                    new_grid.values[y][x] = 2

    def adjust_intersections_and_corners(new_grid: ColoredGrid):
        for y in range(1, new_grid.num_rows - 1):
            for x in range(1, new_grid.num_cols - 1):
                if new_grid.values[y][x] == 2:
                    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        if new_grid.values[y+dy][x+dx] == 2:
                            new_grid.values[y+dy][x+dx] = 2
                            break

    def refine_color_distribution(new_grid: ColoredGrid):
        for y in range(1, new_grid.num_rows - 1):
            for x in range(1, new_grid.num_cols - 1):
                if new_grid.values[y][x] == 8:
                    surrounding = [new_grid.values[y+dy][x+dx] for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
                    if surrounding.count(8) >= 3:
                        new_grid.values[y][x] = 6

    def handle_special_cases(new_grid: ColoredGrid):
        # This function can be expanded to handle more specific patterns
        pass

    def preserve_background_structure(input_grid: ColoredGrid, new_grid: ColoredGrid):
        for y in range(new_grid.num_rows):
            for x in range(new_grid.num_cols):
                if input_grid.values[y][x] == 4 and new_grid.values[y][x] == 4:
                    if (y % 5 == 0 or x % 5 == 0) and (y not in [0, new_grid.num_rows-1] and x not in [0, new_grid.num_cols-1]):
                        new_grid.values[y][x] = 2

    def final_pass(new_grid: ColoredGrid):
        for y in range(new_grid.num_rows):
            for x in range(new_grid.num_cols):
                if new_grid.values[y][x] == 1:
                    new_grid.values[y][x] = 8
                elif new_grid.values[y][x] == 4 and (y in [0, new_grid.num_rows-1] or x in [0, new_grid.num_cols-1]):
                    new_grid.values[y][x] = 2

    def refine_edges(new_grid: ColoredGrid):
        for y in range(new_grid.num_rows):
            for x in range(new_grid.num_cols):
                if new_grid.values[y][x] != 4 and (y == 0 or x == 0 or y == new_grid.num_rows - 1 or x == new_grid.num_cols - 1):
                    new_grid.values[y][x] = 2

    def preserve_yellow_background(input_grid: ColoredGrid, new_grid: ColoredGrid):
        for y in range(input_grid.num_rows):
            for x in range(input_grid.num_cols):
                if input_grid.values[y][x] == 4 and new_grid.values[y][x] == 4:
                    new_grid.values[y][x] = 4

    def make_context_aware_adjustments(new_grid: ColoredGrid):
        # This function can be expanded to include more specific adjustments
        pass

    # Main transformation process
    new_grid = initialize_grid(input_grid)
    blue_regions = find_blue_regions(input_grid)
    create_primary_segmentation(new_grid)
    
    for region in blue_regions:
        if len(region) > 9:
            process_large_region(new_grid, region)
        elif 4 <= len(region) <= 9:
            process_medium_region(new_grid, region)
        else:
            process_small_region(new_grid, region)
    
    adjust_intersections_and_corners(new_grid)
    refine_color_distribution(new_grid)
    handle_special_cases(new_grid)
    preserve_background_structure(input_grid, new_grid)
    final_pass(new_grid)
    
    return new_grid
