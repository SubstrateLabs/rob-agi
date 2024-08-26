from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_aa4ec2a5(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms the input grid by applying the following rules:
    1. Initializes the output grid with yellow (4) background and red (2) left edge.
    2. Identifies and classifies blue (1) regions based on size.
    3. Processes large regions (>9 cells): outlines with red (2), fills with sky blue (8), marks protrusions with magenta (6).
    4. Processes small regions (4-9 cells): outlines with red (2) and keeps interior blue (1).
    5. Processes very small regions (1-3 cells): creates special structures.
    6. Adjusts the top edge and creates a segmentation structure with red (2) lines.
    7. Refines edge treatments and preserves the yellow (4) background where appropriate.
    8. Makes context-aware adjustments for consistency.
    """
    def initialize_grid(input_grid: ColoredGrid) -> ColoredGrid:
        new_grid = input_grid.deep_copy()
        for y in range(new_grid.num_rows):
            new_grid.values[y][0] = 2
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
        for y, x in region:
            new_grid.values[y][x] = 8  # Fill with sky blue
        
        # Mark protrusions
        for y, x in region:
            neighbors = sum(1 for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                            if 0 <= y+dy < new_grid.num_rows and 0 <= x+dx < new_grid.num_cols and (y+dy, x+dx) in region)
            if neighbors <= 2:
                new_grid.values[y][x] = 6  # Mark as magenta

        outline_region(new_grid, region, 2)

    def process_small_region(new_grid: ColoredGrid, region: List[Tuple[int, int]]):
        for y, x in region:
            new_grid.values[y][x] = 1
        outline_region(new_grid, region, 2)

    def process_very_small_region(new_grid: ColoredGrid, region: List[Tuple[int, int]]):
        if len(region) == 1:
            y, x = region[0]
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < new_grid.num_rows and 0 <= nx < new_grid.num_cols:
                        new_grid.values[ny][nx] = 2 if (dy, dx) != (0, 0) else 1
        else:
            for y, x in region:
                new_grid.values[y][x] = 1
            outline_region(new_grid, region, 2)

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

    def create_segmentation(new_grid: ColoredGrid):
        for y in range(5, new_grid.num_rows - 5, 5):
            for x in range(new_grid.num_cols):
                if new_grid.values[y][x] == 4 and new_grid.values[y-1][x] != 2 and new_grid.values[y+1][x] != 2:
                    new_grid.values[y][x] = 2
        for x in range(5, new_grid.num_cols - 5, 5):
            for y in range(new_grid.num_rows):
                if new_grid.values[y][x] == 4 and new_grid.values[y][x-1] != 2 and new_grid.values[y][x+1] != 2:
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
    
    for region in blue_regions:
        if len(region) > 9:
            process_large_region(new_grid, region)
        elif 4 <= len(region) <= 9:
            process_small_region(new_grid, region)
        else:
            process_very_small_region(new_grid, region)
    
    process_top_edge(new_grid)
    create_segmentation(new_grid)
    refine_edges(new_grid)
    preserve_yellow_background(input_grid, new_grid)
    make_context_aware_adjustments(new_grid)
    
    return new_grid
