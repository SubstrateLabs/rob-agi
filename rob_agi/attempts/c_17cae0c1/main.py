from rob_agi.colored_grid import ColoredGrid
from typing import List, Tuple

def solve_17cae0c1(input_grid: ColoredGrid) -> ColoredGrid:
    """
    Transforms a 3x9 input grid into a 3x9 output grid based on the distribution of gray (5) squares.
    
    The input grid is divided into three 3x3 sections. The section with the most gray squares
    (or rightmost in case of a tie) is assigned a color based on its position:
    - If leftmost, it becomes magenta (6)
    - If in the middle, it becomes green (3)
    - If rightmost, it becomes blue (1)
    
    For the remaining two sections:
    - If to the left of the highest rank, assign brown (9) to the higher rank and yellow (4) to the lower rank
    - If to the right of the highest rank, assign yellow (4) to the higher rank and brown (9) to the lower rank
    """
    # Step 1: Parse the input grid
    sections = [
        input_grid.extract_subgrid(0, 0, 3, 3),
        input_grid.extract_subgrid(0, 3, 3, 3),
        input_grid.extract_subgrid(0, 6, 3, 3)
    ]

    # Step 2: Count gray squares
    gray_counts = [(i, sum(row.count(5) for row in section.values)) for i, section in enumerate(sections)]

    # Step 3: Rank the sections (rightmost wins ties)
    ranked_sections = sorted(gray_counts, key=lambda x: (x[1], x[0]), reverse=True)

    # Step 4: Assign colors
    colors = [0, 0, 0]
    highest_rank_index = ranked_sections[0][0]
    
    if highest_rank_index == 0:
        colors[0] = 6  # Magenta for leftmost
        colors[1] = 9  # Brown for middle
        colors[2] = 4  # Yellow for rightmost
    elif highest_rank_index == 1:
        colors[0] = 9  # Brown for left of highest
        colors[1] = 3  # Green for middle highest
        colors[2] = 4  # Yellow for right of highest
    else:  # highest_rank_index == 2
        colors[0] = 9  # Brown for leftmost
        colors[1] = 1  # Blue for middle
        colors[2] = 4  # Yellow for rightmost

    # Step 5: Create the output grid
    output_values = []
    for _ in range(3):
        row = []
        for color in colors:
            row.extend([color] * 3)
        output_values.append(row)

    # Step 6: Return the result
    return ColoredGrid(values=output_values)
