from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_642d658d.main import (
    solve_642d658d,
    calculate_structural_score,
    calculate_pattern_score,
    calculate_interaction_score,
    calculate_distribution_score,
    calculate_position_score,
    calculate_multi_scale_score
)

def run_experiment():
    # Create a sample input grid
    sample_grid = ColoredGrid(values=[
        [8, 8, 8, 3, 8, 0, 8, 1],
        [8, 2, 8, 8, 8, 8, 8, 3],
        [8, 8, 8, 1, 3, 8, 8, 8],
        [0, 8, 8, 2, 3, 0, 8, 8],
        [8, 3, 8, 4, 2, 8, 8, 8],
        [0, 8, 8, 2, 8, 0, 8, 1],
        [8, 8, 8, 8, 8, 8, 8, 8],
        [0, 8, 8, 0, 8, 0, 8, 8]
    ])

    # Identify background color
    color_counts = {}
    for row in sample_grid.values:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1
    background_color = max(color_counts, key=color_counts.get)

    print(f"Background color: {background_color}")

    # Calculate scores for each non-background color
    non_background_colors = set(color_counts.keys()) - {background_color}
    for color in non_background_colors:
        structural_score = calculate_structural_score(sample_grid, color, background_color)
        pattern_score = calculate_pattern_score(sample_grid, color)
        interaction_score = calculate_interaction_score(sample_grid, color, background_color)
        distribution_score = calculate_distribution_score(sample_grid, color)
        position_score = calculate_position_score(sample_grid, color)
        multi_scale_score = calculate_multi_scale_score(sample_grid, color)

        print(f"\nColor: {color}")
        print(f"Structural score: {structural_score}")
        print(f"Pattern score: {pattern_score}")
        print(f"Interaction score: {interaction_score}")
        print(f"Distribution score: {distribution_score}")
        print(f"Position score: {position_score}")
        print(f"Multi-scale score: {multi_scale_score}")

    # Run the solve function and print the result
    result = solve_642d658d(sample_grid)
    print(f"\nSolve result: {result.values[0][0]}")

if __name__ == "__main__":
    run_experiment()
