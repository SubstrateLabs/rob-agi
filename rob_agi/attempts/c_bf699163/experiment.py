from rob_agi.colored_grid import ColoredGrid
from rob_agi.attempts.c_bf699163.main import solve_bf699163, is_valid_pattern
import matplotlib.pyplot as plt
import numpy as np

def calculate_score(color: int, centrality: float, alpha: float = 0.5) -> float:
    """Calculate a combined score based on color and centrality."""
    return alpha * (9 - color) + (1 - alpha) * (1 / (1 + centrality))

def visualize_grid_with_patterns(grid: ColoredGrid, patterns: list, title: str):
    """Visualize the grid with pattern locations and scores."""
    plt.figure(figsize=(12, 10))
    plt.imshow(grid.values, cmap='tab10', vmin=0, vmax=9)
    
    rows, cols = grid.get_dimensions()
    center_row, center_col = (rows - 1) / 2, (cols - 1) / 2

    for row, col, color, centrality in patterns:
        score = calculate_score(color, centrality)
        plt.gca().add_patch(plt.Rectangle((col-1.5, row-1.5), 3, 3, fill=False, edgecolor='white', linewidth=2))
        plt.text(col, row, f'{color},{score:.2f}', color='white', ha='center', va='center', fontweight='bold')

    plt.title(title)
    plt.colorbar(ticks=range(10), label='Color')
    plt.tight_layout()
    plt.show()

def run_experiment():
    # Example grids
    example_0 = ColoredGrid(values=[
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 8, 5, 8, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 5, 5],
        [5, 8, 8, 8, 5, 5, 5, 5, 5, 5, 5, 2, 5, 2, 5, 5],
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 5, 5],
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 5, 5, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 5, 5, 3, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 5, 5, 3, 3, 3, 5, 7, 7, 7, 7, 5, 5, 7, 7],
        [5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 5, 5, 5],
        [5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 1, 5, 1, 5, 5, 5],
        [5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 1, 1, 1, 5, 5, 5],
        [5, 6, 6, 6, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5],
        [5, 6, 5, 6, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5],
        [5, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5]
    ])

    example_1 = ColoredGrid(values=[
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 1, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 5, 5, 5, 5, 5, 3, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 5, 5, 5, 5, 2, 2, 2, 5],
        [5, 7, 7, 7, 5, 7, 7, 7, 5, 5, 5, 5, 5, 5, 2, 5, 2, 5],
        [5, 7, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 5, 2, 2, 2, 5],
        [5, 7, 5, 4, 4, 4, 5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 5, 5, 4, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 7, 5, 4, 4, 4, 5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [5, 7, 5, 5, 5, 5, 5, 7, 5, 5, 5, 8, 8, 8, 5, 5, 5, 5],
        [5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 8, 5, 8, 5, 5, 5, 5],
        [5, 7, 5, 5, 5, 5, 5, 7, 5, 5, 5, 8, 8, 8, 5, 5, 5, 5]
    ])

    for idx, grid in enumerate([example_0, example_1]):
        rows, cols = grid.get_dimensions()
        center_row, center_col = (rows - 1) / 2, (cols - 1) / 2

        patterns = []
        for row in range(1, rows - 1):
            for col in range(1, cols - 1):
                if is_valid_pattern(grid, row, col):
                    color = grid.values[row-1][col]
                    centrality = abs(row - center_row) + abs(col - center_col)
                    patterns.append((row, col, color, centrality))

        visualize_grid_with_patterns(grid, patterns, f"Example {idx}")

    plt.show()

if __name__ == "__main__":
    run_experiment()
