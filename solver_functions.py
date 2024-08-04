from typing import List, Optional

from substrate import sb

from arc_vec import ResearchEvent, SuccessfulSolve
from colored_grid import ColoredGrid
from grid_problem import GridProblem

latest_distillation = {
    "current_total_knowledge": "The tasks involve transforming 2D colored grids represented as lists of lists, with colors encoded as numbers 0-9. Grids range from 1x1 to 30x30 in size. Transformations require pattern recognition, spatial reasoning, and rule application. Key aspects include: 1. Identifying color patterns, shapes, and structures within grids 2. Applying color-based transformations and mappings 3. Performing spatial operations like expanding, shifting, or compressing patterns 4. Implementing rule-based modifications based on grid structure and element relationships 5. Handling grid boundaries and edge cases 6. Executing conditional and multi-step transformations 7. Recognizing and utilizing anchor points or special colors 8. Filling shapes and propagating patterns directionally or symmetrically 9. Extracting sub-patterns and resizing grids 10. Creating frames, borders, and symmetrical arrangements Challenges often combine multiple concepts and require flexible thinking to identify and apply transformation rules to different grid sections or sub-patterns.",
    "ordered_concept_list": [
        "Pattern Recognition",
        "Color Mapping and Replacement",
        "Spatial Reasoning",
        "Rule Identification and Application",
        "Grid Structure Analysis",
        "Conditional Transformations",
        "Multi-step Operations",
        "Shape Identification and Filling",
        "Boundary and Anchor Detection",
        "Pattern Propagation",
        "Directional Transformations",
        "Symmetry and Mirroring",
        "Sub-grid Identification and Transformation",
        "Grid Resizing and Extraction",
        "Complex Shape Recognition",
        "Color-based Conditional Logic",
        "Element Preservation",
        "Frame and Border Creation",
        "Diagonal Pattern Handling",
        "Modular Transformations",
    ],
    "new_knowledge": "Recent challenges emphasize: 1. Identifying and filling hollow shapes, often based on 'seed' colors 2. Directional and symmetrical pattern propagation 3. Extracting and isolating specific patterns, resulting in grid compression 4. Complex multi-step transformations combining shape identification, filling, and propagation 5. Position-dependent transformation rules 6. Creating diagonal patterns and alternating color sequences 7. Grid expansion through pattern replication 8. Preserving certain elements while transforming others These observations highlight the need for a flexible approach capable of handling complex shapes, multiple transformation rules, and both expansion and compression of grid elements within a single challenge.",
}


def arc_intro(short=False) -> str:
    base = """
    This is a test of abstract and reasoning thinking skills. It presents puzzles where you see a few 
    examples and must figure out the hidden rule. Then you apply this rule to new situations. It tests how well 
    one can spot patterns, think abstractly, and solve problems creatively. It uses many different types of 
    puzzles, so you need to be flexible in your thinking. The main challenge is understanding complex ideas 
    from very little information, then using what you've learned in new ways.
    """
    if short:
        return base
    bullet_points = "\n- ".join(latest_distillation["ordered_concept_list"])
    return f"""
    {base}
    
    {latest_distillation["current_total_knowledge"]}
    
    The key concepts to keep in mind are:
    {bullet_points}
    """


def grid_setup() -> str:
    return f"""The grids are represented as a list of lists where each inner list represents a row and each element in the row
    represents a color.
    Number values correspond to colored squares: {ColoredGrid.color_mapping_str}
    
    All grids are rectangular and can be between 1x1 and 30x30 in size.
    """


def get_initial_impression(challenge: GridProblem) -> str:
    return (
        arc_intro()
        + f"""Below is a challenge from the test. You are tasked to solve it.
    The challenge involves transforming one colored 2D grid into another.
    You will be given a few examples of the transform in the form of input output pairs.
    You will also be given a test case that you need to solve.
    The goal is to find the function that fits all the examples, then apply it to the test case to compute a solution. 
    If you consider code here, only consider python pseudocode.
    
    {grid_setup()}
    
    Here is your current challenge:

    {challenge.to_task_description()}
    
    """
    )


def attempt_challenge(challenge: GridProblem, reasoning: str) -> str:
    return (
        arc_intro()
        + f"""Below is a challenge from the test. You are tasked to solve it.

    This is the current challenge:
    {challenge.to_task_description()}
    
    This is your latest reasoning:
    {reasoning}
    
    The ColoredGrid class has many methods that can help you analyze and manipulate the grids:
    {get_grid_class_overview()}
    
    Give it your best effort to consider all of that and solve the challenge in this JSON format:
    
    {SuccessfulSolve.model_json_schema()} 
    """
    )


def explain_research():
    return """

class ResearchEvent(BaseModel):
    #Research phase happens at the beginning and periodically later. Its job is to gather information about the problem
    #space in general. In this phase we can ask questions, gather ideas, think about how to better understand and approach.

    current_total_knowledge: str
    # Organized, detailed summary of all the knowledge about the dataset that that we know so far. This is a living document that grows and refactors as we learn more and see more challenges. It is a place to make sure we don't forget anything important and distill the most important and useful information
    ordered_concept_list: list[str]
    # Running list of ideas, concepts, and knowledge that is necessary to solve all the challenges in the dataset. This list is ordered by importance and relevance. It changes over time as we see and solve more
    new_knowledge: str
    # New knowledge gained from the current research event. This will be incorporated into the current total knowledge, so should be a summary of the most important new things learned
    
    """


def gather_research(challenges: List[GridProblem], previous_research: Optional[str] = None) -> str:
    challenge_list = "\n- ".join([c.to_task_description() for c in challenges])

    base = (
        arc_intro(short=True)
        + f"""Below is a collection of {len(challenges)} abstract reasoning challenges.
    Each challenge involves transforming one colored 2D grid into another.
    Each challenge contains a few examples of the transform demonstrated by input output pairs.
    Each challenge also contains a test case that is unsolved.
    The goal is to be able to identify the correct transformation and be able to implement it as a python function.

    {grid_setup()}
    
    Your goal right now is to gather information about the problem space in general. You are in the research phase.
    Take note of things that you think are important, keep questions in your mind if something is unclear,
    and try to distill what you are taking in. 
    
    <CHALLENGES>
    - {challenge_list}
    </CHALLENGES>
    """
    )
    if not previous_research:
        return base

    base += f"""
        These are the notes from your previous research. Update them with new information.
        Based on the new information refactor or refine your understanding of the problem space.
        The goal is to maintain an organized, comprehensive, detailed summary of all the knowledge
        we gain as we see and solve more challenges. 
        """
    return sb.concat(
        base, previous_research, "Come up with new current_total_knowledge, ordered_concept_list, and new_knowledge."
    )


def distill_research(event: ResearchEvent) -> str:
    return f"""You have been tasked with distilling this research pass.
    The goal is to take the information gathered and summarize it in a way that is concise and informative.
    This summary should be a living document that grows and refactors as we learn more and see more challenges.
    It should be a place to make sure we don't forget anything important and distill the most important and useful information.
    
    The current total knowledge is:
    {event.current_total_knowledge}
    
    The ordered concept list is:
    {event.ordered_concept_list}
    
    The new knowledge gained is:
    {event.new_knowledge}
    """


def extract_result(challenge: GridProblem) -> str:
    return f"""
    You were given this challenge:
    {challenge.to_task_description()}

    Your evaluation was:
    """


def get_grid_class_overview() -> str:
    return """
class ColoredGrid:
    # Represents a grid of colored cells (integers 0-9).

    # Class attributes and methods
    colors: dict[int, str]
    color_mapping_str: str

    @classmethod
    def value_to_color(cls, val: int) -> str: ...

    @classmethod
    def from_subgrids(cls, subgrids: List[List[ColoredGrid]]) -> ColoredGrid: ...

    @classmethod
    def from_rle(cls, rle: List[Tuple[int, int]], width: int) -> ColoredGrid: ...

    @classmethod
    def from_base64(cls, base64_str: str) -> ColoredGrid: ...

    # Instance methods
    def get_dimensions(self) -> Tuple[int, int]: ...

    def get_cell(self, row: int, col: int) -> int: ...

    def set_cell(self, row: int, col: int, value: int) -> None: ...

    def deep_copy(self) -> ColoredGrid: ...

    # Grid transformations
    def rotate_90(self, clockwise: bool = True) -> ColoredGrid: ...

    def flip_horizontal(self) -> ColoredGrid: ...

    def flip_vertical(self) -> ColoredGrid: ...

    def crop(self, top: int, left: int, bottom: int, right: int) -> ColoredGrid: ...

    def expand(self, top: int, right: int, bottom: int, left: int, fill_color: int = 0) -> ColoredGrid: ...

    def replace_color(self, old_color: int, new_color: int) -> ColoredGrid: ...

    def apply_mask(self, mask: ColoredGrid, replace_color: int) -> ColoredGrid: ...

    def scale(self, factor: int) -> ColoredGrid: ...

    # Grid analysis
    def count_color(self, color: int) -> int: ...

    def get_unique_colors(self) -> Set[int]: ...

    def get_color_frequencies(self) -> Dict[int, int]: ...

    def get_bounding_box(self, color: int) -> Optional[Tuple[int, int, int, int]]: ...

    def get_symmetry_axes(self) -> Tuple[bool, bool]: ...

    def find_pattern(self, pattern: ColoredGrid) -> List[Tuple[int, int]]: ...

    def detect_rectangles(self) -> List[Tuple[int, int, int, int, int]]: ...

    def detect_lines(self) -> List[Tuple[int, List[Tuple[int, int]]]]: ...

    def find_connected_regions(self, color: int) -> List[List[Tuple[int, int]]]: ...

    def find_largest_object(self, color: int) -> List[Tuple[int, int]]: ...

    def diff(self, other: ColoredGrid) -> ColoredGrid: ...

    def similarity_score(self, other: ColoredGrid) -> float: ...

    def find_repeating_pattern(self) -> Optional[ColoredGrid]: ...

    def find_center_of_mass(self) -> Tuple[float, float]: ...

    def find_color_transitions(self) -> List[Tuple[int, int, int, int]]: ...

    # Grid manipulation
    def extract_subgrid(self, top: int, left: int, height: int, width: int) -> ColoredGrid: ...

    def tile_grid(self, tiles: int) -> ColoredGrid: ...

    def add(self, other: ColoredGrid, modulo: int = 10) -> ColoredGrid: ...

    def subtract(self, other: ColoredGrid, modulo: int = 10) -> ColoredGrid: ...

    def multiply(self, scalar: int, modulo: int = 10) -> ColoredGrid: ...

    def split_into_subgrids(self, rows: int, cols: int) -> List[List[ColoredGrid]]: ...

    def to_binary(self, threshold: int) -> ColoredGrid: ...

    def compress_rle(self) -> List[Tuple[int, int]]: ...

    def get_edge_cells(self) -> List[Tuple[int, int]]: ...

    def flood_fill(self, row: int, col: int, new_color: int) -> ColoredGrid: ...

    def extrapolate_sequence(self, direction: Literal[
        "right", "left", "up", "down", "up-right", "up-left", "down-right", "down-left"],
                             steps: int = 1) -> ColoredGrid: ...

    def to_base64(self) -> str: ...

    def apply_cellular_automaton(self, rule: Callable[[List[int]], int]) -> ColoredGrid: ...

    def apply_function_to_regions(self, func: Callable[[List[Tuple[int, int]]], int]) -> ColoredGrid: ...
    """
