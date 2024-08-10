from substrate import sb, Box, ComputeText

aggregate = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, well-considered, and adheres to the highest standards of accuracy and reliability. Do not respond as if we're having a conversation, just output the revised response."""

jq_list = 'to_entries | map(((.key + 1) | tostring) + ". " + .value) | join("\n")'

default_models = ["Llama3Instruct70B", "claude-3-5-sonnet-20240620", "gpt-4o"]
default_decider = "claude-3-5-sonnet-20240620"


def_max_tokens = 1800


def get_mixture(q, prev=None, models=None, max_tokens=def_max_tokens):
    if models is None:
        models = default_models
    prompt = sb.concat(aggregate, "\n\nquestion: ", q, "\n\nprevious:\n\n", prev) if prev else q
    return Box(value=[ComputeText(prompt=prompt, model=m, max_tokens=max_tokens).future.text for m in models])


def moa(question: str, num_layers=2, max_tokens: int = def_max_tokens, opts=None, models=None, decider=default_decider):
    if models is None:
        models = default_models
    if opts is None:
        opts = {}
    layers = [get_mixture(question)]

    def last_layer():
        return sb.jq(layers[-1].future.value, jq_list)

    for _ in range(num_layers - 1):
        layers.append(get_mixture(question, prev=last_layer(), models=models, max_tokens=max_tokens))

    final = ComputeText(prompt=sb.concat(aggregate, "\n\n", last_layer()), model=decider, max_tokens=max_tokens, **opts)
    box = Box(value={"layers": [l.future.value for l in layers], "text": final.future.text})
    return box
