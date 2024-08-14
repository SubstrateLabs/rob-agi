import json
import os
import time

from substrate import sb, Box, ComputeText, Substrate

aggregate = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, well-considered, and adheres to the highest standards of accuracy and reliability. Do not respond as if we're having a conversation, just output the revised response."""

jq_list = 'to_entries | map(((.key + 1) | tostring) + ". " + .value) | join("\n")'

# default_models = ["Llama3Instruct70B", "claude-3-5-sonnet-20240620", "gpt-4o"]
default_models = ["claude-3-5-sonnet-20240620", "gpt-4o"]
# default_models = ["Llama3Instruct70B", "gpt-4o"]
default_decider = "claude-3-5-sonnet-20240620"
# default_decider = "gpt-4o"


def_max_tokens = 1800

api_key = os.environ.get("SUBSTRATE_API_KEY")
substrate = Substrate(api_key=api_key, timeout=60 * 2, additional_headers={})


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


current_dir = os.path.dirname(os.path.abspath(__file__))


async def run_moa(
    question: str,
    num_layers=2,
    max_tokens: int = def_max_tokens,
    opts=None,
    models=None,
    decider=default_decider,
    filename_prefix="",
):
    if models is None:
        models = default_models
    io = Box(value=question)
    box = moa(io.future.value, num_layers=num_layers, max_tokens=max_tokens, opts=opts, models=models, decider=decider)
    res = await substrate.async_run(box)
    json_out = res.get(box).value

    with open(os.path.join(current_dir, "moa-base.html"), "r") as f:
        html_template = f.read()

    html = (
        html_template.replace('"{{ individual }}"', json.dumps(json_out["layers"], indent=2))
        .replace('"{{ question }}"', json.dumps(res.get(io).value))
        .replace('"{{ model_names }}"', json.dumps(models))
        .replace('"{{ summaries }}"', f'[{json.dumps(json_out["text"])}]')
    )

    filename = "-".join([filename_prefix, str(time.time_ns())])
    with open(os.path.join(current_dir, "moa-out", f"{filename}.html"), "w") as f:
        f.write(html)

    return json_out
