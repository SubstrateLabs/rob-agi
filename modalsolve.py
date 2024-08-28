import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "pytest", "git+https://github.com/SubstrateLabs/rob-agi.git@d85c3030", "pydantic==2.8.2", "substrate", "numpy"
    )
)
app = modal.App("arc_solver", image=image)


@app.function()
def foo():
    return "hello"


@app.local_entrypoint()
def localfn():
    h = foo.remote()
    print("res", h)
