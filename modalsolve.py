import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .workdir("/app")
    .apt_install("git")
    .pip_install("pytest", "pydantic==2.8.2", "substrate", "numpy", "aider==0.54.0")
)
app = modal.App("arc_solver", image=image)


@app.function(
    mounts=[
        modal.Mount.from_local_dir("data", remote_path="/app/data"),
        modal.Mount.from_local_dir("rob_agi", remote_path="/app/rob_agi"),
    ]
)
def foo():
    return "hello"


@app.local_entrypoint()
def localfn():
    h = foo.remote()
    print("res", h)
