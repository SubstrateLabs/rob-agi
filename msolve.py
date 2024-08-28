import time
from typing import Dict, List, Optional
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import logging

from modal import App, Image, Mount, Volume, web_endpoint, Sandbox, Secret

app = App("rob_arc")
log = logging.getLogger(__name__)
OUTPUT_FILENAME = "result.json"


class RunCodeArgs(BaseModel):
    base_image: Optional[str] = None
    """e.g. ubuntu:22.04, or huanjason/scikit-learn"""

    apt_install: Optional[List[str]]

    entrypoint: str
    """Use shell form, e.g. 'python -m mymodule', maybe support exec form later"""

    workdir: Optional[str] = "/app"
    img_build_commands: Optional[List[str]] = None

    timeout_seconds: int = 90

    secrets: Optional[Dict[str, str]] = None
    env_vars: Optional[Dict[str, str]] = None
    persisted_mounts: Optional[Dict[str, str]] = None
    file_mounts: Optional[Dict[str, str]] = None


class RunCodeOutput(BaseModel):
    json_out: Optional[str]
    stdout: str
    stderr: str


base_pip = []
local_path = Path(__file__).parent


@app.function(
    secrets=[],
    image=Image.debian_slim().pip_install(*base_pip),
    timeout=300,
    allow_concurrent_inputs=20,
    # mounts=[Mount.from_local_file(local_path / EXEC_FILENAME, remote_path=REMOTE_EXEC_PATH)],
)
@web_endpoint(method="POST")
async def api(args: RunCodeArgs):
    try:
        output = await run_code(args)
    except Exception as e:
        log.error(f"unexpected error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "unexpected error while running sandbox", "type": "api_error"},
        )
    res = output.dict()
    return JSONResponse(status_code=status.HTTP_200_OK, content=res)


def get_image(args: RunCodeArgs) -> Image:
    if args.base_image:
        img = Image.from_registry(args.base_image)
    else:
        img = Image.debian_slim()
    if args.apt_install:
        img.apt_install(*args.apt_install)
    if args.img_build_commands:
        img.run_commands(*args.img_build_commands)
    return img


async def run_code(args: RunCodeArgs) -> RunCodeOutput:
    """
    User code should write to /output/result.json if they want to return a JSON object
    """
    with Volume.ephemeral() as output_vol:
        local_dir = Path(TemporaryDirectory().name)
        write_files(args, local_dir)
        cmd = args.entrypoint
        t0 = time.time()
        extra_vols = {k: Volume.from_name(v, create_if_missing=True) for k, v in args.persisted_mounts.items()}

        img = get_image(args)
        env_vars = args.env_vars or {}
        if args.secrets:
            env_vars.update(args.secrets)

        sandbox = await Sandbox.create.aio(
            "bash",
            "-c",
            cmd,
            image=img,
            volumes={"/output": output_vol, **extra_vols},
            mounts=[Mount.from_local_dir(local_dir, remote_path=args.workdir)],
            workdir=args.workdir,
            secrets=[Secret.from_dict(env_vars)],
            timeout=args.timeout_seconds,
        )
        log_lines = 0
        stdout = ""
        async for line in sandbox.stdout:
            log_lines += 1
            stdout += line
        await sandbox.wait.aio()
        t1 = time.time()
        log.info(f"sandbox finished with code: {sandbox.returncode} in {t1 - t0:.2f} seconds")
        if log_lines > 0:
            log.info(f"  > generated {log_lines} log lines")
        stderr = ""
        output = None
        if sandbox.returncode == 0:
            has_output = False
            files = await output_vol.listdir.aio("/")
            for f in files:
                filename = Path(f.path).name
                if filename == OUTPUT_FILENAME:
                    has_output = True
                    break
            if has_output:
                json_str = ""
                async for data in output_vol.read_file.aio(OUTPUT_FILENAME):
                    json_str += data.decode("utf-8")
                output = json_str
        else:
            stderr = sandbox.stderr.read()
        result = RunCodeOutput(stdout=stdout, stderr=stderr, json_out=output)
        return result


def write_files(args: RunCodeArgs, local_dir: Path) -> Dict[str, Path]:
    local_dir.mkdir(parents=True, exist_ok=True)
    mapped_paths = {}
    for k, v in args.file_mounts.items():
        mapped_paths[k] = local_dir / v
        with open(local_dir / k, "w") as f:
            f.write(v)
    return mapped_paths


@app.local_entrypoint()
async def run_fn():
    python_print_foo = "print('foo')"
    python_write_json = "import json\nwith open('/output/result.json', 'w') as f: json.dump({'foo': 'bar'}, f)"
    args = RunCodeArgs(
        base_image="debian:bookworm-slim",
        apt_install=["git"],
        entrypoint="echo hello $MY_VAR && python myfn.py && python write_json.py",
        workdir="/app",
        timeout_seconds=60,
        secrets={},
        env_vars={"MY_VAR": "MY_VAR_VALUE"},
        persisted_mounts={},
        file_mounts={"myfn.py": python_print_foo, "write_json.py": python_write_json},
    )
    output = await run_code(args)
    print(output)


example_curl = """
curl -X 'POST' \
    'http://localhost:8000/api' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "base_image": "debian:bookworm-slim",
    "apt_install": ["git"],
    "entrypoint": "echo hello $MY_VAR && python myfn.py && python write_json.py",
    "workdir": "/app",
    "timeout_seconds": 60,
    "secrets": {},
    "env_vars": {"MY_VAR": "MY_VAR_VALUE"},
    "persisted_mounts": {},
    "file_mounts": {"myfn.py": "print(\u0027foo\u0027)", "write_json.py": "import json\nwith open(\u0027/output/result.json\u0027, \u0027w\u0027) as f: json.dump({\u0027foo\u0027: \u0027bar\u0027}, f)"}
    }'
"""
