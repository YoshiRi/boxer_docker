#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Boxer inside the repository's Docker Compose services."
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use the boxer-gpu service and GPU compose profile.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the selected Docker service before running Boxer.",
    )
    parser.add_argument(
        "--job",
        action="store_true",
        help="Run scripts/run_boxer_job.py instead of run_boxer.py.",
    )
    parser.add_argument(
        "--backend",
        choices=["legacy", "api"],
        default=None,
        help="Pass through to scripts/run_boxer_job.py when --job is used.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the docker command before executing it.",
    )
    parser.add_argument(
        "boxer_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to run_boxer.py or scripts/run_boxer_job.py. Prefix with -- to separate wrapper flags.",
    )
    return parser


def ensure_host_dirs() -> None:
    for name in ("ckpts", "sample_data", "output", "logs"):
        (REPO_ROOT / name).mkdir(parents=True, exist_ok=True)


def normalize_forwarded_args(values: list[str]) -> list[str]:
    if values[:1] == ["--"]:
        return values[1:]
    return values


def build_compose_command(args: argparse.Namespace) -> list[str]:
    service = "boxer-gpu" if args.gpu else "boxer"
    command = ["docker", "compose"]
    if args.gpu:
        command += ["--profile", "gpu"]
    command += ["run", "--rm", service, "python"]
    if args.job:
        command.append("scripts/run_boxer_job.py")
        if args.backend is not None:
            command += ["--backend", args.backend]
    else:
        command.append("run_boxer.py")
    command += normalize_forwarded_args(args.boxer_args)
    return command


def maybe_build(args: argparse.Namespace) -> None:
    if not args.build:
        return
    service = "boxer-gpu" if args.gpu else "boxer"
    command = ["docker", "compose"]
    if args.gpu:
        command += ["--profile", "gpu"]
    command += ["build", service]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.backend is not None and not args.job:
        parser.error("--backend can only be used together with --job")

    ensure_host_dirs()
    maybe_build(args)

    command = build_compose_command(args)
    if args.print_command:
        print(" ".join(command))

    completed = subprocess.run(command, cwd=REPO_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
