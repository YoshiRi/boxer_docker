#!/usr/bin/env python3

from __future__ import annotations

import argparse
import cgi
import html
import json
import mimetypes
import os
import shutil
import subprocess
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from input_sources.frame_source import sanitize_sequence_name
from scripts.run_boxer_job import run_boxer_job


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "output" / "webui"
UPLOAD_ROOT = REPO_ROOT / "webui_uploads"
ALLOWED_FILE_ROOTS = [
    OUTPUT_ROOT,
    REPO_ROOT / "output",
    REPO_ROOT / "logs",
    UPLOAD_ROOT,
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

_jobs_lock = threading.Lock()
_run_lock = threading.Lock()
_jobs: dict[str, "JobState"] = {}
_runner_mode = "local"


@dataclass(slots=True)
class JobState:
    job_id: str
    label: str
    source_path: str
    created_at: str
    status: str = "queued"
    error: str | None = None
    output_root: str | None = None
    manifest: dict[str, Any] | None = None
    manifest_path: str | None = None
    log_path: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _list_sample_sequences() -> list[str]:
    sample_root = REPO_ROOT / "sample_data"
    if not sample_root.exists():
        return []
    names = []
    for entry in sorted(sample_root.iterdir()):
        if entry.is_dir():
            names.append(entry.name)
    return names


def _guess_upload_mode(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "file"
    if suffix in VIDEO_SUFFIXES:
        return "cv2"
    return "auto"


def _make_job_label(source_path: str, write_name: str) -> str:
    return f"{write_name}:{Path(source_path).name}"


def _job_output_links(job: JobState) -> list[tuple[str, str]]:
    links = []
    if job.log_path:
        links.append(("job_log", f"/files?path={quote(job.log_path)}"))
    if not job.manifest:
        return links
    for artifact in job.manifest.get("artifacts", []):
        if artifact.get("exists"):
            path = artifact.get("path")
            if isinstance(path, str):
                links.append((artifact.get("name", Path(path).name), f"/files?path={quote(path)}"))
    return links


def _source_preview(job: JobState) -> tuple[str, str] | None:
    resolved = _safe_resolve_file(job.source_path)
    if resolved is None or not resolved.exists():
        return None
    suffix = resolved.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return (f"/files?path={quote(str(resolved))}", "image")
    if suffix in VIDEO_SUFFIXES:
        return (f"/files?path={quote(str(resolved))}", "video")
    return None


def _job_preview_path(job: JobState) -> tuple[str, str] | None:
    if not job.manifest:
        return _source_preview(job)
    for artifact in job.manifest.get("artifacts", []):
        path = artifact.get("path")
        if not artifact.get("exists") or not isinstance(path, str):
            continue
        if path.endswith("_viz_current.jpg"):
            return (f"/files?path={quote(path)}", "image")
    return _source_preview(job)


def _preview_html(job: JobState) -> str:
    preview = _job_preview_path(job)
    if preview is None:
        return ""
    preview_url, preview_kind = preview
    if preview_kind == "video":
        return f'<video src="{preview_url}" controls muted></video>'
    return f'<img src="{preview_url}" alt="preview">'


def _safe_resolve_file(path_value: str) -> Path | None:
    path_value = _container_to_host_path(path_value)
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    for allowed_root in ALLOWED_FILE_ROOTS:
        try:
            path.relative_to(allowed_root.resolve())
            return path
        except ValueError:
            continue
    return None


def _save_upload(field_item, job_id: str) -> Path:
    filename = Path(field_item.filename or "upload.bin").name
    target_dir = UPLOAD_ROOT / job_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    with target_path.open("wb") as handle:
        shutil.copyfileobj(field_item.file, handle)
    return target_path


def _enqueue_job(job: JobState) -> None:
    with _jobs_lock:
        _jobs[job.job_id] = job
    worker = threading.Thread(target=_run_job, args=(job.job_id,), daemon=True)
    worker.start()


def _host_to_container_path(host_path: str) -> str:
    resolved = Path(host_path).resolve()
    root_mappings = {
        REPO_ROOT.resolve(): Path("/opt/boxer"),
        (REPO_ROOT / "sample_data").resolve(): Path("/opt/boxer/sample_data"),
        (REPO_ROOT / "output").resolve(): Path("/opt/boxer/output"),
        (REPO_ROOT / "logs").resolve(): Path("/opt/boxer/logs"),
        UPLOAD_ROOT.resolve(): Path("/opt/boxer/webui_uploads"),
    }
    for host_root, container_root in root_mappings.items():
        try:
            relative = resolved.relative_to(host_root)
            return str(container_root / relative)
        except ValueError:
            continue
    return str(resolved)


def _container_to_host_path(path_value: str) -> str:
    prefix = "/opt/boxer/"
    if path_value == "/opt/boxer":
        return str(REPO_ROOT)
    if path_value.startswith(prefix):
        return str(REPO_ROOT / path_value[len(prefix):])
    return path_value


def _normalize_manifest_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_manifest_paths(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_manifest_paths(item) for item in value]
    if isinstance(value, str):
        return _container_to_host_path(value)
    return value


def _run_job_local(job: JobState) -> dict[str, Any]:
    return run_boxer_job(**job.params)


def _docker_job_command(job: JobState) -> list[str]:
    service = "boxer-gpu" if _runner_mode == "docker-gpu" else "boxer"
    command = ["docker", "compose"]
    if _runner_mode == "docker-gpu":
        command += ["--profile", "gpu"]
    command += [
        "run",
        "--rm",
        "-v",
        f"{REPO_ROOT.resolve()}:/opt/boxer",
        service,
        "python",
        "scripts/run_boxer_job.py",
    ]
    params = job.params
    command += ["--input", _host_to_container_path(params["input"])]
    command += ["--input_mode", params.get("input_mode", "auto")]
    command += ["--output_dir", _host_to_container_path(params["output_dir"])]
    command += ["--write_name", params["write_name"]]
    command += ["--max_n", str(params["max_n"])]
    if params.get("stream_name"):
        command += ["--stream_name", params["stream_name"]]
    if params.get("track"):
        command.append("--track")
    if params.get("skip_viz"):
        command.append("--skip_viz")
    if params.get("force_cpu"):
        command.append("--force_cpu")
    labels = params.get("labels")
    if labels:
        command.append(f"--labels={','.join(labels)}")
    return command


def _run_job_docker(job: JobState) -> dict[str, Any]:
    log_dir = OUTPUT_ROOT / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.job_id}.log"
    result = subprocess.run(
        _docker_job_command(job),
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    combined_output = result.stdout + result.stderr
    log_path.write_text(combined_output, encoding="utf-8")
    job.log_path = str(log_path)
    if result.returncode != 0:
        tail = "\n".join(combined_output.strip().splitlines()[-60:])
        raise RuntimeError(f"Docker job failed with exit code {result.returncode}\n{tail}")
    sequence_name = job.params.get("stream_name") or sanitize_sequence_name(Path(job.params["input"]).name)
    manifest_path = Path(job.params["output_dir"]) / sequence_name / "job_manifest.json"
    if not manifest_path.exists():
        manifests = sorted(Path(job.params["output_dir"]).glob("*/job_manifest.json"))
        if not manifests:
            raise FileNotFoundError(f"Expected manifest was not written under {job.params['output_dir']}")
        manifest_path = manifests[-1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _normalize_manifest_paths(manifest)


def _run_job(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.status = "waiting_for_runner"
    try:
        with _run_lock:
            with _jobs_lock:
                job = _jobs[job_id]
                job.status = "running"
            if _runner_mode == "local":
                manifest = _run_job_local(job)
            else:
                manifest = _run_job_docker(job)
            with _jobs_lock:
                job = _jobs[job_id]
                job.status = "completed"
                job.manifest = manifest
                job.output_root = manifest.get("output_root")
                manifest_path = Path(job.output_root or OUTPUT_ROOT) / "job_manifest.json"
                job.manifest_path = str(manifest_path)
    except Exception:
        with _jobs_lock:
            job = _jobs[job_id]
            job.status = "failed"
            job.error = traceback.format_exc()


def _render_layout(title: str, body: str) -> bytes:
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3efe5;
      --panel: #fffdf8;
      --ink: #1f1a17;
      --muted: #6f655e;
      --line: #d7cbbb;
      --accent: #a64521;
      --accent-2: #245c73;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background:
        radial-gradient(circle at top right, rgba(166, 69, 33, 0.10), transparent 35%),
        linear-gradient(180deg, #f7f3ea, var(--bg));
      color: var(--ink);
    }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 64px; }}
    h1, h2, h3 {{ margin: 0 0 12px; font-weight: 600; }}
    p {{ margin: 0 0 14px; line-height: 1.5; }}
    .hero {{
      display: grid;
      gap: 18px;
      margin-bottom: 24px;
      padding: 24px;
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,248,239,0.92));
      box-shadow: 0 20px 45px rgba(61, 46, 36, 0.08);
    }}
    .grid {{ display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
    .panel {{
      padding: 20px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: 0 10px 28px rgba(61, 46, 36, 0.06);
    }}
    label {{ display: block; margin: 12px 0 6px; font-size: 0.95rem; }}
    input, select, button {{
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      color: var(--ink);
      font: inherit;
    }}
    input[type=checkbox] {{ width: auto; margin-right: 8px; }}
    .checkbox {{ display: flex; align-items: center; gap: 8px; margin-top: 14px; }}
    button {{
      margin-top: 18px;
      border: 0;
      background: linear-gradient(135deg, var(--accent), #c96b42);
      color: white;
      font-weight: 600;
      cursor: pointer;
    }}
    a {{ color: var(--accent-2); }}
    .muted {{ color: var(--muted); }}
    .job {{
      padding: 14px 0;
      border-top: 1px solid var(--line);
    }}
    .job:first-child {{ border-top: 0; padding-top: 0; }}
    .status {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      background: rgba(36, 92, 115, 0.12);
      color: var(--accent-2);
      font-size: 0.85rem;
    }}
    ul {{ margin: 10px 0 0 18px; padding: 0; }}
    img, video {{
      max-width: 100%;
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #ebe4d8;
    }}
    pre {{
      overflow-x: auto;
      padding: 12px;
      border-radius: 8px;
      background: #f1ebdf;
      border: 1px solid var(--line);
      white-space: pre-wrap;
    }}
    @media (max-width: 720px) {{
      main {{ padding: 20px 14px 40px; }}
      .hero, .panel {{ padding: 16px; }}
    }}
  </style>
</head>
<body>
  <main>
    {body}
  </main>
</body>
</html>"""
    return page.encode("utf-8")


class BoxerWebHandler(BaseHTTPRequestHandler):
    server_version = "BoxerWebUI/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(self._render_index())
            return
        if parsed.path.startswith("/jobs/"):
            job_id = parsed.path.split("/", 2)[2]
            self._send_html(self._render_job(job_id))
            return
        if parsed.path == "/files":
            query = parse_qs(parsed.query)
            requested = query.get("path", [None])[0]
            if not requested:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing path")
                return
            resolved = _safe_resolve_file(requested)
            if resolved is None or not resolved.exists() or not resolved.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            self._send_file(resolved)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/jobs":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")
            return
        self._handle_create_job()

    def log_message(self, format: str, *args) -> None:
        return

    def _send_html(self, payload: bytes, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def _send_file(self, path: Path) -> None:
        mime_type, _ = mimetypes.guess_type(path.name)
        content_type = mime_type or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def _render_index(self) -> bytes:
        sample_options = "".join(
            f'<option value="{html.escape(name)}">{html.escape(name)}</option>'
            for name in _list_sample_sequences()
        )
        with _jobs_lock:
            jobs = list(reversed(list(_jobs.values())))
        job_items = []
        for job in jobs[:12]:
            job_items.append(
                f"""
                <div class="job">
                  <div><a href="/jobs/{job.job_id}">{html.escape(job.label)}</a></div>
                  <div class="muted">{html.escape(job.created_at)}</div>
                  <div><span class="status">{html.escape(job.status)}</span></div>
                  {_preview_html(job)}
                </div>
                """
            )
        jobs_html = "".join(job_items) or '<p class="muted">No jobs yet.</p>'
        body = f"""
        <section class="hero">
          <h1>Boxer Web UI</h1>
          <p>Upload an image or video, or replay a local sample sequence, then inspect the emitted CSV, JPG, MP4, and manifest artifacts.</p>
          <p class="muted">Runner: <code>{html.escape(_runner_mode)}</code>. This UI wraps <code>run_boxer_job.py</code> and serializes execution to one inference job at a time.</p>
        </section>
        <section class="grid">
          <section class="panel">
            <h2>New Job</h2>
            <form action="/jobs" method="post" enctype="multipart/form-data">
              <label for="source_kind">Source</label>
              <select id="source_kind" name="source_kind">
                <option value="sample">Local sample_data sequence</option>
                <option value="upload">Upload image or video</option>
              </select>

              <label for="sample_name">Sample sequence</label>
              <select id="sample_name" name="sample_name">
                <option value="">Select sample</option>
                {sample_options}
              </select>

              <label for="upload_file">Upload file</label>
              <input id="upload_file" name="upload_file" type="file" accept="image/*,video/*">

              <label for="max_n">Max frames</label>
              <input id="max_n" name="max_n" type="number" min="1" value="30">

              <label for="write_name">Write name</label>
              <input id="write_name" name="write_name" type="text" value="webui">

              <label for="labels">Labels override</label>
              <input id="labels" name="labels" type="text" placeholder="chair,table,lamp">

              <div class="checkbox">
                <input id="track" name="track" type="checkbox" value="1" checked>
                <label for="track">Enable tracker</label>
              </div>

              <div class="checkbox">
                <input id="force_cpu" name="force_cpu" type="checkbox" value="1">
                <label for="force_cpu">Force CPU</label>
              </div>

              <button type="submit">Run Boxer</button>
            </form>
          </section>

          <section class="panel">
            <h2>Recent Jobs</h2>
            {jobs_html}
          </section>
        </section>
        """
        return _render_layout("Boxer Web UI", body)

    def _render_job(self, job_id: str) -> bytes:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            return _render_layout(
                "Job Not Found",
                '<section class="panel"><h1>Job not found</h1><p><a href="/">Back</a></p></section>',
            )
        refresh = (
            "<script>setTimeout(function(){ window.location.reload(); }, 2000);</script>"
            if job.status in {"queued", "waiting_for_runner", "running"}
            else ""
        )
        artifact_items = "".join(
            f'<li><a href="{url}">{html.escape(name)}</a></li>'
            for name, url in _job_output_links(job)
        ) or '<li class="muted">No artifacts yet.</li>'
        manifest_json = html.escape(json.dumps(job.manifest, indent=2)) if job.manifest else ""
        error_block = f"<pre>{html.escape(job.error or '')}</pre>" if job.error else ""
        body = f"""
        {refresh}
        <section class="hero">
          <h1>{html.escape(job.label)}</h1>
          <p><span class="status">{html.escape(job.status)}</span></p>
          <p class="muted">Created at {html.escape(job.created_at)}</p>
          <p><a href="/">Back to jobs</a></p>
        </section>
        <section class="grid">
          <section class="panel">
            <h2>Source</h2>
            <p><code>{html.escape(job.source_path)}</code></p>
            <h3>Artifacts</h3>
            <ul>{artifact_items}</ul>
            {_preview_html(job)}
          </section>
          <section class="panel">
            <h2>Manifest</h2>
            <pre>{manifest_json or 'Manifest not available yet.'}</pre>
            {error_block}
          </section>
        </section>
        """
        return _render_layout(f"Job {job.job_id}", body)

    def _handle_create_job(self) -> None:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        source_kind = form.getfirst("source_kind", "sample")
        job_id = uuid.uuid4().hex[:12]
        write_name = sanitize_sequence_name(form.getfirst("write_name", "webui"), default="webui")
        max_n = max(1, int(form.getfirst("max_n", "30")))
        labels_text = (form.getfirst("labels", "") or "").strip()
        labels = [item.strip() for item in labels_text.split(",") if item.strip()]
        track = form.getfirst("track") == "1"
        force_cpu = form.getfirst("force_cpu") == "1"

        if source_kind == "sample":
            sample_name = (form.getfirst("sample_name", "") or "").strip()
            if not sample_name:
                self._send_html(
                    _render_layout(
                        "Missing Sample",
                        '<section class="panel"><h1>Sample sequence is required</h1><p><a href="/">Back</a></p></section>',
                    ),
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            source_path = str((REPO_ROOT / "sample_data" / sample_name).resolve())
            input_mode = "auto"
        else:
            upload_item = form["upload_file"] if "upload_file" in form else None
            if upload_item is None or not getattr(upload_item, "filename", ""):
                self._send_html(
                    _render_layout(
                        "Missing Upload",
                        '<section class="panel"><h1>Upload image or video is required</h1><p><a href="/">Back</a></p></section>',
                    ),
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            uploaded_path = _save_upload(upload_item, job_id)
            source_path = str(uploaded_path.resolve())
            input_mode = _guess_upload_mode(uploaded_path)

        params: dict[str, Any] = {
            "input": source_path,
            "input_mode": input_mode,
            "output_dir": str(OUTPUT_ROOT),
            "write_name": write_name,
            "stream_name": sanitize_sequence_name(f"{write_name}_{job_id}"),
            "max_n": max_n,
            "track": track,
            "force_cpu": force_cpu,
        }
        if labels:
            params["labels"] = labels
        if source_kind == "upload":
            params["skip_viz"] = True

        job = JobState(
            job_id=job_id,
            label=_make_job_label(source_path, write_name),
            source_path=source_path,
            created_at=_now_iso(),
            params=params,
        )
        _enqueue_job(job)
        self._send_redirect(f"/jobs/{job.job_id}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Boxer batch web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--runner",
        default="local",
        choices=["local", "docker", "docker-gpu"],
        help="Inference runner backend",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    global _runner_mode
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _runner_mode = args.runner
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), BoxerWebHandler)
    print(f"==> Boxer Web UI listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
