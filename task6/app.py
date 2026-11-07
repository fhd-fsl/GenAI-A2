"""Task 6: Standalone Web Application Server.

Exposes REST APIs and serves interactive single-page web UI via aiohttp:
- GET /: Serves single-page web UI
- GET /api/status: Hardware and model status
- GET /api/presets: Clinical cases (unseen & held-out test cases)
- POST /api/screen: Patient eligibility screening with evidence highlighting
- POST /api/compare_decoding: Live comparative execution across Uncached, KV-Cached, and Speculative
- POST /api/run_attention: Live Task 4 attention kernel benchmark runner
- GET /api/benchmarks: Tasks 1-5 pre-computed evaluation & benchmark suites
"""

from __future__ import annotations

import json
import os
import sys
import webbrowser
from pathlib import Path

from aiohttp import web

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task6.engine_manager import controller

STATIC_DIR = Path(__file__).parent / "static"


async def handle_index(request: web.Request) -> web.Response:
    """Serve main single-page interface."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return web.Response(text="index.html not found", status=404)
    with open(index_file, "r", encoding="utf-8") as f:
        html = f.read()
    return web.Response(text=html, content_type="text/html")


async def handle_status(request: web.Request) -> web.Response:
    """Return live system and GPU hardware diagnostics."""
    status = controller.get_hardware_status()
    return web.json_response(status)


async def handle_presets(request: web.Request) -> web.Response:
    """Return preset test split cases and unseen clinical vignettes."""
    return web.json_response({
        "presets": controller.preset_cases,
        "count": len(controller.preset_cases),
    })


async def handle_screen(request: web.Request) -> web.Response:
    """Execute clinical trial eligibility screening on provided patient record."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON payload in request"}, status=400)

    raw_note = data.get("note", "").strip()
    criteria = data.get("criteria", [])
    decoding_mode = data.get("decoding_mode", "kv_cached").lower()
    gamma = int(data.get("gamma", 3))
    max_new_tokens = int(data.get("max_new_tokens", 256))

    if not raw_note:
        return web.json_response({"error": "Patient description note is required."}, status=400)
    if not criteria or not isinstance(criteria, list):
        return web.json_response({"error": "At least one eligibility criterion is required."}, status=400)
    if decoding_mode not in ["kv_cached", "speculative", "uncached"]:
        return web.json_response({"error": f"Invalid decoding mode '{decoding_mode}'. Choose from kv_cached, speculative, uncached."}, status=400)

    try:
        results = controller.screen_patient(
            raw_note=raw_note,
            criteria=criteria,
            decoding_mode=decoding_mode,
            gamma=gamma,
            max_new_tokens=max_new_tokens,
        )
        return web.json_response(results)
    except Exception as e:
        return web.json_response({"error": f"Inference execution failed: {str(e)}"}, status=500)


async def handle_compare_decoding(request: web.Request) -> web.Response:
    """Run exact same prompt across Uncached, KV-Cached, and Speculative modes."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON payload in request"}, status=400)

    prompt = data.get("prompt", "").strip()
    gamma = int(data.get("gamma", 3))
    max_new_tokens = int(data.get("max_new_tokens", 48))

    if not prompt:
        return web.json_response({"error": "Prompt text is required."}, status=400)

    try:
        comparison = controller.compare_decoding_modes(
            prompt_text=prompt,
            gamma=gamma,
            max_new_tokens=max_new_tokens,
        )
        return web.json_response(comparison)
    except Exception as e:
        return web.json_response({"error": f"Decoding comparison failed: {str(e)}"}, status=500)


async def handle_run_attention(request: web.Request) -> web.Response:
    """Run Task 4 attention kernels live."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON payload in request"}, status=400)

    seq_len = int(data.get("seq_len", 512))
    head_dim = int(data.get("head_dim", 128))
    num_heads = int(data.get("num_heads", 8))
    block_r = int(data.get("block_r", 32))
    block_c = int(data.get("block_c", 32))
    is_causal = bool(data.get("is_causal", True))

    try:
        att_res = controller.run_live_attention_test(
            seq_len=seq_len,
            head_dim=head_dim,
            num_heads=num_heads,
            block_r=block_r,
            block_c=block_c,
            is_causal=is_causal,
        )
        return web.json_response(att_res)
    except Exception as e:
        return web.json_response({"error": f"Attention run failed: {str(e)}"}, status=500)


async def handle_benchmarks(request: web.Request) -> web.Response:
    """Load and return pre-computed benchmark JSON files from Tasks 1-5."""
    benchmarks = {}

    def load_json_safe(rel_path: str):
        full = PROJECT_ROOT / rel_path
        if full.exists():
            try:
                with open(full, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    benchmarks["task2_metrics"] = load_json_safe("task2/outputs/test_metrics.json")
    benchmarks["task3_benchmark"] = load_json_safe("task3/outputs/benchmark_results.json")
    benchmarks["task3_correctness"] = load_json_safe("task3/outputs/correctness_results.json")
    benchmarks["task4_benchmark"] = load_json_safe("task4/outputs/benchmark_results.json")
    benchmarks["task4_verification"] = load_json_safe("task4/outputs/verification_results.json")
    benchmarks["task5_benchmark"] = load_json_safe("task5/outputs/benchmark_results.json")
    benchmarks["task5_correctness"] = load_json_safe("task5/outputs/correctness_results.json")

    return web.json_response(benchmarks)


def create_app() -> web.Application:
    """Create and configure aiohttp web application."""
    app = web.Application()

    # Route definitions
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/presets", handle_presets)
    app.router.add_post("/api/screen", handle_screen)
    app.router.add_post("/api/compare_decoding", handle_compare_decoding)
    app.router.add_post("/api/run_attention", handle_run_attention)
    app.router.add_get("/api/benchmarks", handle_benchmarks)

    # Static assets
    if STATIC_DIR.exists():
        app.router.add_static("/static/", path=str(STATIC_DIR), name="static")

    return app


def main():
    host = "127.0.0.1"
    port = 8000
    print("\n" + "=" * 76)
    print("  Task 6: Clinical Trial Screening & Inference Benchmarking Application")
    print(f"  Access UI at: http://{host}:{port}")
    print("=" * 76 + "\n")

    app = create_app()
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    main()

