#!/usr/bin/env python3
"""Compile-only evidence harness for Triton warp specialization on SM103.

The module intentionally imports Triton only inside the compile path, after the
locked target and sanitized compiler environment have been established.  It
never loads or launches the cubin produced by ``triton.compile``.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import datetime as dt
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable


LAB_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = LAB_ROOT / "manifest.json"
RAW_ROOT = LAB_ROOT / "build" / "raw"
CURATED_ROOT = LAB_ROOT / "build" / "curated"
EVIDENCE_ROOT = LAB_ROOT.parent / "evidence" / "sm103-compile-only"

LOCKED_TARGET = {
    "alias": "sm103",
    "backend": "cuda",
    "arch": 103,
    "warp_size": 32,
    "canonical": "cuda:103:32",
}

TEXT_ARTIFACT_NAMES = {
    "ttir": "ttir.mlir",
    "ttgir": "ttgir.mlir",
    "llir": "llir.ll",
    "ptx": "kernel.ptx",
}
BINARY_ARTIFACT_NAMES = {"cubin": "kernel.cubin"}
VARIANTS = ("off", "on")

# Any inherited value in this set can alter frontend semantics, pass
# registration/scheduling, LLVM optimization, target feature selection, or
# emitted PTX.  The lab intentionally resets them to upstream defaults.  The
# only dump knobs re-enabled below are the two hash-bound MLIR trace paths.
SANITIZED_COMPILER_ENV_VARS = (
    "ALLOW_LHS_TMEM_LAYOUT_CONVERSION",
    "DISABLE_LLVM_OPT",
    "DISABLE_MMA_V3",
    "DISABLE_MMA_V5",
    "DISABLE_PTXAS_OPT",
    "LLVM_ENABLE_TIMING",
    "LLVM_EXTRACT_DI_LOCAL_VARIABLES",
    "LLVM_IR_ENABLE_DUMP",
    "LLVM_PASS_PLUGIN_PATH",
    "MLIR_DISABLE_MULTITHREADING",
    "MLIR_ENABLE_DIAGNOSTICS",
    "MLIR_ENABLE_DUMP",
    "MLIR_DUMP_PATH",
    "MLIR_ENABLE_TIMING",
    "NVPTX_ENABLE_DUMP",
    "PTXAS_OPTIONS",
    "TRITON_ALLOW_NON_CONSTEXPR_GLOBALS",
    "TRITON_CONSAN_INIT_ALLOCATIONS",
    "TRITON_CUDACRT_PATH",
    "TRITON_CUDART_PATH",
    "TRITON_DEBUG",
    "TRITON_DEFAULT_FP_FUSION",
    "TRITON_DISABLE_LINE_INFO",
    "TRITON_DUMP_MIR",
    "TRITON_DUMP_PTXAS_LOG",
    "TRITON_ENABLE_ASAN",
    "TRITON_ENABLE_EXPERIMENTAL_CONSAN",
    "TRITON_ENABLE_LLVM_DEBUG",
    "TRITON_ENABLE_PYTHON_STACKTRACE",
    "TRITON_F32_DEFAULT",
    "TRITON_FPSAN_HOMOMORPHIC_CASTS",
    "TRITON_FRONT_END_DEBUGGING",
    "TRITON_INSTRUMENTATION_MODE",
    "TRITON_KERNEL_DUMP",
    "TRITON_KERNEL_OVERRIDE",
    "TRITON_LIBCUDA_PATH",
    "TRITON_LIBDEVICE_PATH",
    "TRITON_LLVM_DEBUG_ONLY",
    "TRITON_MOCK_PTX_VERSION",
    "TRITON_OVERRIDE_ARCH",
    "TRITON_OVERRIDE_DIR",
    "TRITON_PARTITION_SCHEDULING_DUMP_DATA_ONLY",
    "TRITON_PARTITION_SCHEDULING_DUMP_LOOP_ONLY",
    "TRITON_PARTITION_SCHEDULING_ENABLE_DUMP_DOT",
    "TRITON_PLUGIN_PATHS",
    "TRITON_PLUGIN_VERSION_CHECK",
    "TRITON_REPRODUCER_PATH",
    "TRITON_STORE_BINARY_ONLY",
    "USE_IR_LOC",
)


class LabError(RuntimeError):
    """Expected user-facing failure."""


def _repo_root() -> Path:
    for candidate in (LAB_ROOT, *LAB_ROOT.parents):
        if (candidate / ".git").exists() and (candidate / "python" / "triton").exists():
            return candidate
    raise LabError("cannot locate the Triton repository root")


REPO_ROOT = _repo_root()
FROZEN_TRITON_REVISION = "bf64a5db1bc8aab0fd4f0076e60f6c367852e47d"
TRUSTED_PYTHON_CASES = {
    "tma_matmul": (
        "cases/tma_matmul.py",
        "make_source",
        "d07cbb6151cf90a1a61f4bc61bc0fb2549ca58730fde7a9a3181e96261fe9935",
    ),
}
ALLOWED_COMPILE_OPTIONS = {"num_warps", "num_stages", "num_ctas"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "_asdict"):
        return _jsonable(value._asdict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"byte_count": len(value), "sha256": _sha256_bytes(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return repr(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read {MANIFEST_PATH}: {exc}") from exc
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise LabError("manifest schema_version must be 1")

    target = manifest.get("target", {})
    for key, expected in LOCKED_TARGET.items():
        if target.get(key) != expected:
            raise LabError(f"manifest target.{key} must be {expected!r}, got {target.get(key)!r}")

    families = manifest.get("families")
    cases = manifest.get("cases")
    if not isinstance(families, dict) or not families:
        raise LabError("manifest must define non-empty families")
    if not isinstance(cases, dict) or not cases:
        raise LabError("manifest must define non-empty cases")

    for family_name, family in families.items():
        patterns = family.get("patterns", {})
        if not isinstance(patterns, dict) or not patterns:
            raise LabError(f"family {family_name!r} has no patterns")
        for stage, regexes in patterns.items():
            if stage not in {*TEXT_ARTIFACT_NAMES, "trace"}:
                raise LabError(f"family {family_name!r} uses unknown stage {stage!r}")
            if not isinstance(regexes, list) or not regexes:
                raise LabError(f"family {family_name!r} stage {stage!r} must contain regexes")
            for regex in regexes:
                try:
                    re.compile(regex)
                except re.error as exc:
                    raise LabError(f"invalid regex for family {family_name!r}: {regex!r}: {exc}") from exc
        proof_stages = family.get("proof_stages")
        if not isinstance(proof_stages, list) or not proof_stages:
            raise LabError(f"family {family_name!r} must define non-empty proof_stages")
        unknown_proof_stages = set(proof_stages) - set(patterns)
        if unknown_proof_stages:
            raise LabError(
                f"family {family_name!r} proof_stages lack patterns: {sorted(unknown_proof_stages)}"
            )
        if family.get("required_match") not in {"any", "all"}:
            raise LabError(f"family {family_name!r} required_match must be 'any' or 'all'")

    for case_name, case in cases.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", case_name):
            raise LabError(f"invalid case name {case_name!r}")
        if case.get("kind") not in {"python_ast", "ir"}:
            raise LabError(f"case {case_name!r} has unsupported kind {case.get('kind')!r}")
        _case_source_paths(case_name, case)
        options = case.get("options", {})
        if not isinstance(options, dict):
            raise LabError(f"case {case_name!r} options must be an object")
        unknown_options = set(options) - ALLOWED_COMPILE_OPTIONS
        if unknown_options:
            raise LabError(
                f"case {case_name!r} uses non-reproducible compile options: {sorted(unknown_options)}"
            )
        for name, value in options.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise LabError(f"case {case_name!r} option {name!r} must be a positive integer")
        if case.get("kind") == "python_ast":
            trusted = TRUSTED_PYTHON_CASES.get(case_name)
            actual = (case.get("source"), case.get("entry"))
            if trusted is None or actual != trusted[:2]:
                raise LabError(
                    f"Python case {case_name!r} is not in the reviewed case allowlist: {actual!r}"
                )
            source_path = _case_source_paths(case_name, case)["on"]
            if _sha256_path(source_path) != trusted[2]:
                raise LabError(
                    f"reviewed Python case {case_name!r} changed; audit it and update its trusted digest"
                )
        expectations = case.get("expectations", {})
        for variant in VARIANTS:
            expectation = expectations.get(variant)
            if not isinstance(expectation, dict):
                raise LabError(f"case {case_name!r} is missing {variant!r} expectations")
            seen: set[str] = set()
            for disposition in ("required", "optional", "forbidden"):
                names = expectation.get(disposition, [])
                if not isinstance(names, list):
                    raise LabError(f"{case_name}.{variant}.{disposition} must be a list")
                unknown = set(names) - set(families)
                if unknown:
                    raise LabError(f"{case_name}.{variant} refers to unknown families: {sorted(unknown)}")
                overlap = seen.intersection(names)
                if overlap:
                    raise LabError(f"{case_name}.{variant} classifies families twice: {sorted(overlap)}")
                seen.update(names)
            missing_families = set(families) - seen
            if missing_families:
                raise LabError(
                    f"{case_name}.{variant} leaves families unclassified: {sorted(missing_families)}"
                )


def _safe_case_path(relative: str) -> Path:
    path = (LAB_ROOT / relative).resolve()
    cases_root = (LAB_ROOT / "cases").resolve()
    try:
        path.relative_to(cases_root)
    except ValueError as exc:
        raise LabError(f"case source escapes cases/: {relative!r}") from exc
    if not path.is_file():
        raise LabError(f"case source does not exist: {path}")
    return path


def _case_source_paths(case_name: str, case: dict[str, Any]) -> dict[str, Path]:
    kind = case.get("kind")
    if kind == "python_ast":
        source = case.get("source")
        if not isinstance(source, str):
            raise LabError(f"Python case {case_name!r} must define source")
        path = _safe_case_path(source)
        return {variant: path for variant in VARIANTS}
    sources = case.get("sources")
    if not isinstance(sources, dict):
        raise LabError(f"IR case {case_name!r} must define per-variant sources")
    return {variant: _safe_case_path(sources.get(variant, "")) for variant in VARIANTS}


def _static_python_case_safety(path: Path) -> list[str]:
    """Check the reviewed case shape and reject direct runtime escape hatches.

    This is defense in depth, not a Python sandbox.  The manifest separately
    refuses arbitrary Python plug-ins: every executable case is a versioned,
    reviewed file named in ``TRUSTED_PYTHON_CASES``.
    """
    errors: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return [f"cannot parse {path}: {exc}"]

    allowed_imports = {"triton", "triton.language", "triton.compiler"}
    banned_call_names = {
        "__import__",
        "compile",
        "eval",
        "exec",
        "get_current_device",
        "get_current_stream",
        "load_binary",
        "launch",
        "run",
    }
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            errors.append(f"line {node.lineno}: executable module-level statement is not allowed")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed_imports:
                    errors.append(f"line {node.lineno}: non-Triton import is not allowed: {alias.name!r}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module not in allowed_imports:
                errors.append(f"line {node.lineno}: non-Triton import is not allowed: {node.module!r}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Subscript):
                errors.append(f"line {node.lineno}: subscript-call syntax can launch a JIT kernel")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in banned_call_names:
                errors.append(f"line {node.lineno}: banned runtime call .{node.func.attr}(...)")
            elif isinstance(node.func, ast.Name) and node.func.id in banned_call_names:
                errors.append(f"line {node.lineno}: banned dynamic call {node.func.id}(...)")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"line {node.lineno}: dunder attribute access is not allowed")
    return errors


def _static_case_checks(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for case_name, case in manifest["cases"].items():
        paths = _case_source_paths(case_name, case)
        if case["kind"] == "python_ast":
            errors.extend(f"{case_name}: {error}" for error in _static_python_case_safety(paths["on"]))
        else:
            for variant, path in paths.items():
                text = path.read_text(encoding="utf-8")
                if '"ttg.target" = "cuda:103"' not in text:
                    errors.append(f"{case_name}/{variant}: IR does not declare ttg.target = cuda:103")
                if re.search(r"cuda:(?!103\b)\d+", text):
                    errors.append(f"{case_name}/{variant}: IR contains a non-SM103 CUDA target")
    return errors


def _case_spec(manifest: dict[str, Any], case_name: str) -> dict[str, Any]:
    try:
        return manifest["cases"][case_name]
    except KeyError as exc:
        available = ", ".join(sorted(manifest["cases"]))
        raise LabError(f"unknown case {case_name!r}; available cases: {available}") from exc


def _require_target(target_alias: str) -> None:
    if target_alias != LOCKED_TARGET["alias"]:
        raise LabError(
            f"this lab only accepts --target {LOCKED_TARGET['alias']}; "
            f"the compiler target is locked to {LOCKED_TARGET['canonical']}"
        )


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str | None]):
    previous = {name: os.environ.get(name) for name in updates}
    try:
        for name, value in updates.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_python_case(case_name: str, path: Path, entry_name: str, ws_enabled: bool):
    trusted = TRUSTED_PYTHON_CASES.get(case_name)
    if trusted is None or (str(path.relative_to(LAB_ROOT)), entry_name, _sha256_path(path)) != trusted:
        raise LabError(f"Python case {case_name!r} does not match its reviewed path, entry, and digest")
    safety_errors = _static_python_case_safety(path)
    if safety_errors:
        raise LabError("unsafe Python case:\n  " + "\n  ".join(safety_errors))
    module_name = f"_triton_ws_lab_{case_name}_{'on' if ws_enabled else 'off'}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise LabError(f"cannot import case module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry = getattr(module, entry_name, None)
    if not callable(entry):
        raise LabError(f"case module {path} does not expose callable {entry_name}(...)")
    return entry(ws_enabled)


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _compiler_tree_changes() -> list[str]:
    """Return changes outside this documentation package relative to the lock."""
    commands = (
        ("diff", "--name-only", FROZEN_TRITON_REVISION, "--"),
        ("ls-files", "--others", "--exclude-standard"),
    )
    changed: set[str] = set()
    for command in commands:
        output = _git_output(*command)
        if output is None:
            raise LabError(f"cannot audit repository state with git {' '.join(command)}")
        changed.update(line for line in output.splitlines() if line)
    allowed_prefix = "docs/blackwell-warp-specialization/"
    return sorted(path for path in changed if not path.startswith(allowed_prefix))


def _import_local_triton():
    """Import Triton only from this checkout and reject an already-loaded foreign copy."""
    python_root = (REPO_ROOT / "python").resolve()
    python_root_text = str(python_root)
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != python_root]
    sys.path.insert(0, python_root_text)
    importlib.invalidate_caches()
    try:
        import triton
        import triton._C.libtriton as libtriton
        from triton.backends.compiler import GPUTarget
    except Exception as exc:
        raise LabError(f"cannot import the in-tree Triton build: {type(exc).__name__}: {exc}") from exc
    package_file = Path(triton.__file__).resolve()
    expected_package = (python_root / "triton").resolve()
    try:
        package_file.relative_to(expected_package)
    except ValueError as exc:
        raise LabError(f"import resolved to foreign Triton package: {package_file}") from exc
    native_file = Path(libtriton.__file__).resolve()
    if not native_file.is_file():
        raise LabError(f"in-tree Triton native extension is missing: {native_file}")
    return triton, GPUTarget, package_file, native_file


def _source_hashes(case_name: str, case: dict[str, Any], variant: str) -> dict[str, str]:
    path = _case_source_paths(case_name, case)[variant]
    return {str(path.relative_to(LAB_ROOT)): _sha256_path(path)}


def _compile_environment(output_dir: Path, trace_path: Path | None) -> dict[str, str | None]:
    environment: dict[str, str | None] = {name: None for name in SANITIZED_COMPILER_ENV_VARS}
    environment.update({
        "PYTHONPATH": str(REPO_ROOT / "python")
        + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
        # Skip installed entry points: only backends shipped by this frozen
        # checkout may participate in compilation.
        "TRITON_BACKENDS_IN_TREE": "1",
        "TRITON_CACHE_DIR": str(output_dir / "cache"),
        "TRITON_ALWAYS_COMPILE": "1",
        "TRITON_STORE_BINARY_ONLY": "0",
        "TRITON_KERNEL_OVERRIDE": "0",
        "TRITON_OVERRIDE_DIR": None,
        # A caller-provided override would silently replace GPUTarget.arch in
        # CUDABackend.parse_options.  Clear it instead of setting one: SM103 is
        # selected only by the explicit GPUTarget("cuda", 103, 32) below.
        "TRITON_OVERRIDE_ARCH": None,
        "PTXAS_OPTIONS": None,
        "DISABLE_PTXAS_OPT": None,
        "TRITON_DUMP_PTXAS_LOG": None,
        "TRITON_INTERPRET": "0",
        "TRITON_INSTRUMENTATION_MODE": "",
        "MLIR_ENABLE_DUMP": None,
        "MLIR_DUMP_PATH": None,
    })
    if trace_path is not None:
        environment["MLIR_ENABLE_DUMP"] = "1"
        environment["MLIR_DUMP_PATH"] = str(trace_path)
    return environment


def _compile_case(
    manifest: dict[str, Any],
    case_name: str,
    variant: str,
    *,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise LabError(f"invalid WS variant {variant!r}")
    static_errors = _static_case_checks(manifest)
    if static_errors:
        raise LabError(
            "reviewed case sources failed the static target/safety checks:\n  "
            + "\n  ".join(static_errors)
        )
    case = _case_spec(manifest, case_name)
    compiler_changes = _compiler_tree_changes()
    if compiler_changes:
        rendered = "\n  ".join(compiler_changes[:40])
        raise LabError(
            "compiler source differs from the frozen Triton snapshot outside this study package:\n  "
            + rendered
        )
    output_dir = RAW_ROOT / case_name / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = output_dir / "compile-error.json"
    failure_path.unlink(missing_ok=True)
    # Invalidate any older successful record before beginning.  If this
    # process is interrupted or import fails, validation sees an incomplete
    # attempt instead of silently accepting stale artifacts.
    _write_json(
        output_dir / "compile.json",
        {
            "schema_version": 1,
            "status": "compile-started-not-launched",
            "case": case_name,
            "variant": variant,
            "target": LOCKED_TARGET,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "safety": {"cubin_loaded": False, "cubin_launched": False},
        },
    )

    def record_failure(exc: Exception, phase: str) -> None:
        failure = {
            "schema_version": 1,
            "status": "compile-error",
            "phase": phase,
            "case": case_name,
            "variant": variant,
            "target": LOCKED_TARGET,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "safety": {"cubin_loaded": False, "cubin_launched": False},
        }
        _write_json(failure_path, failure)
        _write_json(output_dir / "compile.json", failure)

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")

    environment = _compile_environment(output_dir, trace_path)
    with _temporary_env(environment):
        # Delayed import is a safety property: target checks and override
        # sanitization happen before Triton/backend initialization.
        try:
            triton, GPUTarget, triton_package_file, libtriton_file = _import_local_triton()
        except Exception as exc:
            record_failure(exc, "import-local-triton")
            raise LabError(
                "cannot import the local Triton build; run `python3 ws_study.py inspect` "
                f"for readiness details ({type(exc).__name__}: {exc})"
            ) from exc

        target = GPUTarget("cuda", 103, 32)
        if (target.backend, target.arch, target.warp_size) != ("cuda", 103, 32):
            raise LabError(f"internal target construction failed: {target!r}")

        try:
            source_paths = _case_source_paths(case_name, case)
            if case["kind"] == "python_ast":
                source = _load_python_case(case_name, source_paths[variant], case["entry"], variant == "on")
            else:
                source = str(source_paths[variant])
            # A reviewed case is executed to construct ASTSource, then all
            # architecture/compiler knobs are reset once more before the
            # backend parses options.  No case can smuggle an effective arch.
            with _temporary_env(environment):
                from triton.backends.nvidia.compiler import get_ptxas

                ptxas_file = Path(get_ptxas(103).path).resolve()
                if not ptxas_file.is_file():
                    raise LabError(f"Blackwell ptxas does not exist: {ptxas_file}")
                compiled = triton.compile(source, target=target, options=dict(case.get("options", {})))
        except Exception as exc:
            record_failure(exc, "triton-compile")
            raise LabError(f"compile failed for {case_name}/{variant}: {type(exc).__name__}: {exc}") from exc

    try:
        artifacts: dict[str, dict[str, Any]] = {}
        for stage, filename in TEXT_ARTIFACT_NAMES.items():
            if stage not in compiled.asm:
                continue
            data = compiled.asm[stage]
            if not isinstance(data, str):
                raise LabError(f"compiler returned non-text data for {stage}")
            path = output_dir / filename
            path.write_text(data, encoding="utf-8")
            artifacts[stage] = {
                "path": filename,
                "size": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
        for stage, filename in BINARY_ARTIFACT_NAMES.items():
            if stage not in compiled.asm:
                continue
            data = compiled.asm[stage]
            if not isinstance(data, bytes):
                raise LabError(f"compiler returned non-binary data for {stage}")
            path = output_dir / filename
            path.write_bytes(data)
            artifacts[stage] = {
                "path": filename,
                "size": path.stat().st_size,
                "sha256": _sha256_path(path),
                "execution_policy": "data-only; never load or launch",
            }

        if "ptx" not in artifacts or "cubin" not in artifacts:
            raise LabError(f"compiler did not produce both PTX and cubin for {case_name}/{variant}")

        metadata = _jsonable(compiled.metadata)
        record = {
            "schema_version": 1,
            "status": "compiled-not-launched",
            "case": case_name,
            "title": case["title"],
            "variant": variant,
            "target": LOCKED_TARGET,
            "constructed_gpu_target": ["cuda", 103, 32],
            "metadata": metadata,
            "options": case.get("options", {}),
            "artifacts": artifacts,
            "source_hashes": _source_hashes(case_name, case, variant),
            "manifest_sha256": _sha256_path(MANIFEST_PATH),
            "frozen_triton_revision": FROZEN_TRITON_REVISION,
            "compiler_tree_changes": compiler_changes,
            "triton_package_file": str(triton_package_file),
            "harness_sha256": _sha256_path(Path(__file__).resolve()),
            "native_extension": {
                "path": str(libtriton_file),
                "sha256": _sha256_path(libtriton_file),
            },
            "assembler": {
                "path": str(ptxas_file),
                "sha256": _sha256_path(ptxas_file),
                "version": _tool_version(ptxas_file),
            },
            "environment_contract": {
                "backends_in_tree_only": True,
                "cleared_codegen_variables": list(SANITIZED_COMPILER_ENV_VARS),
                "override_arch_cleared_before_source_and_compile": True,
                "trace_dump_is_the_only_reenabled_compiler_dump": trace_path is not None,
            },
            "git_revision": _git_output("rev-parse", "HEAD"),
            "git_dirty": bool(_git_output("status", "--porcelain")),
            "python": sys.version,
            "command": [str(Path(sys.argv[0]).resolve()), *sys.argv[1:]],
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "safety": {
                "cubin_loaded": False,
                "cubin_launched": False,
                "note": "Only triton.compile and CompiledKernel.asm were used.",
            },
        }
        _write_json(output_dir / "compile.json", record)
        return record
    except Exception as exc:
        record_failure(exc, "artifact-recording")
        if isinstance(exc, LabError):
            raise
        raise LabError(f"failed to record artifacts for {case_name}/{variant}: {exc}") from exc


def _artifact_paths(record_path: Path, record: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    root = record_path.parent.resolve()
    for stage, info in record.get("artifacts", {}).items():
        relative = info.get("path")
        if not isinstance(relative, str):
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise LabError(f"artifact path escapes its record directory: {relative!r}") from exc
        paths[stage] = path
    return paths


def _read_text_artifacts(record_path: Path, record: dict[str, Any]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for stage, path in _artifact_paths(record_path, record).items():
        if stage in TEXT_ARTIFACT_NAMES or stage == "trace":
            try:
                texts[stage] = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise LabError(f"cannot read artifact {path}: {exc}") from exc
    return texts


def _scan_families(manifest: dict[str, Any], texts: dict[str, str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for family_name, family in manifest["families"].items():
        hits: list[dict[str, Any]] = []
        proof_hits: list[dict[str, Any]] = []
        proof_matches: list[bool] = []
        for stage, regexes in family["patterns"].items():
            text = texts.get(stage)
            for regex in regexes:
                regex_hits: list[dict[str, Any]] = []
                if text is None:
                    if stage in family["proof_stages"]:
                        proof_matches.append(False)
                    continue
                lines = text.splitlines()
                matcher = re.compile(regex)
                for line_number, line in enumerate(lines, start=1):
                    if matcher.search(line):
                        regex_hits.append(
                            {
                                "stage": stage,
                                "line": line_number,
                                "text": line.strip()[:500],
                                "pattern": regex,
                            }
                        )
                        if len(regex_hits) >= 4:
                            break
                hits.extend(regex_hits)
                if stage in family["proof_stages"]:
                    proof_matches.append(bool(regex_hits))
                    proof_hits.extend(regex_hits)
        match_mode = family["required_match"]
        present = (all(proof_matches) if match_mode == "all" else any(proof_matches)) if proof_matches else False
        results[family_name] = {
            "present": present,
            "observed_anywhere": bool(hits),
            "description": family["description"],
            "proof_stages": family["proof_stages"],
            "required_match": match_mode,
            "proof_hits": proof_hits[:12],
            "trace_hits": [hit for hit in hits if hit["stage"] == "trace"][:12],
            "hits": hits[:24],
        }
    return results


def _compiled_target_tuple(metadata: dict[str, Any]) -> tuple[Any, Any, Any]:
    target = metadata.get("target", {}) if isinstance(metadata, dict) else {}
    if isinstance(target, dict):
        return target.get("backend"), target.get("arch"), target.get("warp_size")
    return None, None, None


def _evaluate_record(
    manifest: dict[str, Any], record_path: Path, record: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    errors.extend(
        f"current reviewed source failed its static target/safety check: {error}"
        for error in _static_case_checks(manifest)
    )
    case_name = record.get("case")
    variant = record.get("variant")
    try:
        case = _case_spec(manifest, case_name)
    except LabError as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": [], "families": {}}
    if variant not in VARIANTS:
        return {"ok": False, "errors": [f"invalid record variant {variant!r}"], "warnings": [], "families": {}}

    directory_case = record_path.parent.parent.name
    directory_variant = record_path.parent.name
    if (case_name, variant) != (directory_case, directory_variant):
        errors.append(
            f"record identity {case_name}/{variant} does not match directory "
            f"{directory_case}/{directory_variant}"
        )
    if record.get("schema_version") != 1:
        errors.append(f"unsupported record schema {record.get('schema_version')!r}")
    if record.get("manifest_sha256") != _sha256_path(MANIFEST_PATH):
        errors.append("record was produced from a different manifest")
    if record.get("source_hashes") != _source_hashes(case_name, case, variant):
        errors.append("record source hashes do not match the reviewed case source")
    if record.get("frozen_triton_revision") != FROZEN_TRITON_REVISION:
        errors.append("record does not name the frozen Triton source revision")
    if record.get("compiler_tree_changes") != []:
        errors.append("record reports compiler changes outside this study package")
    current_compiler_changes = _compiler_tree_changes()
    if current_compiler_changes:
        errors.append("current compiler tree differs from the frozen source snapshot")
    record_revision = record.get("git_revision")
    if not isinstance(record_revision, str) or not _git_is_ancestor(FROZEN_TRITON_REVISION, record_revision):
        errors.append(f"record git revision is not descended from the frozen snapshot: {record_revision!r}")
    if record.get("constructed_gpu_target") != ["cuda", 103, 32]:
        errors.append("record did not construct exactly GPUTarget('cuda', 103, 32)")
    if record.get("options") != case.get("options", {}):
        errors.append("record compile options differ from the reviewed manifest")
    if record.get("harness_sha256") != _sha256_path(Path(__file__).resolve()):
        errors.append("record was produced by a different ws_study.py harness")
    environment_contract = record.get("environment_contract", {})
    expected_environment_contract = {
        "backends_in_tree_only": True,
        "cleared_codegen_variables": list(SANITIZED_COMPILER_ENV_VARS),
        "override_arch_cleared_before_source_and_compile": True,
        "trace_dump_is_the_only_reenabled_compiler_dump": "trace" in record.get("artifacts", {}),
    }
    if environment_contract != expected_environment_contract:
        errors.append("record does not preserve the frozen in-tree compiler environment contract")
    safety = record.get("safety", {})
    if safety.get("cubin_loaded") is not False or safety.get("cubin_launched") is not False:
        errors.append("record does not preserve the no-load/no-launch safety invariant")
    package_file = record.get("triton_package_file")
    try:
        Path(package_file).resolve().relative_to((REPO_ROOT / "python" / "triton").resolve())
    except (TypeError, ValueError):
        errors.append(f"record used a foreign Triton Python package: {package_file!r}")

    for label, field in (("native extension", "native_extension"), ("assembler", "assembler")):
        provenance = record.get(field, {})
        tool_path_value = provenance.get("path") if isinstance(provenance, dict) else None
        try:
            tool_path = Path(tool_path_value).resolve()
        except TypeError:
            errors.append(f"record has no {label} path")
            continue
        if not tool_path.is_file():
            errors.append(f"recorded {label} is missing: {tool_path}")
        elif provenance.get("sha256") != _sha256_path(tool_path):
            errors.append(f"recorded {label} SHA-256 no longer matches")
    assembler_version = record.get("assembler", {}).get("version", {})
    if assembler_version.get("returncode") != 0:
        errors.append("recorded Blackwell assembler did not report a successful version probe")

    if record.get("target") != LOCKED_TARGET:
        errors.append(f"record target is not exactly {LOCKED_TARGET!r}")
    actual_target = _compiled_target_tuple(record.get("metadata", {}))
    if actual_target != ("cuda", 103, 32):
        errors.append(f"compiled metadata target is {actual_target!r}, expected ('cuda', 103, 32)")
    if record.get("status") != "compiled-not-launched":
        errors.append(f"unexpected compile status {record.get('status')!r}")

    artifact_paths = _artifact_paths(record_path, record)
    for required_stage in ("ptx", "cubin"):
        if required_stage not in record.get("artifacts", {}):
            errors.append(f"record omits required {required_stage} artifact")
    for stage, info in record.get("artifacts", {}).items():
        path = artifact_paths.get(stage)
        if path is None or not path.is_file():
            errors.append(f"missing {stage} artifact")
            continue
        expected_hash = info.get("sha256")
        actual_hash = _sha256_path(path)
        if expected_hash != actual_hash:
            errors.append(f"{stage} SHA-256 mismatch")
        if path.stat().st_size == 0:
            errors.append(f"{stage} artifact is empty")

    texts = _read_text_artifacts(record_path, record)
    ptx = texts.get("ptx", "")
    ptx_target_regex = manifest["target"]["ptx_target_regex"]
    if not re.search(ptx_target_regex, ptx):
        errors.append("PTX does not declare .target sm_103a")
    if re.search(r"(?m)^\s*\.target\s+sm_(?!103a(?:\s|,|$))", ptx):
        errors.append("PTX contains a non-SM103a target")

    families = _scan_families(manifest, texts)
    expectation = case["expectations"][variant]
    dispositions: dict[str, str] = {}
    for disposition in ("required", "optional", "forbidden"):
        for family_name in expectation.get(disposition, []):
            dispositions[family_name] = disposition
    for family_name, result in families.items():
        disposition = dispositions.get(family_name, "unclassified")
        result["expectation"] = disposition
        if disposition == "required" and not result["present"]:
            errors.append(f"required family {family_name!r} is absent")
        elif disposition == "forbidden" and result["observed_anywhere"]:
            errors.append(f"forbidden family {family_name!r} is present")
        elif disposition == "optional" and not result["present"]:
            warnings.append(f"optional family {family_name!r} was not observed")

    return {
        "ok": not errors,
        "case": case_name,
        "variant": variant,
        "record": str(record_path.relative_to(LAB_ROOT)),
        "errors": errors,
        "warnings": warnings,
        "families": families,
    }


def _record_paths(case_filter: str | None = None, variant_filter: str | None = None) -> list[Path]:
    paths = sorted(RAW_ROOT.glob("*/*/compile.json")) if RAW_ROOT.exists() else []
    selected: list[Path] = []
    for path in paths:
        variant = path.parent.name
        case_name = path.parent.parent.name
        if case_filter is not None and case_name != case_filter:
            continue
        if variant_filter is not None and variant != variant_filter:
            continue
        selected.append(path)
    return selected


def _load_record(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read compile record {path}: {exc}") from exc


def _find_repo_tool(names: Iterable[str]) -> Path | None:
    for name in names:
        env_name = "TRITON_" + name.upper().replace("-", "_") + "_PATH"
        if value := os.environ.get(env_name):
            candidate = Path(value).expanduser().resolve()
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        candidate = REPO_ROOT / "python" / "triton" / "backends" / "nvidia" / "bin" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
        for candidate in sorted((REPO_ROOT / "build").glob(f"*/bin/{name}")):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()
    return None


def _tool_version(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"found": False}
    try:
        result = subprocess.run(
            [str(path), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = (result.stdout + result.stderr).strip()
        return {"found": True, "path": str(path), "returncode": result.returncode, "version": output[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"found": True, "path": str(path), "error": str(exc)}


def _triton_import_probe() -> dict[str, Any]:
    expected_package = str((REPO_ROOT / "python" / "triton").resolve())
    code = (
        "from pathlib import Path; import triton; "
        "from triton.backends.compiler import GPUTarget; "
        "t=GPUTarget('cuda',103,32); "
        f"assert Path(triton.__file__).resolve().is_relative_to(Path({expected_package!r})); "
        "print(triton.__file__); print(getattr(triton,'__version__','unknown')); print(t)"
    )
    env = os.environ.copy()
    for name in SANITIZED_COMPILER_ENV_VARS:
        env.pop(name, None)
    env["PYTHONPATH"] = str(REPO_ROOT / "python") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["TRITON_BACKENDS_IN_TREE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ready": False, "error": str(exc)}
    return {
        "ready": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def cmd_inspect(args: argparse.Namespace) -> int:
    manifest = _load_manifest()
    static_errors = _static_case_checks(manifest)
    compiler_tree_changes = _compiler_tree_changes()
    tools = {
        "triton_opt": _tool_version(_find_repo_tool(("triton-opt",))),
        "ptxas_blackwell": _tool_version(_find_repo_tool(("ptxas-blackwell",))),
        "cuobjdump": _tool_version(_find_repo_tool(("cuobjdump",))),
        "nvdisasm": _tool_version(_find_repo_tool(("nvdisasm",))),
    }
    report = {
        "lab": str(LAB_ROOT),
        "repo": str(REPO_ROOT),
        "target": LOCKED_TARGET,
        "compile_only": True,
        "cubin_launch_supported": False,
        "manifest_sha256": _sha256_path(MANIFEST_PATH),
        "git_revision": _git_output("rev-parse", "HEAD"),
        "cases": {
            name: {
                "title": case["title"],
                "kind": case["kind"],
                "sources": {
                    variant: str(path.relative_to(LAB_ROOT))
                    for variant, path in _case_source_paths(name, case).items()
                },
            }
            for name, case in manifest["cases"].items()
        },
        "interesting_passes": manifest["interesting_passes"],
        "static_case_errors": static_errors,
        "compiler_tree_changes": compiler_tree_changes,
        "triton_import": _triton_import_probe(),
        "tools": tools,
    }
    report["compile_ready"] = (
        not static_errors
        and not compiler_tree_changes
        and report["triton_import"]["ready"]
        and tools["ptxas_blackwell"].get("returncode") == 0
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"lab: {report['lab']}")
        print(f"locked target: {LOCKED_TARGET['canonical']} (alias: sm103)")
        print("execution policy: compile/disassemble only; cubin launch is not implemented")
        print(f"static cases: {'OK' if not static_errors else 'FAILED'}")
        for error in static_errors:
            print(f"  ERROR {error}")
        for path in compiler_tree_changes:
            print(f"  ERROR compiler source differs from lock: {path}")
        print(f"local Triton import: {'ready' if report['triton_import']['ready'] else 'not ready'}")
        if report["triton_import"].get("stderr"):
            print("  " + report["triton_import"]["stderr"].splitlines()[-1])
        for name, info in tools.items():
            state = info.get("path", "not found") if info.get("found") else "not found"
            print(f"{name}: {state}")
        print("cases:")
        for name, case in report["cases"].items():
            print(f"  {name}: {case['title']}")
        print("interesting passes:")
        for pass_name in report["interesting_passes"]:
            print(f"  {pass_name}")
        print(f"compile readiness: {'READY' if report['compile_ready'] else 'NOT READY'}")
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    _require_target(args.target)
    manifest = _load_manifest()
    _case_spec(manifest, args.case)
    static_errors = _static_case_checks(manifest)
    if static_errors:
        raise LabError("static case checks failed:\n  " + "\n  ".join(static_errors))
    record = _compile_case(manifest, args.case, args.ws)
    output_dir = RAW_ROOT / args.case / args.ws
    print(f"compiled {args.case}/{args.ws} for {record['target']['canonical']}")
    print(f"raw artifacts: {output_dir}")
    print("cubin status: generated as data, never loaded or launched")
    return 0


_TRACE_HEADER = re.compile(r"(?m)^.*IR Dump (?:Before|After).*$")


def _trace_sections(text: str) -> list[tuple[str, str]]:
    matches = list(_TRACE_HEADER.finditer(text))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group(0).strip(), text[match.start():end]))
    return sections


def cmd_trace(args: argparse.Namespace) -> int:
    manifest = _load_manifest()
    _case_spec(manifest, args.case)
    variants = VARIANTS if args.ws == "both" else (args.ws,)
    for variant in variants:
        trace_dir = RAW_ROOT / args.case / variant / "trace"
        full_path = trace_dir / "full.mlir.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_trace-worker",
            "--case",
            args.case,
            "--ws",
            variant,
        ]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
            raise LabError(f"trace worker failed for {args.case}/{variant}: {detail}")
        if not full_path.is_file() or full_path.stat().st_size == 0:
            raise LabError(
                f"MLIR pass trace for {args.case}/{variant} is empty; "
                "ensure the native pass manager honors MLIR_ENABLE_DUMP"
            )
        record_path = RAW_ROOT / args.case / variant / "compile.json"
        record = _load_record(record_path)
        record.setdefault("artifacts", {})["trace"] = {
            "path": str(full_path.relative_to(record_path.parent)),
            "size": full_path.stat().st_size,
            "sha256": _sha256_path(full_path),
            "execution_policy": "compiler pass dump only",
        }
        record["trace_worker_command"] = command
        _write_json(record_path, record)
        text = full_path.read_text(encoding="utf-8", errors="replace")
        sections = _trace_sections(text)
        index = {
            "case": args.case,
            "variant": variant,
            "target": LOCKED_TARGET,
            "full_trace": str(full_path.relative_to(LAB_ROOT)),
            "section_count": len(sections),
            "headers": [header for header, _ in sections],
        }
        _write_json(trace_dir / "pass-index.json", index)
        if args.passes == "all":
            print(f"trace {args.case}/{variant}: {full_path} ({len(sections)} IR dump sections)")
            continue
        needle = args.passes.casefold()
        selected = [(header, section) for header, section in sections if needle in header.casefold()]
        if not selected:
            suggestions = [header for header, _ in sections if any(part in header.casefold() for part in needle.split("-"))]
            hint = "\n".join(f"  {item}" for item in suggestions[:12])
            raise LabError(
                f"pass filter {args.passes!r} matched no IR dump section for {args.case}/{variant}."
                + (f"\nNearby headers:\n{hint}" if hint else " See trace/pass-index.json for available headers.")
            )
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.passes).strip("_") or "pass"
        filtered_path = trace_dir / f"pass-{slug}.mlir.log"
        filtered_path.write_text("".join(section for _, section in selected), encoding="utf-8")
        print(f"trace {args.case}/{variant}: {filtered_path} ({len(selected)} matching sections)")
    return 0


def cmd_trace_worker(args: argparse.Namespace) -> int:
    """Compile one traced variant in its own process so MLIR dump streams flush."""
    manifest = _load_manifest()
    _case_spec(manifest, args.case)
    trace_path = RAW_ROOT / args.case / args.ws / "trace" / "full.mlir.log"
    _compile_case(manifest, args.case, args.ws, trace_path=trace_path)
    return 0


def _disassembler(tool_choice: str) -> tuple[str, Path]:
    choices = (tool_choice,) if tool_choice != "auto" else ("cuobjdump", "nvdisasm")
    for choice in choices:
        path = _find_repo_tool((choice,))
        if path is not None:
            return choice, path
    raise LabError(
        "no cubin disassembler found; build/download Triton's cuobjdump or nvdisasm, "
        "or set TRITON_CUOBJDUMP_PATH/TRITON_NVDISASM_PATH"
    )


def cmd_disassemble(args: argparse.Namespace) -> int:
    manifest = _load_manifest()
    _case_spec(manifest, args.case)
    variants = VARIANTS if args.ws == "both" else (args.ws,)
    tool_name, tool_path = _disassembler(args.tool)
    processed = 0
    for variant in variants:
        record_path = RAW_ROOT / args.case / variant / "compile.json"
        if not record_path.is_file():
            if args.ws == "both":
                continue
            raise LabError(f"compile record is missing: {record_path}")
        record = _load_record(record_path)
        evaluation = _evaluate_record(manifest, record_path, record)
        if not evaluation["ok"]:
            raise LabError(
                f"refusing to disassemble invalid/stale record {args.case}/{variant}: "
                + "; ".join(evaluation["errors"])
            )
        cubin = _artifact_paths(record_path, record).get("cubin")
        if cubin is None or not cubin.is_file():
            raise LabError(f"cubin artifact is missing for {args.case}/{variant}")
        command = [str(tool_path), "-sass", str(cubin)] if tool_name == "cuobjdump" else [str(tool_path), str(cubin)]
        result = subprocess.run(command, check=False, capture_output=True, timeout=120)
        output_dir = record_path.parent
        stderr_path = output_dir / "disassemble.stderr.log"
        stderr_path.write_bytes(result.stderr)
        disassembly_path = output_dir / "kernel.sass"
        if result.returncode != 0:
            _write_json(
                output_dir / "disassemble.json",
                {
                    "status": "error",
                    "case": args.case,
                    "variant": variant,
                    "tool": str(tool_path),
                    "command": command,
                    "returncode": result.returncode,
                    "stderr": str(stderr_path.relative_to(output_dir)),
                    "cubin_sha256": _sha256_path(cubin),
                    "cubin_executed": False,
                },
            )
            raise LabError(
                f"{tool_name} failed for {args.case}/{variant} with exit code {result.returncode}; "
                f"see {stderr_path}"
            )
        disassembly_path.write_bytes(result.stdout)
        _write_json(
            output_dir / "disassemble.json",
            {
                "status": "disassembled-not-executed",
                "case": args.case,
                "variant": variant,
                "tool": str(tool_path),
                "command": command,
                "returncode": result.returncode,
                "output": disassembly_path.name,
                "output_sha256": _sha256_path(disassembly_path),
                "cubin_sha256": _sha256_path(cubin),
                "target": LOCKED_TARGET,
                "cubin_executed": False,
            },
        )
        print(f"disassembled {args.case}/{variant}: {disassembly_path}")
        processed += 1
    if processed == 0:
        raise LabError(f"no compiled variants found for case {args.case!r}")
    print("cubin status: decoded as data, never loaded or launched")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    manifest = _load_manifest()
    if args.case is not None:
        _case_spec(manifest, args.case)
    static_errors = _static_case_checks(manifest)
    if static_errors:
        raise LabError("static case checks failed:\n  " + "\n  ".join(static_errors))
    paths = _record_paths(args.case, None if args.ws == "both" else args.ws)
    if not paths:
        if args.require_artifacts:
            raise LabError("no compile records matched; compile a case before artifact validation")
        print("static manifest/source validation passed; no compiled artifacts were present")
        return 0

    failed = False
    for path in paths:
        record = _load_record(path)
        result = _evaluate_record(manifest, path, record)
        _write_json(path.parent / "validation.json", result)
        label = f"{result.get('case')}/{result.get('variant')}"
        print(f"{label}: {'PASS' if result['ok'] else 'FAIL'}")
        for error in result["errors"]:
            print(f"  ERROR {error}")
        if args.show_optional:
            for warning in result["warnings"]:
                print(f"  NOTE  {warning}")
        failed |= not result["ok"]
    if failed:
        raise LabError("one or more compile records failed structural validation")
    print(f"validated {len(paths)} compile-only record(s); no cubin was loaded or launched")
    return 0


def _metadata_summary(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    keys = ("num_warps", "num_ctas", "shared", "tmem_size", "maxnreg", "name")
    return {key: metadata.get(key) for key in keys if metadata.get(key) is not None}


def _markdown_report(manifest: dict[str, Any], evaluations: list[dict[str, Any]]) -> str:
    lines = [
        "# Curated SM103 compile-only evidence",
        "",
        f"Locked target: `{LOCKED_TARGET['canonical']}`. Cubins were compiled; any disassembly treats them only as data, and none were launched.",
        "",
        "| Case | WS | Result | Required families present |",
        "|---|---:|---:|---|",
    ]
    for result in evaluations:
        required = [
            name
            for name, family in result["families"].items()
            if family.get("expectation") == "required" and family.get("present")
        ]
        lines.append(
            f"| `{result['case']}` | `{result['variant']}` | "
            f"{'PASS' if result['ok'] else 'FAIL'} | {', '.join(f'`{name}`' for name in required) or '—'} |"
        )
    for result in evaluations:
        lines.extend(["", f"## {result['case']} / WS {result['variant']}", ""])
        if result["errors"]:
            lines.append("Errors: " + "; ".join(result["errors"]))
            lines.append("")
        lines.extend(["| Family | Contract | Observed | First evidence |", "|---|---|---:|---|"])
        for name, family in result["families"].items():
            observed = family["observed_anywhere"] if family.get("expectation") == "forbidden" else family["present"]
            hit = (family["proof_hits"] or family["hits"] or [None])[0] if observed else None
            evidence = (
                f"`{hit['stage']}:{hit['line']}` `{hit['text'].replace('|', '&#124;')}`" if hit else "—"
            )
            lines.append(
                f"| `{name}` | {family.get('expectation', 'unclassified')} | "
                f"{'yes' if observed else 'no'} | {evidence} |"
            )
    lines.extend(
        [
            "",
            "This report proves compiler structure only. It contains no runtime correctness or performance result.",
            "",
        ]
    )
    return "\n".join(lines)


def _html_report(evaluations: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    details: list[str] = []
    for result in evaluations:
        required = [
            name
            for name, family in result["families"].items()
            if family.get("expectation") == "required" and family.get("present")
        ]
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(result['case'])}</code></td>"
            f"<td><code>{html.escape(result['variant'])}</code></td>"
            f"<td class={'ok' if result['ok'] else 'bad'}>{'PASS' if result['ok'] else 'FAIL'}</td>"
            f"<td>{', '.join(f'<code>{html.escape(name)}</code>' for name in required) or '—'}</td>"
            "</tr>"
        )
        family_rows: list[str] = []
        for name, family in result["families"].items():
            observed = family["observed_anywhere"] if family.get("expectation") == "forbidden" else family["present"]
            hit = (family["proof_hits"] or family["hits"] or [None])[0] if observed else None
            evidence = (
                f"<code>{html.escape(hit['stage'])}:{hit['line']}</code> "
                f"<span class=snippet>{html.escape(hit['text'])}</span>"
                if hit
                else "—"
            )
            family_rows.append(
                "<tr>"
                f"<td><code>{html.escape(name)}</code></td>"
                f"<td>{html.escape(family.get('expectation', 'unclassified'))}</td>"
                f"<td>{'yes' if observed else 'no'}</td>"
                f"<td>{evidence}</td>"
                "</tr>"
            )
        errors = "".join(f"<li>{html.escape(error)}</li>" for error in result["errors"])
        details.append(
            f"<section><h2>{html.escape(result['case'])} / WS {html.escape(result['variant'])}</h2>"
            + (f"<ul class=bad>{errors}</ul>" if errors else "")
            + "<table><thead><tr><th>Family</th><th>Contract</th><th>Observed</th><th>First evidence</th></tr></thead>"
            f"<tbody>{''.join(family_rows)}</tbody></table></section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SM103 compile-only evidence</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#202124}}
table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}th,td{{border:1px solid #d8dee4;padding:.45rem;vertical-align:top;text-align:left}}
th{{background:#f6f8fa}}code,.snippet{{font-family:ui-monospace,SFMono-Regular,monospace}}.snippet{{overflow-wrap:anywhere}}.ok{{color:#137333;font-weight:700}}.bad{{color:#b3261e;font-weight:700}}
</style></head><body>
<h1>Curated SM103 compile-only evidence</h1>
<p>Locked target: <code>{LOCKED_TARGET['canonical']}</code>. Cubins were compiled; any disassembly treats them only as data, and none were launched.</p>
<table><thead><tr><th>Case</th><th>WS</th><th>Result</th><th>Required families present</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{''.join(details)}
<p><strong>Scope:</strong> compiler structure only; no runtime correctness or performance result.</p>
</body></html>
"""


def _publish_curated_evidence(*, force: bool) -> tuple[int, int]:
    """Copy deterministic curated files to the versionable evidence tree."""
    names = ("evidence.json", "evidence.md", "index.html")
    missing = [name for name in names if not (CURATED_ROOT / name).is_file()]
    if missing:
        raise LabError(f"curated intermediate is incomplete: missing {', '.join(missing)}")

    payloads = {name: (CURATED_ROOT / name).read_bytes() for name in names}
    conflicts = [
        EVIDENCE_ROOT / name
        for name, payload in payloads.items()
        if (EVIDENCE_ROOT / name).is_file() and (EVIDENCE_ROOT / name).read_bytes() != payload
    ]
    if conflicts and not force:
        rendered = "\n  ".join(str(path) for path in conflicts)
        raise LabError(
            "refusing to overwrite different committed evidence; inspect the build/curated diff "
            f"or rerun with --force:\n  {rendered}"
        )

    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    written = 0
    unchanged = 0
    for name, payload in payloads.items():
        destination = EVIDENCE_ROOT / name
        if destination.is_file() and destination.read_bytes() == payload:
            unchanged += 1
            continue
        destination.write_bytes(payload)
        written += 1
    return written, unchanged


def cmd_render(args: argparse.Namespace) -> int:
    manifest = _load_manifest()
    if args.case is not None:
        _case_spec(manifest, args.case)
    if args.force and not args.publish:
        raise LabError("--force is only meaningful together with --publish")
    if args.publish and (args.case is not None or args.ws != "both"):
        raise LabError("--publish requires the canonical unfiltered view (all cases and --ws both)")
    paths = _record_paths(args.case, None if args.ws == "both" else args.ws)
    if not paths:
        raise LabError("no compile records matched; render refuses to invent evidence")
    if args.publish:
        expected = {(case_name, variant) for case_name in manifest["cases"] for variant in VARIANTS}
        observed = {(path.parent.parent.name, path.parent.name) for path in paths}
        missing = sorted(expected - observed)
        if missing:
            labels = ", ".join(f"{case}/{variant}" for case, variant in missing)
            raise LabError(f"canonical publication is incomplete; missing compile records: {labels}")
        missing_traces = []
        for path in paths:
            record = _load_record(path)
            if "trace" not in record.get("artifacts", {}):
                missing_traces.append(f"{path.parent.parent.name}/{path.parent.name}")
        if missing_traces:
            raise LabError(
                "canonical publication requires a hashed pass trace for every variant; missing: "
                + ", ".join(missing_traces)
            )
    evaluations: list[dict[str, Any]] = []
    for path in paths:
        record = _load_record(path)
        evaluation = _evaluate_record(manifest, path, record)
        evaluation["metadata_summary"] = _metadata_summary(record)
        evaluations.append(evaluation)

    CURATED_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "scope": "compile-only; no cubin launch and no runtime claims",
        "target": LOCKED_TARGET,
        "manifest_sha256": _sha256_path(MANIFEST_PATH),
        "source_records": [
            {
                "path": str(path.relative_to(LAB_ROOT)),
                "sha256": _sha256_path(path),
            }
            for path in paths
        ],
        "evaluations": evaluations,
    }
    _write_json(CURATED_ROOT / "evidence.json", payload)
    (CURATED_ROOT / "evidence.md").write_text(_markdown_report(manifest, evaluations), encoding="utf-8")
    (CURATED_ROOT / "index.html").write_text(_html_report(evaluations), encoding="utf-8")
    print(f"rendered {len(evaluations)} record(s) into {CURATED_ROOT}")
    if args.publish:
        failed = [f"{item['case']}/{item['variant']}" for item in evaluations if not item["ok"]]
        if failed:
            raise LabError(
                "publication requires every structural validation to pass; failed: " + ", ".join(failed)
            )
        written, unchanged = _publish_curated_evidence(force=args.force)
        print(f"published evidence: {EVIDENCE_ROOT} ({written} written, {unchanged} unchanged)")
    else:
        print("publication skipped; review build/curated, then rerun render --publish")
    print("scope: compiler evidence only; no runtime results were created")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Locked-target, compile-only SM103 warp-specialization evidence lab",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect cases and local toolchain readiness")
    inspect_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    inspect_parser.set_defaults(func=cmd_inspect)

    compile_parser = subparsers.add_parser("compile", help="compile one case without loading or launching it")
    compile_parser.add_argument("--case", required=True, help="manifest case name")
    compile_parser.add_argument(
        "--target",
        required=True,
        choices=("sm103",),
        help="the only accepted alias; maps to cuda:103:32",
    )
    compile_parser.add_argument("--ws", required=True, choices=VARIANTS, help="compile WS on or off variant")
    compile_parser.set_defaults(func=cmd_compile)

    trace_parser = subparsers.add_parser("trace", help="recompile while recording pass-manager IR dumps")
    trace_parser.add_argument("--case", required=True, help="manifest case name")
    trace_parser.add_argument("--passes", required=True, help="all or a pass-name substring")
    trace_parser.add_argument("--ws", choices=(*VARIANTS, "both"), default="on", help="variant to trace")
    trace_parser.set_defaults(func=cmd_trace)

    disassemble_parser = subparsers.add_parser("disassemble", help="decode cubin data without loading it")
    disassemble_parser.add_argument("--case", required=True, help="manifest case name")
    disassemble_parser.add_argument("--ws", choices=(*VARIANTS, "both"), default="both")
    disassemble_parser.add_argument("--tool", choices=("auto", "cuobjdump", "nvdisasm"), default="auto")
    disassemble_parser.set_defaults(func=cmd_disassemble)

    render_parser = subparsers.add_parser("render", help="derive curated Markdown/HTML/JSON from raw records")
    render_parser.add_argument("--case", help="limit rendering to one case")
    render_parser.add_argument("--ws", choices=(*VARIANTS, "both"), default="both")
    render_parser.add_argument(
        "--publish",
        action="store_true",
        help="copy the canonical view to ../evidence/sm103-compile-only",
    )
    render_parser.add_argument(
        "--force",
        action="store_true",
        help="with --publish, replace committed evidence whose content differs",
    )
    render_parser.set_defaults(func=cmd_render)

    validate_parser = subparsers.add_parser("validate", help="check target, hashes, and structural families")
    validate_parser.add_argument("--case", help="limit validation to one case")
    validate_parser.add_argument("--ws", choices=(*VARIANTS, "both"), default="both")
    validate_parser.add_argument("--require-artifacts", action="store_true", help="fail if no compile record exists")
    validate_parser.add_argument("--show-optional", action="store_true", help="show absent optional-family notes")
    validate_parser.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "_trace-worker":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--case", required=True)
        parser.add_argument("--ws", required=True, choices=VARIANTS)
        args = parser.parse_args(raw_argv[1:])
        args.func = cmd_trace_worker
    else:
        parser = _build_parser()
        args = parser.parse_args(raw_argv)
    try:
        return int(args.func(args))
    except LabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
