from __future__ import annotations

import base64
import mimetypes
import shutil
from pathlib import Path
from typing import Any, Callable

MAX_TEXT_CHARS = 8000
MAX_BINARY_PREVIEW_BYTES = 4096


def _resolve_path(base_directory: str, path: str) -> Path:
    base = Path(base_directory).resolve()
    candidate = (base / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()

    # keep operations scoped to base directory
    if base == candidate or base in candidate.parents:
        return candidate

    raise ValueError(f"Path '{path}' resolves outside the allowed base directory.")


def _result(operation: str, ok: bool, message: str, **details: Any) -> dict[str, Any]:
    return {
        "operation": operation,
        "ok": ok,
        "message": message,
        "details": details,
    }


def make_directory(*, base_directory: str, path: str) -> dict[str, Any]:
    target = _resolve_path(base_directory, path)
    target.mkdir(parents=True, exist_ok=True)
    return _result("make_directory", True, f"Created directory: {target}", path=str(target))


def make_file(*, base_directory: str, path: str, content: str = "") -> dict[str, Any]:
    target = _resolve_path(base_directory, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return _result("make_file", True, f"Created file: {target}", path=str(target), bytes_written=len(content.encode("utf-8")))


def move_file(*, base_directory: str, source_path: str, destination_path: str) -> dict[str, Any]:
    source = _resolve_path(base_directory, source_path)
    destination = _resolve_path(base_directory, destination_path)

    if not source.exists() or not source.is_file():
        return _result("move_file", False, f"Source file not found: {source}", source=str(source), destination=str(destination))

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return _result("move_file", True, f"Moved file: {source} -> {destination}", source=str(source), destination=str(destination))


def move_directory(*, base_directory: str, source_path: str, destination_path: str) -> dict[str, Any]:
    source = _resolve_path(base_directory, source_path)
    destination = _resolve_path(base_directory, destination_path)

    if not source.exists() or not source.is_dir():
        return _result("move_directory", False, f"Source directory not found: {source}", source=str(source), destination=str(destination))

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return _result("move_directory", True, f"Moved directory: {source} -> {destination}", source=str(source), destination=str(destination))


def copy_file(*, base_directory: str, source_path: str, destination_path: str) -> dict[str, Any]:
    source = _resolve_path(base_directory, source_path)
    destination = _resolve_path(base_directory, destination_path)

    if not source.exists() or not source.is_file():
        return _result("copy_file", False, f"Source file not found: {source}", source=str(source), destination=str(destination))

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return _result("copy_file", True, f"Copied file: {source} -> {destination}", source=str(source), destination=str(destination))


def copy_directory(*, base_directory: str, source_path: str, destination_path: str) -> dict[str, Any]:
    source = _resolve_path(base_directory, source_path)
    destination = _resolve_path(base_directory, destination_path)

    if not source.exists() or not source.is_dir():
        return _result("copy_directory", False, f"Source directory not found: {source}", source=str(source), destination=str(destination))

    shutil.copytree(source, destination, dirs_exist_ok=True)
    return _result("copy_directory", True, f"Copied directory: {source} -> {destination}", source=str(source), destination=str(destination))


def delete_file(*, base_directory: str, path: str) -> dict[str, Any]:
    target = _resolve_path(base_directory, path)

    if not target.exists() or not target.is_file():
        return _result("delete_file", False, f"File not found: {target}", path=str(target))

    target.unlink()
    return _result("delete_file", True, f"Deleted file: {target}", path=str(target))


def delete_directory(*, base_directory: str, path: str) -> dict[str, Any]:
    target = _resolve_path(base_directory, path)

    if not target.exists() or not target.is_dir():
        return _result("delete_directory", False, f"Directory not found: {target}", path=str(target))

    shutil.rmtree(target)
    return _result("delete_directory", True, f"Deleted directory: {target}", path=str(target))


def list_files(*, base_directory: str, path: str = ".") -> dict[str, Any]:
    target = _resolve_path(base_directory, path)

    if not target.exists() or not target.is_dir():
        return _result("list_files", False, f"Directory not found: {target}", path=str(target), files=[])

    files = sorted(item.name for item in target.iterdir() if item.is_file())
    return _result("list_files", True, f"Listed {len(files)} files in {target}", path=str(target), files=files, count=len(files))


def list_directories(*, base_directory: str, path: str = ".") -> dict[str, Any]:
    target = _resolve_path(base_directory, path)

    if not target.exists() or not target.is_dir():
        return _result("list_directories", False, f"Directory not found: {target}", path=str(target), directories=[])

    directories = sorted(item.name for item in target.iterdir() if item.is_dir())
    return _result(
        "list_directories",
        True,
        f"Listed {len(directories)} directories in {target}",
        path=str(target),
        directories=directories,
        count=len(directories),
    )


def find_directory(
    *,
    base_directory: str,
    name: str | None = None,
    directory_name: str | None = None,
    folder_name: str | None = None,
    path: str = ".",
) -> dict[str, Any]:
    target = _resolve_path(base_directory, path)

    if not target.exists() or not target.is_dir():
        return _result("find_directory", False, f"Directory not found: {target}", path=str(target), matches=[])

    requested_name = name or directory_name or folder_name
    if not requested_name:
        return _result(
            "find_directory",
            False,
            "Missing required search name. Provide 'name' (or alias 'directory_name').",
            path=str(target),
            matches=[],
        )

    query = requested_name.strip().lower()
    matches: list[str] = []

    for candidate in target.rglob("*"):
        if candidate.is_dir() and query in candidate.name.lower():
            matches.append(str(candidate.relative_to(Path(base_directory).resolve())))

    return _result(
        "find_directory",
        True,
        f"Found {len(matches)} matching directories for '{requested_name}'",
        path=str(target),
        name=requested_name,
        matches=sorted(matches),
        count=len(matches),
    )


def read_file(*, base_directory: str, path: str) -> dict[str, Any]:
    target = _resolve_path(base_directory, path)

    if not target.exists() or not target.is_file():
        return _result("read_file", False, f"File not found: {target}", path=str(target))

    mime_type, _ = mimetypes.guess_type(str(target))
    mime_type = mime_type or "application/octet-stream"
    raw = target.read_bytes()

    if mime_type.startswith("text/") or target.suffix.lower() in {".md", ".py", ".json", ".yaml", ".yml", ".toml", ".csv", ".xml"}:
        text = raw.decode("utf-8", errors="replace")
        truncated = len(text) > MAX_TEXT_CHARS
        visible_text = text[:MAX_TEXT_CHARS]
        return _result(
            "read_file",
            True,
            f"Read text file: {target}",
            path=str(target),
            file_kind="text",
            mime_type=mime_type,
            content=visible_text,
            truncated=truncated,
            total_chars=len(text),
        )

    if mime_type.startswith("image/"):
        encoded = base64.b64encode(raw).decode("utf-8")
        return _result(
            "read_file",
            True,
            f"Read image file: {target}",
            path=str(target),
            file_kind="image",
            mime_type=mime_type,
            data_base64=encoded,
            size_bytes=len(raw),
        )

    preview = base64.b64encode(raw[:MAX_BINARY_PREVIEW_BYTES]).decode("utf-8")
    return _result(
        "read_file",
        True,
        f"Read binary file metadata: {target}",
        path=str(target),
        file_kind="binary",
        mime_type=mime_type,
        size_bytes=len(raw),
        preview_base64=preview,
        preview_bytes=min(len(raw), MAX_BINARY_PREVIEW_BYTES),
    )


OPERATION_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "make_directory": make_directory,
    "make_file": make_file,
    "move_file": move_file,
    "move_directory": move_directory,
    "copy_file": copy_file,
    "copy_directory": copy_directory,
    "delete_file": delete_file,
    "delete_directory": delete_directory,
    "list_files": list_files,
    "list_directories": list_directories,
    "find_directory": find_directory,
    "read_file": read_file,
}


def execute_step(base_directory: str, operation: str, args: dict[str, Any]) -> dict[str, Any]:
    args = dict(args or {})

    if operation == "find_directory":
        if "name" not in args:
            for alias in ("directory_name", "folder_name", "query"):
                if alias in args:
                    args["name"] = args.pop(alias)
                    break

        if "path" not in args and "directory" in args:
            args["path"] = args.pop("directory")

    func = OPERATION_REGISTRY.get(operation)
    if func is None:
        return _result(operation, False, f"Unknown operation: {operation}", args=args)

    try:
        return func(base_directory=base_directory, **args)
    except Exception as exc:
        return _result(operation, False, f"Execution error: {exc}", args=args)
