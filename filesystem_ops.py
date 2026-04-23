from __future__ import annotations

import base64
import mimetypes
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

MAX_TEXT_CHARS = 8000
MAX_BINARY_PREVIEW_BYTES = 4096


def _isoformat_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _path_metadata(target: Path, base_directory: str) -> dict[str, Any]:
    stats = target.stat()
    base = Path(base_directory).resolve()

    return {
        "path": str(target),
        "relative_path": str(target.relative_to(base)) if (target == base or base in target.parents) else str(target),
        "exists": target.exists(),
        "name": target.name,
        "suffix": target.suffix,
        "file_type": "directory" if target.is_dir() else "file" if target.is_file() else "other",
        "size_bytes": stats.st_size,
        "mime_type": mimetypes.guess_type(str(target))[0] or "application/octet-stream",
        "created_at": _isoformat_timestamp(stats.st_ctime),
        "modified_at": _isoformat_timestamp(stats.st_mtime),
        "accessed_at": _isoformat_timestamp(stats.st_atime),
        "permissions_octal": oct(stats.st_mode & 0o777),
    }


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


def write_file(*, base_directory: str, path: str, content: str, mode: str = "overwrite") -> dict[str, Any]:
    target = _resolve_path(base_directory, path)
    target.parent.mkdir(parents=True, exist_ok=True)

    normalized_mode = (mode or "overwrite").strip().lower()
    if normalized_mode not in {"overwrite", "append"}:
        return _result(
            "write_file",
            False,
            "Invalid mode. Use 'overwrite' or 'append'.",
            path=str(target),
            mode=mode,
        )

    file_mode = "a" if normalized_mode == "append" else "w"
    with target.open(file_mode, encoding="utf-8") as handle:
        handle.write(content)

    return _result(
        "write_file",
        True,
        f"Wrote to file: {target}",
        path=str(target),
        mode=normalized_mode,
        bytes_written=len(content.encode("utf-8")),
    )


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


def get_metadata(*, base_directory: str, path: str) -> dict[str, Any]:
    target = _resolve_path(base_directory, path)

    if not target.exists():
        return _result("get_metadata", False, f"Path not found: {target}", path=str(target), exists=False)

    metadata = _path_metadata(target, base_directory)
    return _result("get_metadata", True, f"Collected metadata for: {target}", **metadata)


OPERATION_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "make_directory": make_directory,
    "make_file": make_file,
    "write_file": write_file,
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
    "get_metadata": get_metadata,
}


def execute_step(base_directory: str, operation: str, args: dict[str, Any]) -> dict[str, Any]:
    args = dict(args or {})

    if operation in {"make_directory", "delete_directory", "delete_file", "read_file", "get_metadata"}:
        if "path" not in args:
            for alias in ("name", "directory", "target", "file", "filename"):
                if alias in args:
                    args["path"] = args.pop(alias)
                    break

    if operation in {"list_files", "list_directories"}:
        if "path" not in args:
            for alias in ("directory", "target"):
                if alias in args:
                    args["path"] = args.pop(alias)
                    break

    if operation in {"make_file", "write_file"}:
        if "path" not in args:
            for alias in ("filename", "name", "target"):
                if alias in args:
                    args["path"] = args.pop(alias)
                    break

        if "content" not in args:
            for alias in ("text", "body"):
                if alias in args:
                    args["content"] = args.pop(alias)
                    break

    if operation == "find_directory":
        if "name" not in args:
            for alias in ("directory_name", "folder_name", "query"):
                if alias in args:
                    args["name"] = args.pop(alias)
                    break

        if "path" not in args and "directory" in args:
            args["path"] = args.pop("directory")

    if operation in {"move_file", "copy_file", "move_directory", "copy_directory"}:
        if "source_path" not in args:
            for alias in ("source", "from", "from_path"):
                if alias in args:
                    args["source_path"] = args.pop(alias)
                    break

        if "destination_path" not in args:
            for alias in ("destination", "to", "to_path", "target"):
                if alias in args:
                    args["destination_path"] = args.pop(alias)
                    break

    func = OPERATION_REGISTRY.get(operation)
    if func is None:
        return _result(operation, False, f"Unknown operation: {operation}", args=args)

    try:
        return func(base_directory=base_directory, **args)
    except Exception as exc:
        return _result(operation, False, f"Execution error: {exc}", args=args)
