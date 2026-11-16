"""Install the pinned H3-08 workflows in a Windows ComfyUI portable directory."""
from argparse import ArgumentParser
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
from tempfile import TemporaryFile
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
BLOCK = 8 * 1024**2
GIB = 1024**3
HEADERS = {"User-Agent": "H3-08-ComfyUI-Setup/1.0", "Accept-Encoding": "identity"}


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path is outside the installation directory: {relative}")
    return path


def sha256(path, normalize_text=False):
    if normalize_text:
        return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def missing_files(root, entries, normalized=False):
    missing = []
    for relative, manifest in entries.items():
        path = inside(root, relative)
        if not path.exists():
            missing.append((relative, path, manifest))
        elif not path.is_file() or sha256(path, normalized and manifest.get("text", False)) != manifest["sha256"]:
            raise ValueError(f"Different file already exists: {path}")
    return missing


def install_file(target, content):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        output.write(content)


def install_optimizer(optimizer, missing, supplied_zip):
    if not missing:
        return
    if supplied_zip is not None:
        with ZipFile(supplied_zip) as archive:
            copy_optimizer(optimizer, missing, archive)
        return
    print("Downloading the pinned H3-Optimizations archive...", flush=True)
    request = Request(optimizer["url"], headers=HEADERS)
    with TemporaryFile() as archive_file:
        with urlopen(request, timeout=90) as response:
            shutil.copyfileobj(response, archive_file, BLOCK)
        archive_file.seek(0)
        with ZipFile(archive_file) as archive:
            copy_optimizer(optimizer, missing, archive)


def copy_optimizer(optimizer, missing, archive):
    verified = []
    for relative, target, manifest in missing:
        data = archive.read(optimizer["archive_prefix"] + relative)
        checked = data.replace(b"\r\n", b"\n") if manifest["text"] else data
        if hashlib.sha256(checked).hexdigest() != manifest["sha256"]:
            raise ValueError(f"H3-Optimizations archive failed verification: {relative}")
        verified.append((target, data))
    for target, data in verified:
        install_file(target, data)


def get_model(target, item):
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name("." + target.name + ".h3-part")
    total = item["bytes"]
    url = item["source"].replace("/blob/", "/resolve/", 1)
    print(f"{target.name} ({total / GIB:.2f} GiB)", flush=True)

    for attempt in range(3):
        offset = part.stat().st_size if part.is_file() else 0
        if offset == total:
            break
        if offset > total:
            raise ValueError(f"Check the partial file size: {part}")
        headers = dict(HEADERS)
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                if offset and response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if content_range != f"bytes {offset}-{total - 1}/{total}":
                        raise ValueError(f"Unexpected file range from model host: {url}")
                    mode = "ab"
                elif response.status == 200:
                    offset = 0
                    mode = "wb"
                else:
                    raise ValueError(f"Unexpected response {response.status} from {url}")
                with part.open(mode) as out:
                    last_report = offset
                    while data := response.read(BLOCK):
                        offset += len(data)
                        if offset > total:
                            raise ValueError(f"File exceeds the expected size: {url}")
                        out.write(data)
                        if offset - last_report >= GIB:
                            print(f"  {offset / GIB:.1f} / {total / GIB:.1f} GiB", flush=True)
                            last_report = offset
            if offset == total:
                break
            if attempt == 2:
                raise ValueError(f"Incomplete download, rerun to resume: {part}")
        except (URLError, TimeoutError) as error:
            denied = isinstance(error, HTTPError) and error.code in (401, 403, 404)
            if denied or attempt == 2:
                raise ValueError(f"Model download interrupted: {error}. Source: {item['source']}") from error
        sleep(2 * (attempt + 1))

    if part.stat().st_size != total or sha256(part) != item["sha256"]:
        raise ValueError(f"Model checksum failed. Inspect or remove this partial file before retrying: {part}")
    if target.exists():
        raise FileExistsError(target)
    part.rename(target)


def show_runtime(expected_versions):
    for package, expected in expected_versions.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "not installed"
        if actual != expected:
            print(f"{package}: {actual}. Benchmarked with {expected}.")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", required=True, type=Path, help="ComfyUI directory containing main.py")
    parser.add_argument("--download-models", action="store_true", help="Download missing weights from the pinned upstream sites")
    parser.add_argument("--check", action="store_true", help="Check local files without downloading or writing")
    parser.add_argument("--optimizer-zip", type=Path, help="Use the pinned optimizer ZIP from disk")
    args = parser.parse_args()
    root = args.comfyui.resolve(strict=True)
    if not (root / "main.py").is_file() or not (root / "comfyui_version.py").is_file():
        raise ValueError(f"Select the ComfyUI directory with main.py: {root}")

    manifest = json.loads((ROOT / "manifest.json").read_bytes())
    payload = manifest["payload"]
    if (stage := root / "custom_nodes/moss_3c_phase/character_stage.py").is_file():
        if sha256(stage, True) == manifest["existing_moss_stage_sha256"]:
            payload = {name: item for name, item in payload.items()
                       if not name.startswith("custom_nodes/ComfyUI-H3-StageRelease/")}
    optimizer_root = inside(root, "custom_nodes/H3-Optimizations")
    optimizer = manifest["optimizer"]
    missing = missing_files(root, payload)
    missing_optimizer = missing_files(optimizer_root, optimizer["files"], normalized=True)
    show_runtime(manifest["runtime_versions"])
    if args.check:
        for name, _, _ in missing + missing_optimizer:
            print(f"Missing {name}")
        return int(bool(missing or missing_optimizer))

    install_optimizer(optimizer, missing_optimizer, args.optimizer_zip)
    models = []
    for relative, target, item in missing:
        if relative.startswith("models/"):
            models.append((target,item))
        else:
            data = inside(ROOT / "ComfyUI", relative).read_bytes()
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"Source package verification failed: {relative}")
            install_file(target, data)
    if models and not args.download_models:
        for target, item in models:
            print(f"Missing {target.name}\n  {item['source']}")
        print("See the model license in the README. Place these files in the stated folders or rerun with --download-models.")
        return 1
    for target, item in models:
        get_model(target, item)
    print(f"H3-08 and H3-08B installed in {root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(str(error))

