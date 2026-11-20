"""Checkpoint architecture detection for Forge Neo's UI Preset filter.

This module is deliberately free of WebUI imports at import time so it can be
tested (and inspected) without starting Forge.  The only input it needs is a
checkpoint path.  It answers one question: which Forge Neo UI preset does this
checkpoint belong to?

Sources, highest priority first:
  1. ``<checkpoint>.json`` written by Forge's checkpoint metadata editor
     (``sd_arch`` field).
  2. The safetensors header: ``modelspec.architecture`` and
     ``ss_base_model_version``; then ``modelspec.title`` and
     ``ss_sd_model_name`` as a weaker hint.
  3. The file name, for the many checkpoints that carry no metadata at all
     (and for ``.ckpt`` / ``.gguf`` files, whose metadata is not read here).
  4. Civitai sidecars written by Civitai Helper / browser downloaders
     (``<checkpoint>.civitai.info`` or ``<checkpoint>.civitai.json``), but
     only when the filename also failed to identify the model.  The
     ``baseModel`` field is then mapped to Forge's preset names.

Only the JSON header of a safetensors file is read.  Tensor data is never
touched, so this is fast and memory-friendly.  Results are cached in
``storage/arch_cache.json`` keyed by the file's mtime/size and the sidecar
JSON's mtime, so editing the manual ``sd_arch`` value or replacing a Civitai
sidecar invalidates the entry.
"""

from __future__ import annotations

import json
import os
import re
import struct
import threading
import time
import sys
import types

# Forge Neo's script loader (modules/script_loading.py) does not register
# the module in sys.modules before exec_module(). Python 3.13's dataclasses
# implementation requires sys.modules[cls.__module__] to exist, otherwise
# @dataclass raises AttributeError. Register a placeholder so the decorator
# can find a namespace.
if __name__ not in sys.modules:
    sys.modules[__name__] = types.ModuleType(__name__)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Bump when detection rules change; old cache entries are discarded.
CACHE_VERSION = 5
CACHE_FILENAME = "arch_cache.json"
MAX_HEADER_BYTES = 100 * 1024 * 1024
MAX_CIVITAI_BYTES = 4 * 1024 * 1024

# Forge Neo removed SD2 and SD3, so these must never be offered under a preset.
UNSUPPORTED_MARKERS = (
    "stable-diffusion-v2",
    "stable-diffusion-v3",
    "stable-diffusion-2",
    "stable-diffusion-3",
    "sd_v2",
    "sd-v2",
    "sd-v3",
    "sd_v3",
    "sd2",
    "sd3",
    # Forge Neo has no SVD / AnimateDiff / IC-Light checkpoint presets.
    "stable-video-diffusion",
    "animatediff",
    "iclight",
    "wan2.1",
    "wan-2.1",
    "wan_2.1",
    "wan21",
)

# Metadata values (lower-cased substring match, first match wins).
# The order matters: qwen21 before qwen, klein before flux, ph3/h3i before h3.
PRESET_MAPPING: dict[str, tuple[str, ...]] = {
    "qwen21": ("qwen-image-2.1", "qwen_image_2.1", "qwen2.1", "qwen-2.1", "qwen21"),
    "qwen": ("qwen-image", "qwen_image", "qwen"),
    "klein": ("flux-2", "flux2", "flux.2", "klein"),
    "flux": ("flux-1", "flux1", "flux.1", "flux-dev", "flux-schnell", "chroma", "flux"),
    "krea": ("krea-2", "krea2", "krea_2"),
    "moss": ("moss-image", "moss_image", "moss"),
    "lumina": ("lumina-image", "lumina2", "lumina_2", "lumina"),
    "zit": ("z-image", "z_image", "zimage", "z-anime", "z_anime", "zanime"),
    "wan": ("wan-2.2", "wan2.2", "wan_2.2", "wan22", "wan"),
    "anima": ("anima", "wai-anima", "wai_anima", "waianima"),
    "ernie": ("ernie-image", "ernie_image", "ernie"),
    "pid": ("pid-1.5", "pid_1.5", "pid"),
    "xl": ("stable-diffusion-xl", "sdxl", "pony", "illustrious", "noobai", "animagine-xl", "animagine_xl"),
    "sd": ("stable-diffusion-v1", "stable-diffusion-v1-5", "sd_v1", "sd-v1", "sd1", "anything-v"),
    "ph3": ("p-h3-image", "p_h3_image", "ph3"),
    "h3i": ("minimax-h3-image", "minimax_h3_image"),
    "h3": ("minimax-h3", "minimax_h3", "minimax h3"),
}

KNOWN_PRESETS = frozenset(PRESET_MAPPING)

# File-name fallback.  Basename only (folder names are deliberately ignored so
# that a folder called "Stable Diffusion" cannot classify everything).
_FILENAME_RULES_RAW: tuple[tuple[str, str], ...] = (
    ("ph3", r"(?:p[-_ ]?h3[-_ ]?image|ph3)"),
    ("h3i", r"(?:minimax[-_ ]?h3[-_ ]?image|(?<![a-z0-9])h3i(?![a-z0-9]))"),
    ("h3", r"(?:minimax[-_ ]?h3|(?<![a-z0-9])h3(?![a-z0-9]))"),
    ("qwen21", r"(?:qwen[-_ ]?image[-_ ]?2[._-]?1|qwen[-_ ]?2[._-]?1|qwen21)"),
    ("qwen", r"qwen"),
    ("klein", r"klein|flux[-_ ]?2(?:[^0-9]|$)"),
    ("flux", r"flux|chroma"),
    ("krea", r"krea[-_ ]?2|krea2"),
    ("moss", r"moss"),
    ("lumina", r"lumina"),
    ("zit", r"z[-_ ]?image|zimage|z[-_ ]?anime"),
    ("wan", r"wan[-_ ]?2[._-]?2|wan2[._-]?2|(?<![a-z0-9])wan(?![a-z0-9])"),
    ("anima", r"(?<![a-z0-9])anima(?![a-z0-9])|wai[-_ ]?anima"),
    ("ernie", r"ernie"),
    ("pid", r"(?<![a-z])pid(?![a-z])"),
    # Explicit SD1.5 must win over the weak "pony" / "xl" hints below.
    ("sd", r"sd[-_ ]?v?1[-_ ]?5"),
    (
        "xl",
        r"sdxl|sd[_-]?xl|(?<![a-z0-9])xl|[a-z]xl(?=$|[^a-z]|lcm|animated)"
        r"|illustrious|pony|noobai|awpainting_cgillu|anything[-_ ]?xl|wai[-_ ]?illustrious|wai.*noob",
    ),
    ("sd", r"sd[-_ ]?v?1[-_ ]?5|sd[-_ ]?v1(?:[^0-9]|$)|sd1|v1[-_ ]?5|anything|awpainting_v"),
)

_FILENAME_RULES = tuple((preset, re.compile(pattern, re.IGNORECASE)) for preset, pattern in _FILENAME_RULES_RAW)
_UNSUPPORTED_FILENAME_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:"
    r"sd[-_ ]?[23](?:[^0-9]|$)|stable[-_ ]?diffusion[-_ ]?[23](?:[^0-9]|$)|"
    # Audio / video / auxiliary architectures that Forge Neo cannot load as
    # a Checkpoint, even when their names contain "sd15".
    r"acestep|ace[-_ ]?step|minimax[-_ ]?music|"
    r"stable[-_ ]?video[-_ ]?diffusion|(?<![a-z0-9])svd(?![a-z0-9])|"
    r"iclight|(?:^|[^a-z0-9])mm[-_ ]?sd|(?:^|[^a-z0-9])v3[-_ ]?sd15|"
    r"wan[-_ ]?2[-_ .]?1(?:[^0-9]|$)|wan[-_ ]?21(?![0-9])|wan[-_ ]?21[-_ ]?14b"
    r")",
    re.IGNORECASE,
)

# Civitai's ``baseModel`` values, normalized (case-folded, whitespace-collapsed,
# ``-``/``_`` replaced by spaces).  Kept explicit rather than reusing
# PRESET_MAPPING because Civitai says "SD 1.5", not "stable-diffusion-v1".
_CIVITAI_BASE_MODELS: dict[str, str] = {
    "sd1.5": "sd",
    "sd 1.5": "sd",
    "sd 1.4": "sd",
    "sd 1.5 lcm": "sd",
    "sd 1.5 hyper": "sd",
    "sd 1.5 ays": "sd",
    "sd 1.5 amd": "sd",
    "sdxl": "xl",
    "sdxl1.0": "xl",
    "sdxl 0.9": "xl",
    "sdxl 1.0": "xl",
    "sdxl 1.0 lcm": "xl",
    "sdxl hyper": "xl",
    "sdxl lightning": "xl",
    "sdxl turbo": "xl",
    "sdxl distilled": "xl",
    "illustrious": "xl",
    "illustrious xl": "xl",
    "pony": "xl",
    "pony xl": "xl",
    "noobai": "xl",
    "noobai xl": "xl",
    "animagine xl": "xl",
    "flux1 d": "flux",
    "flux1 s": "flux",
    "flux1 dev": "flux",
    "flux1 schnell": "flux",
    "flux.1 d": "flux",
    "flux.1 s": "flux",
    "flux.1 dev": "flux",
    "flux.1 schnell": "flux",
    "flux.1 krea": "flux",
    "flux.2 klein": "klein",
    "qwen image": "qwen",
    "qwen image edit": "qwen",
    "qwen2.1": "qwen21",
    "qwen 2.1": "qwen21",
    "qwen image 2.1": "qwen21",
    "lumina": "lumina",
    "lumina 2": "lumina",
    "lumina image 2.0": "lumina",
    "zimage": "zit",
    "zimageturbo": "zit",
    "zimagebase": "zit",
    "zimage turbo": "zit",
    "zimage base": "zit",
    "z image": "zit",
    "z image turbo": "zit",
    "wan 2.2": "wan",
    "wan2.2": "wan",
    "wan video 2.2": "wan",
    "wan video 2.2 t2v": "wan",
    "wan video 2.2 i2v": "wan",
    "anima": "anima",
    "ernie": "ernie",
    "ernie image": "ernie",
    "pid": "pid",
    "pid 1.5": "pid",
    "krea2": "krea",
    "krea 2": "krea",
    "moss": "moss",
    "moss image": "moss",
    "minimax h3": "h3",
    "minimax h3 image": "h3i",
    "p h3 image": "ph3",
    "ph3": "ph3",
}

# Civitai values that exist but Forge Neo cannot load.  Checked before the
# positive table so "SD 2.1" never falls through to "sd".
_CIVITAI_UNSUPPORTED_PREFIXES = (
    "sd 2",
    "sd 3",
    "stable diffusion 2",
    "stable diffusion 3",
    "wan 2.1",
    "wan video 2.1",
    "flux.2 d",
    "flux.2 dev",
) 

# Settings names, also used by ui_binding and the README.
SETTING_ENABLED = "preset_model_matcher_enabled"
SETTING_SHOW_UNKNOWN = "preset_model_matcher_show_unknown"
SETTING_FALLBACK_ALL = "preset_model_matcher_fallback_all"

_DEFAULT_SETTINGS = {
    SETTING_ENABLED: True,
    SETTING_SHOW_UNKNOWN: False,
    SETTING_FALLBACK_ALL: True,
}


@dataclass(frozen=True)
class ArchRecord:
    """One checkpoint's classification.

    ``arch`` is a Forge preset name (``sd``, ``xl``, ...), or None when the
    file could not be classified.  ``source`` is one of ``manual``,
    ``metadata``, ``filename``, ``unsupported``, ``unknown`` or ``missing``.
    """

    arch: str | None
    source: str
    evidence: str = ""

    @property
    def is_known(self) -> bool:
        return self.arch is not None


def extension_root() -> Path:
    """Return .../extensions/sd-webui-preset-model-matcher."""
    return Path(__file__).resolve().parents[1]


def default_cache_path() -> Path:
    return extension_root() / "storage" / CACHE_FILENAME


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _civitai_sidecar_paths(path: Path) -> list[Path]:
    """Return existing Civitai metadata files for ``path``.

    Supports both layouts used in the wild::

        model.civitai.info
        model.safetensors.civitai.info

    and the ``.civitai.json`` variants.  The case-insensitive fallback keeps
    this working on Linux/NAS mounts where the downloader's casing may differ.
    """
    candidates = [
        path.with_suffix(".civitai.info"),
        path.with_name(path.name + ".civitai.info"),
        path.with_suffix(".civitai.json"),
        path.with_name(path.name + ".civitai.json"),
    ]
    found: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            found.append(candidate)

    if found:
        return found
    if os.name == "nt":
        # Windows filesystems are case-insensitive, so the exact candidates
        # above already cover differently-cased sidecar names.
        return found

    wanted = {candidate.name.casefold() for candidate in candidates}
    try:
        for child in path.parent.iterdir():
            if child.is_file() and child.name.casefold() in wanted:
                found.append(child)
    except OSError:
        pass
    return found


def _sidecar_mtime(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def _fingerprint(path: Path) -> list[int]:
    """Model identity for the cache: model plus every metadata sidecar mtime."""
    stat = path.stat()
    civitai_mtimes = [_sidecar_mtime(sidecar) for sidecar in _civitai_sidecar_paths(path)]
    return [
        stat.st_mtime_ns,
        stat.st_size,
        _sidecar_mtime(_sidecar_path(path)),
        *civitai_mtimes,
    ]


def _cache_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def read_safetensors_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read only the ``__metadata__`` object from a safetensors header.

    Uses ``safetensors.safe_open(...).metadata()`` when the library is
    available, then falls back to parsing the 8-byte length + JSON header
    directly.  Neither path maps tensor data or imports torch.

    Raises ``ValueError``/``OSError``/``json.JSONDecodeError`` on a malformed
    file when the fallback is needed.
    """
    file_path = Path(path)

    try:
        from safetensors import safe_open  # type: ignore

        with safe_open(str(file_path), framework="np", device="cpu") as handle:
            metadata = handle.metadata()
            return dict(metadata) if metadata else {}
    except ImportError:
        pass
    except Exception:
        # Old/odd safetensors build, or a header the library rejects: the
        # direct parser below is the compatibility path.
        pass

    with file_path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("file shorter than 8 bytes")
        (length,) = struct.unpack("<Q", raw)
        if length <= 0:
            return {}
        if length > MAX_HEADER_BYTES:
            raise ValueError(f"header length {length} exceeds {MAX_HEADER_BYTES}")
        payload = handle.read(length)
        if len(payload) != length:
            raise ValueError("truncated safetensors header")
    header = json.loads(payload)
    metadata = header.get("__metadata__") if isinstance(header, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def _manual_arch(path: Path) -> ArchRecord | None:
    """Read Forge's per-checkpoint ``<name>.json`` user metadata."""
    sidecar = _sidecar_path(path)
    if not sidecar.is_file():
        return None
    try:
        with sidecar.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("sd_arch")
    if raw is None:
        return None
    value = str(raw).strip()
    aliases = {"sd1": "sd", "sdxl": "xl", "sdxl1.0": "xl"}
    value = aliases.get(value.casefold(), value)
    if not value or value.casefold() in {"unknown", "none", "auto"}:
        return None
    if value in PRESET_MAPPING:
        return ArchRecord(value, "manual", f"{sidecar.name}:sd_arch={raw}")
    return None


def _match_text(value: str, source: str) -> ArchRecord | None:
    normalized = value.strip().casefold()
    if not normalized:
        return None
    for marker in UNSUPPORTED_MARKERS:
        if marker in normalized:
            return ArchRecord(None, "unsupported", f"{source}={value}")
    for preset, markers in PRESET_MAPPING.items():
        for marker in markers:
            if marker in normalized:
                return ArchRecord(preset, source, f"{source}={value}")
    return None


def detect_from_metadata(metadata: Mapping[str, Any] | None) -> ArchRecord | None:
    """Map safetensors metadata to a preset, or None when it says nothing."""
    if not metadata:
        return None
    for key in ("modelspec.architecture", "ss_base_model_version"):
        value = metadata.get(key)
        if value:
            record = _match_text(str(value), f"metadata:{key}")
            if record is not None:
                return record
    for key in ("modelspec.title", "ss_sd_model_name"):
        value = metadata.get(key)
        if value:
            record = _match_text(str(value), f"metadata:{key}")
            if record is not None:
                return record
    return None


def _normalize_civitai_base_model(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return re.sub(r"[\s_-]+", " ", text).strip()


def _civitai_base_model_record(value: str, sidecar_name: str) -> ArchRecord | None:
    normalized = _normalize_civitai_base_model(value)
    if not normalized:
        return None
    for prefix in _CIVITAI_UNSUPPORTED_PREFIXES:
        if normalized.startswith(prefix):
            return ArchRecord(
                None,
                "unsupported",
                f"{sidecar_name}:baseModel={value} (unsupported architecture)",
            )
    for label, preset in sorted(
        _CIVITAI_BASE_MODELS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if normalized == label or normalized.startswith(f"{label} ") or normalized.startswith(
            f"{label}("
        ):
            return ArchRecord(preset, "civitai", f"{sidecar_name}:baseModel={value}")
    return None


def read_civitai_metadata(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read the first usable ``.civitai.info`` / ``.civitai.json`` sidecar.

    Returns the parsed API response (normally a model-version object with a
    ``baseModel`` field), or None when no non-empty sidecar exists.
    """
    for sidecar in _civitai_sidecar_paths(Path(path)):
        try:
            if sidecar.stat().st_size > MAX_CIVITAI_BYTES:
                continue
            with sidecar.open("r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data:
            return data
    return None


def detect_from_civitai(path: str | os.PathLike[str]) -> ArchRecord | None:
    """Map a Civitai Helper sidecar's ``baseModel`` to a preset.

    Civitai sidecars are only consulted for models whose safetensors header
    and filename did not decide.  A sidecar explicitly describing a LoRA or
    another non-Checkpoint resource is ignored so a mismatched download cannot
    reclassify the checkpoint.
    """
    for sidecar in _civitai_sidecar_paths(Path(path)):
        try:
            if sidecar.stat().st_size > MAX_CIVITAI_BYTES:
                continue
            with sidecar.open("r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or not data:
            continue

        model = data.get("model")
        model = model if isinstance(model, dict) else {}
        model_type = str(model.get("type") or "").strip().casefold()
        if model_type and model_type != "checkpoint":
            continue

        base_model = data.get("baseModel") or model.get("baseModel")
        if not base_model:
            continue
        record = _civitai_base_model_record(str(base_model), sidecar.name)
        if record is not None:
            return record
    return None


def detect_from_filename(path: str | os.PathLike[str]) -> ArchRecord:
    """Classify by basename; never raises."""
    name = Path(path).stem
    if _UNSUPPORTED_FILENAME_RE.search(name):
        return ArchRecord(None, "unsupported", f"filename={name} (unsupported architecture)")
    for preset, pattern in _FILENAME_RULES:
        if pattern.search(name):
            return ArchRecord(preset, "filename", f"filename={name}")
    return ArchRecord(None, "unknown", f"filename={name}")


class ArchCache:
    """Small JSON cache with atomic writes and a process-wide lock."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path) if path is not None else default_cache_path()
        self._lock = threading.RLock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._load()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def _load(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
            return
        entries = data.get("entries")
        if isinstance(entries, dict):
            self._entries = {str(k): v for k, v in entries.items() if isinstance(v, dict)}

    def get(self, path: Path, fingerprint: Sequence[int]) -> ArchRecord | None:
        key = _cache_key(path)
        with self._lock:
            entry = self._entries.get(key)
            if not entry or entry.get("fingerprint") != list(fingerprint):
                return None
            arch = entry.get("arch")
            source = entry.get("source", "unknown")
            evidence = entry.get("evidence", "")
        if arch is not None and not isinstance(arch, str):
            return None
        return ArchRecord(arch, str(source), str(evidence))

    def put(self, path: Path, fingerprint: Sequence[int], record: ArchRecord) -> None:
        key = _cache_key(path)
        with self._lock:
            self._entries[key] = {
                "fingerprint": list(fingerprint),
                "arch": record.arch,
                "source": record.source,
                "evidence": record.evidence,
                "scanned_at": int(time.time()),
            }
            self._dirty = True

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            payload = {"version": CACHE_VERSION, "entries": self._entries}
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temp = self.path.with_name(self.path.name + ".tmp")
                with temp.open("w", encoding="utf-8", newline="\n") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                    handle.write("\n")
                os.replace(temp, self.path)
                self._dirty = False
            except OSError:
                # A read-only extension folder must not break model loading.
                pass

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._dirty = True


_DEFAULT_CACHE: ArchCache | None = None
_DEFAULT_CACHE_LOCK = threading.Lock()


def get_default_cache() -> ArchCache:
    global _DEFAULT_CACHE
    with _DEFAULT_CACHE_LOCK:
        if _DEFAULT_CACHE is None:
            _DEFAULT_CACHE = ArchCache()
        return _DEFAULT_CACHE


def classify_checkpoint(
    path: str | os.PathLike[str],
    cache: ArchCache | None = None,
    *,
    refresh: bool = False,
) -> ArchRecord:
    """Classify one checkpoint, using (and filling) the on-disk cache."""
    file_path = Path(path)
    cache = cache or get_default_cache()
    try:
        fingerprint = _fingerprint(file_path)
    except OSError as exc:
        return ArchRecord(None, "missing", str(exc))

    if not refresh:
        cached = cache.get(file_path, fingerprint)
        if cached is not None:
            return cached

    record = _classify_uncached(file_path)
    cache.put(file_path, fingerprint, record)
    return record


def _classify_uncached(path: Path) -> ArchRecord:
    manual = _manual_arch(path)
    if manual is not None:
        return manual

    metadata: dict[str, Any] | None = None
    if path.suffix.casefold() in {".safetensors", ".sft"}:
        try:
            metadata = read_safetensors_metadata(path)
        except (OSError, ValueError, json.JSONDecodeError):
            metadata = None

    if metadata is not None:
        from_metadata = detect_from_metadata(metadata)
        if from_metadata is not None:
            return from_metadata

    from_filename = detect_from_filename(path)
    if from_filename.source != "unknown":
        return from_filename

    from_civitai = detect_from_civitai(path)
    if from_civitai is not None:
        return from_civitai
    return from_filename


def classify_checkpoints(
    paths: Iterable[str | os.PathLike[str]],
    cache: ArchCache | None = None,
    *,
    refresh: bool = False,
) -> dict[str, ArchRecord]:
    """Classify many files and persist the cache once."""
    cache = cache or get_default_cache()
    result: dict[str, ArchRecord] = {}
    try:
        for path in paths:
            result[str(path)] = classify_checkpoint(path, cache, refresh=refresh)
    finally:
        cache.flush()
    return result


def _setting(name: str, default: Any) -> Any:
    try:
        from modules import shared  # type: ignore
    except Exception:
        return default
    return getattr(shared.opts, name, default)


def current_preset() -> str | None:
    value = _setting("forge_preset", None)
    return str(value) if value else None


def _info_value(info: Any, key: str) -> Any:
    if isinstance(info, Mapping):
        return info.get(key)
    return getattr(info, key, None)


def _index_infos(infos: Iterable[Any] | None) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for info in infos or ():
        filename = _info_value(info, "filename")
        for key in ("title", "name", "short_title"):
            value = _info_value(info, key)
            if value:
                index.setdefault(str(value), info)
        if filename:
            index.setdefault(os.path.basename(str(filename)), info)
    return index


def filter_checkpoint_choices(
    choices: Sequence[str],
    preset: str | None,
    infos: Iterable[Any] | None = None,
    *,
    show_unknown: bool | None = None,
    fallback_all: bool | None = None,
    cache: ArchCache | None = None,
) -> list[str]:
    """Return the subset of ``choices`` compatible with ``preset``.

    ``infos`` is any iterable of objects/dicts with ``filename`` plus at least
    one of ``name``/``title``/``short_title`` (normally
    ``modules.sd_models.checkpoints_list.values()``).  Choices that cannot be
    resolved to a known checkpoint are left alone rather than hidden.
    """
    choices = [str(choice) for choice in (choices or ())]
    if not choices or not preset:
        return choices

    preset = str(preset)
    # A preset added by another extension and not in our table must keep all
    # models: filtering it would hide everything on a guess.
    if preset not in KNOWN_PRESETS:
        return choices

    if not bool(_setting(SETTING_ENABLED, True)):
        return choices
    if show_unknown is None:
        show_unknown = bool(_setting(SETTING_SHOW_UNKNOWN, False))
    if fallback_all is None:
        fallback_all = bool(_setting(SETTING_FALLBACK_ALL, True))

    index = _index_infos(infos)
    cache = cache or get_default_cache()
    filtered: list[str] = []
    try:
        for choice in choices:
            info = index.get(choice) or index.get(os.path.normpath(choice))
            if info is None:
                filtered.append(choice)
                continue
            filename = _info_value(info, "filename")
            if not filename:
                filtered.append(choice)
                continue
            record = classify_checkpoint(str(filename), cache)
            if record.arch == preset:
                filtered.append(choice)
            elif record.arch is None and record.source == "unknown" and show_unknown:
                filtered.append(choice)
    finally:
        cache.flush()

    if not filtered and fallback_all:
        print(
            f"[Preset Model Matcher] no checkpoint matched preset {preset!r}; "
            "showing the unfiltered list (see settings)."
        )
        return choices
    return filtered


def choose_checkpoint_value(update_value: Any, choices: Sequence[str], preset: str | None) -> str | None:
    """Pick a dropdown value that is guaranteed to be one of ``choices``."""
    candidates = [update_value, _setting("sd_model_checkpoint", None)]
    if preset:
        candidates.insert(0, _setting(f"forge_checkpoint_{preset}", None))
    choice_set = set(choices)
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate)
        if text and text in choice_set:
            return text
    return choices[0] if choices else None


def make_checkpoint_update(update: Any, choices: Sequence[str], preset: str | None) -> Any:
    """Rebuild Forge's first preset update with filtered choices and a valid value."""
    if isinstance(update, Mapping) and update.get("__type__") == "skip":
        return update
    value = update.get("value") if isinstance(update, Mapping) else None
    import gradio as gr  # local: the core stays importable without gradio

    if isinstance(choices, (list, tuple)) and choices:
        value = choose_checkpoint_value(value, choices, preset)
    else:
        value = None
    return gr.update(choices=list(choices), value=value)


def describe_checkpoint(path: str | os.PathLike[str], cache: ArchCache | None = None) -> dict[str, Any]:
    """Diagnostic helper used by Harness_Tool's self-test and bug reports."""
    record = classify_checkpoint(path, cache)
    return {
        "path": str(path),
        "arch": record.arch,
        "source": record.source,
        "evidence": record.evidence,
    }
