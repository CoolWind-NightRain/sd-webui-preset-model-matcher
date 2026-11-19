"""Forge Neo hook that filters the Checkpoint dropdown by the UI Preset.

No WebUI source file is modified.  Forge creates the preset and checkpoint
dropdowns and wires ``ui_forge_preset.change`` to ``main_entry.on_preset_change``;
this module replaces those two module-level callables before the UI is built:

  * ``main_entry.refresh_models`` gains a filtering step for its checkpoint
    list (covers startup, the refresh button and the root ``load`` event).
  * ``main_entry.on_preset_change`` keeps Forge's 22 updates but replaces the
    first one (the checkpoint dropdown) with filtered choices and a value that
    is guaranteed to exist in those choices.

The originals are restored on ``on_script_unloaded``.  The replacement is done
in ``on_before_ui``: ``forge_main_entry`` binds the callbacks later, and
``modules/ui.py`` calls ``main_entry.refresh_models()`` during ``create_ui``,
so both paths see the wrapper.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_core():
    """Use the script-loader's instance when possible, else load it by path."""
    try:
        from modules import script_loading  # type: ignore

        for path, module in script_loading.loaded_scripts.items():
            if Path(path).name == "preset_model_matcher.py":
                return module
    except Exception:
        pass

    try:
        import preset_model_matcher as core  # type: ignore

        return core
    except Exception:
        pass

    path = Path(__file__).with_name("preset_model_matcher.py")
    spec = importlib.util.spec_from_file_location("sd_webui_preset_model_matcher_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = _load_core()

_INSTALLED = False
_ORIGINALS: dict[str, Any] = {}


def _report(message: str, exc: BaseException | None = None) -> None:
    prefix = "[Preset Model Matcher]"
    try:
        from modules import errors  # type: ignore

        if exc is None:
            print(f"{prefix} {message}")
        else:
            errors.report(f"{prefix} {message}: {exc}", exc_info=True)
    except Exception:
        print(f"{prefix} {message}" + (f": {exc}" if exc is not None else ""))


def _checkpoint_infos() -> list[Any]:
    try:
        from modules import sd_models  # type: ignore

        return list(sd_models.checkpoints_list.values())
    except Exception:
        return []


def _checkpoint_choices() -> list[str]:
    try:
        from modules import shared, shared_items  # type: ignore

        return list(shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short))
    except Exception:
        return []


def _filter(choices: list[str], preset: str | None) -> list[str]:
    return core.filter_checkpoint_choices(choices, preset, _checkpoint_infos())


def _install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    try:
        from modules_forge import main_entry  # type: ignore
    except Exception as exc:
        _report("cannot import modules_forge.main_entry", exc)
        return

    current_refresh = main_entry.refresh_models
    current_preset_change = main_entry.on_preset_change

    if getattr(current_refresh, "_pmm_wrapped", False) or getattr(current_preset_change, "_pmm_wrapped", False):
        # A previous import of this module installed the hooks; keep them.
        _INSTALLED = True
        return

    def refresh_models():
        checkpoints, vaes = current_refresh()
        try:
            checkpoints = _filter(checkpoints, core.current_preset())
        except Exception as exc:
            _report("checkpoint filtering failed during refresh", exc)
        return checkpoints, vaes

    def on_preset_change(preset):
        updates = current_preset_change(preset)
        try:
            if updates:
                choices = _filter(_checkpoint_choices(), preset)
                updates[0] = core.make_checkpoint_update(updates[0], choices, preset)
        except Exception as exc:
            _report(f"checkpoint filtering failed for preset {preset!r}", exc)
        return updates

    refresh_models._pmm_wrapped = True  # type: ignore[attr-defined]
    refresh_models._pmm_original = current_refresh  # type: ignore[attr-defined]
    on_preset_change._pmm_wrapped = True  # type: ignore[attr-defined]
    on_preset_change._pmm_original = current_preset_change  # type: ignore[attr-defined]

    main_entry.refresh_models = refresh_models
    main_entry.on_preset_change = on_preset_change

    _ORIGINALS["module"] = main_entry
    _ORIGINALS["refresh_models"] = current_refresh
    _ORIGINALS["on_preset_change"] = current_preset_change
    _INSTALLED = True
    _report("installed checkpoint filter hook")


def _restore() -> None:
    global _INSTALLED
    try:
        from modules_forge import main_entry  # type: ignore
    except Exception:
        _INSTALLED = False
        return
    for name in ("refresh_models", "on_preset_change"):
        current = getattr(main_entry, name, None)
        original = getattr(current, "_pmm_original", None)
        if getattr(current, "_pmm_wrapped", False) and original is not None:
            setattr(main_entry, name, original)
    _ORIGINALS.clear()
    _INSTALLED = False


def _settings() -> None:
    try:
        from modules import shared  # type: ignore

        section = ("preset_model_matcher", "Preset Model Matcher")
        options = (
            (core.SETTING_ENABLED, True, "Filter the Checkpoint dropdown by the current UI Preset"),
            (core.SETTING_SHOW_UNKNOWN, False, "Also show checkpoints whose architecture cannot be detected"),
            (
                core.SETTING_FALLBACK_ALL,
                True,
                "If no checkpoint matches the preset, show the unfiltered list instead of an empty one",
            ),
        )
        for key, default, label in options:
            if key not in shared.opts.data_labels:
                shared.opts.add_option(key, shared.OptionInfo(default, label, section=section))
    except Exception as exc:
        _report("cannot register settings", exc)


def _on_app_started(*_args) -> None:
    try:
        preset = core.current_preset()
        count = len(_checkpoint_infos())
        print(f"[Preset Model Matcher] ready: preset={preset!r}, checkpoints={count}")
    except Exception:
        pass


try:
    from modules import script_callbacks  # type: ignore
except Exception:
    script_callbacks = None

if script_callbacks is not None:
    script_callbacks.on_ui_settings(_settings)
    script_callbacks.on_before_ui(_install)
    script_callbacks.on_script_unloaded(_restore)
    script_callbacks.on_app_started(_on_app_started)
