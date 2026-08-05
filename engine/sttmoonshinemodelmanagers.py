import os
import logging
import importlib
import importlib.util
import threading

from pathlib import Path
from enum import Enum

from gi.repository import GObject, GLib

LOG_MSG = logging.getLogger()

_ARCH_NAMES = {
    0: "tiny",
    1: "base",
    2: "tiny-streaming",
    3: "base-streaming",
    4: "small-streaming",
    5: "medium-streaming",
}

_ARCH_SIZES = {
    "tiny":             "~50 MB",
    "tiny-streaming":   "~70 MB",
    "base":             "~120 MB",
    "base-streaming":   "~120 MB",
    "small-streaming":  "~260 MB",
    "medium-streaming": "~520 MB",
}

_ARCH_QUALITY = {
    "tiny":             "Fastest, lowest accuracy",
    "tiny-streaming":   "Very fast streaming, low accuracy",
    "base":             "Balanced",
    "base-streaming":   "Balanced streaming",
    "small-streaming":  "Good balance of speed and accuracy",
    "medium-streaming": "Most accurate, slower and resource-heavy",
}

def _arch_to_string(model_arch):
    if model_arch is None:
        return "base"
    return _ARCH_NAMES.get(int(model_arch), "base")

class STTDownloadState(float, Enum):
    STOPPED = -1.0
    UNKNOWN_PROGRESS = -0.5
    UNPACKING = -0.6
    ONGOING = 0.0

def _lang_of_locale(locale_str):
    if not locale_str:
        return None
    return locale_str[0:2].lower()

def moonshine_installed():
    #True when moonshine-voice can be imported right now
    try:
        if importlib.util.find_spec("moonshine_voice") is not None:
            return True
        importlib.invalidate_caches()
        return importlib.util.find_spec("moonshine_voice") is not None
    except (ImportError, ValueError, TypeError) as error:
        LOG_MSG.debug("cannot look up moonshine_voice (%s)", error)
        return False

def _moonshine_cache_dir():
    #Same directory as moonshine_voice.download_file.get_cache_dir(), computed without importing the package
    override = os.environ.get("MOONSHINE_VOICE_CACHE")
    if override:
        return Path(override)

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "moonshine_voice"

    return Path.home() / ".cache" / "moonshine_voice"

_CATALOG = None          # {model_name: {...}}
_CATALOG_LOCALES = None  # {lang: [model_name, ...]}

def _build_catalog():
    from moonshine_voice.download import find_model_info, supported_languages

    models = {}
    locales = {}

    for lang in supported_languages():
        for arch_value in sorted(_ARCH_NAMES):
            try:
                info = find_model_info(lang, arch_value)
            except Exception:
                # This language simply has no model for that architecture.
                continue

            arch_name = _ARCH_NAMES[arch_value]
            name = "%s-%s" % (arch_name, lang)
            models[name] = {
                "name": name,
                "lang": lang,
                "arch": arch_value,
                "arch_name": arch_name,
                "download_url": info.get("download_url", ""),
            }
            locales.setdefault(lang, []).append(name)

    return models, locales

def _catalog():
    global _CATALOG, _CATALOG_LOCALES

    if _CATALOG is not None:
        return _CATALOG, _CATALOG_LOCALES

    if not moonshine_installed():
        return {}, {}

    try:
        models, locales = _build_catalog()
    except Exception as error:
        LOG_MSG.warning("moonshine_voice model catalog unavailable (%s). "
                        "Install/upgrade with: pip install -U moonshine-voice",
                        error)
        return {}, {}

    if not models:
        LOG_MSG.warning("moonshine_voice returned an empty model catalog")
        return {}, {}

    _CATALOG = models
    _CATALOG_LOCALES = locales
    LOG_MSG.debug("moonshine catalog built (%d models, %d languages)",
                  len(models), len(locales))
    return _CATALOG, _CATALOG_LOCALES

def _model_root(entry):
    url = entry.get("download_url") or ""
    if not url:
        return None
    return Path(_moonshine_cache_dir(), url.replace("https://", ""))

def _model_present(entry):
    root = _model_root(entry)
    if root is None or not root.is_dir():
        return False

    components = None
    try:
        from moonshine_voice.download import (find_model_info,
                                              get_components_for_model_info)
        components = get_components_for_model_info(
            find_model_info(entry["lang"], entry["arch"]))
    except Exception as error:
        LOG_MSG.debug("cannot list components of %s (%s)",
                      entry["name"], error)

    if components:
        return all((root / component).is_file() for component in components)

    try:
        return any(root.iterdir())
    except OSError:
        return False

class STTMoonshineModelDescription(GObject.Object):
    __gtype_name__ = "STTMoonshineModelDescription"

    def __init__(self, init_model=None):
        super().__init__()
        self.name = init_model.name if init_model is not None else ""
        self.custom = init_model.custom if init_model is not None else False
        self.is_obsolete = False
        self.paths = init_model.paths if init_model is not None else []
        self.size = init_model.size if init_model is not None else ""
        self.type = init_model.type if init_model is not None else ""
        self.locale = init_model.locale if init_model is not None else ""
        self.url = init_model.url if init_model is not None else ""
        self.arch = init_model.arch if init_model is not None else None
        self.lang = init_model.lang if init_model is not None else None
        self.quality = init_model.quality if init_model is not None else ""

        self._operation = None
        self._downloaded_path = None
        self.download_progress = STTDownloadState.STOPPED

    def _download_finished(self):
        self._operation = None
        self.download_progress = STTDownloadState.STOPPED

        path = self._downloaded_path
        self._downloaded_path = None

        if path is None:
            entry = _catalog()[0].get(self.name)
            if entry is not None and _model_present(entry):
                path = str(_model_root(entry))

        if path is not None:
            self.paths = [path]
            stt_moonshine_local_model_manager()._notify_added(
                self.name, path, self.lang)
        return False

    def _download_thread(self, cancelled):
        try:
            from moonshine_voice import get_model_for_language
            path, _arch = get_model_for_language(self.lang, self.arch)
            self._downloaded_path = str(path)
        except Exception as e:
            LOG_MSG.error("Moonshine download failed (%s): %s", self.name, e)
        if not cancelled.is_set():
            GLib.idle_add(self._download_finished)
        else:
            self.download_progress = STTDownloadState.STOPPED

    def start_downloading(self):
        if self._operation is not None:
            return
        if not moonshine_installed():
            LOG_MSG.error("cannot download, moonshine_voice not installed")
            return

        LOG_MSG.debug("start downloading moonshine model (%s)", self.name)
        self.download_progress = STTDownloadState.UNKNOWN_PROGRESS
        self._operation = threading.Event()
        thread = threading.Thread(target=self._download_thread,
                                  args=(self._operation,), daemon=True)
        thread.start()

    def stop_downloading(self):
        if self._operation is not None:
            self._operation.set()
            self._operation = None
        self.download_progress = STTDownloadState.STOPPED

    def get_best_path_for_model(self):
        if self.paths in [None, []]:
            return None
        return self.paths[0]

    def delete_paths(self):
        if self.custom is True:
            return
        for path in list(self.paths):
            root = Path(path)
            try:
                if root.is_dir():
                    for child in root.iterdir():
                        if child.is_file():
                            child.unlink()
                    root.rmdir()
            except Exception as e:
                LOG_MSG.error("Failed to delete %s: %s", path, e)
        self._operation = None
        self.download_progress = STTDownloadState.STOPPED
        old_paths = self.paths
        self.paths = []
        for path in old_paths:
            stt_moonshine_local_model_manager()._notify_removed(self.name, path)


class STTMoonshineLocalModelManager(GObject.Object):
    __gtype_name__ = "STTMoonshineLocalModelManager"

    __gsignals__ = {
        "added": (GObject.SIGNAL_RUN_FIRST, None, (str, str,)),
        "removed": (GObject.SIGNAL_RUN_FIRST, None, (str, str,)),
    }

    def __init__(self):
        super().__init__()
        self._present = {}
        self._custom_paths = {}
        self._scanned_models = 0

    def _scan_present_models(self):
        catalog, _locales = _catalog()
        if not catalog or self._scanned_models == len(catalog):
            return

        self._scanned_models = len(catalog)
        for name, entry in catalog.items():
            if _model_present(entry):
                self._present[name] = str(_model_root(entry))
                LOG_MSG.debug("moonshine model present on disk (%s)", name)

    def _notify_added(self, model_name, path, lang=None):
        self._present[model_name] = path
        self.emit("added", model_name, path)

    def _notify_removed(self, model_name, path):
        self._present.pop(model_name, None)
        self.emit("removed", model_name, path)

    def path_available(self, model_path):
        self._scan_present_models()
        return (model_path in self._present.values()
                or model_path in self._custom_paths)

    def get_best_path_for_model(self, model_name):
        if model_name is None:
            return None
        self._scan_present_models()
        return self._present.get(model_name, None)

    def get_arch_for_model(self, model_name):
        entry = _catalog()[0].get(model_name)
        return entry["arch"] if entry is not None else None

    def get_lang_for_model(self, model_name):
        entry = _catalog()[0].get(model_name)
        return entry["lang"] if entry is not None else None

    @staticmethod
    def _infer_arch_for_folder(model_path):
        root = Path(model_path)
        name = root.name.lower()

        if (root / "streaming_config.json").is_file():
            for token, arch in (("medium", 5), ("small", 4),
                                ("base", 3), ("tiny", 2)):
                if token in name:
                    return arch
            return 4
        if "tiny" in name:
            return 0
        return 1

    def register_custom_model_path(self, model_path, locale_str):
        self._custom_paths[model_path] = locale_str

    def unregister_custom_model_path(self, model_path):
        self._custom_paths.pop(model_path, None)

    def custom_path_available(self, model_path):
        root = Path(model_path)
        return root.is_dir() and (root / "tokenizer.bin").is_file()


_GLOBAL_LOCAL_MANAGER = None

def stt_moonshine_local_model_manager():
    global _GLOBAL_LOCAL_MANAGER
    if _GLOBAL_LOCAL_MANAGER is None:
        _GLOBAL_LOCAL_MANAGER = STTMoonshineLocalModelManager()
    return _GLOBAL_LOCAL_MANAGER


class STTMoonshineOnlineModelManager(GObject.Object):
    __gtype_name__ = "STTMoonshineOnlineModelManager"
    __gsignals__ = {
        "added": (GObject.SIGNAL_RUN_FIRST, None, (object,)),
        "changed": (GObject.SIGNAL_RUN_FIRST, None, (object,)),
        "removed": (GObject.SIGNAL_RUN_FIRST, None, (object,)),
    }

    def __init__(self):
        super().__init__()
        self._models = {}
        self._locales_dict = {}

        local = stt_moonshine_local_model_manager()
        local.connect("added", self._model_path_added_cb)
        local.connect("removed", self._model_path_removed_cb)

    def _ensure_catalog(self):
        catalog, locales = _catalog()
        if not catalog or len(self._models) == len(catalog):
            return

        local = stt_moonshine_local_model_manager()
        self._models = {}
        self._locales_dict = {}

        for name, entry in catalog.items():
            arch_name = entry["arch_name"]

            desc = STTMoonshineModelDescription()
            desc.name = name
            desc.lang = entry["lang"]
            desc.locale = entry["lang"]
            desc.arch = entry["arch"]
            desc.type = arch_name
            desc.url = entry["download_url"]
            desc.size = _ARCH_SIZES.get(arch_name, "")
            desc.quality = _ARCH_QUALITY.get(arch_name, "")

            path = local.get_best_path_for_model(name)
            if path is not None:
                desc.paths = [path]

            self._models[name] = desc
            self._locales_dict.setdefault(entry["lang"], []).append(desc)

    def _model_path_added_cb(self, manager, model_name, model_path):
        desc = self._models.get(model_name, None)
        if desc is None:
            return
        if model_path not in desc.paths:
            desc.paths = [model_path]
        self.emit("changed", desc)

    def _model_path_removed_cb(self, manager, model_name, model_path):
        desc = self._models.get(model_name, None)
        if desc is None:
            return
        desc.paths = []
        self.emit("changed", desc)

    def get_model_description(self, model_name):
        self._ensure_catalog()
        return self._models.get(model_name, None)

    def get_models_for_locale(self, locale_str):
        self._ensure_catalog()
        lang = _lang_of_locale(locale_str)
        return list(self._locales_dict.get(lang, []))

    def supported_locales(self):
        self._ensure_catalog()
        return list(self._locales_dict.keys())

_GLOBAL_ONLINE_MANAGER = None

def stt_moonshine_online_model_manager():
    global _GLOBAL_ONLINE_MANAGER
    if _GLOBAL_ONLINE_MANAGER is None:
        _GLOBAL_ONLINE_MANAGER = STTMoonshineOnlineModelManager()
    return _GLOBAL_ONLINE_MANAGER
