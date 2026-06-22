import os
import logging
import threading

from pathlib import Path
from enum import Enum

from gi.repository import GObject, GLib

LOG_MSG = logging.getLogger()


try:
    from moonshine_voice import ModelArch
    from moonshine_voice.download import (
        MODEL_INFO,
        find_model_info,
        get_components_for_model_info,
        get_model_for_language,
    )
    from moonshine_voice.download_file import get_cache_dir
    MOONSHINE_AVAILABLE = True
except Exception as e:
    LOG_MSG.warning("moonshine_voice model catalog unavailable (%s). "
                    "Install/upgrade with: pip install -U moonshine-voice", e)
    MOONSHINE_AVAILABLE = False
    MODEL_INFO = {}
    ModelArch = None

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
    return {
        0: "tiny",
        1: "base",
        2: "tiny-streaming",
        3: "base-streaming",
        4: "small-streaming",
        5: "medium-streaming",
    }.get(int(model_arch), "base")


class STTDownloadState(float, Enum):
    STOPPED = -1.0
    UNKNOWN_PROGRESS = -0.5
    UNPACKING = -0.6
    ONGOING = 0.0

def _lang_of_locale(locale_str):
    if not locale_str:
        return None
    return locale_str[0:2].lower()

def _all_model_infos():
    for lang, entry in MODEL_INFO.items():
        for model in entry.get("models", []):
            yield model["model_name"], lang, model

def _expected_model_path(model_info):
    cache_dir = get_cache_dir()
    folder = model_info["download_url"].replace("https://", "")
    return Path(cache_dir, folder)

def _model_present(model_info):
    root = _expected_model_path(model_info)
    if not root.is_dir():
        return False
    try:
        components = get_components_for_model_info(model_info)
    except Exception:
        components = ["tokenizer.bin"]
    return all((root / component).is_file() for component in components)


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
        self.download_progress = STTDownloadState.STOPPED

    def _download_finished(self):
        self._operation = None
        self.download_progress = STTDownloadState.STOPPED
        info = find_model_info(self.lang, self.arch)
        if _model_present(info):
            path = str(_expected_model_path(info))
            self.paths = [path]
            stt_moonshine_local_model_manager()._notify_added(self.name, path, self.lang)
        return False

    def _download_thread(self, cancelled):
        try:
            get_model_for_language(self.lang, self.arch)
        except Exception as e:
            LOG_MSG.error("Moonshine download failed (%s): %s", self.name, e)
        if not cancelled.is_set():
            GLib.idle_add(self._download_finished)
        else:
            self.download_progress = STTDownloadState.STOPPED

    def start_downloading(self):
        if self._operation is not None:
            return
        if not MOONSHINE_AVAILABLE:
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
        self._scan_present_models()

    def _scan_present_models(self):
        if not MOONSHINE_AVAILABLE:
            return
        for model_name, lang, info in _all_model_infos():
            info = dict(info, language=lang)
            if _model_present(info):
                self._present[model_name] = str(_expected_model_path(info))
                LOG_MSG.debug("moonshine model present on disk (%s)", model_name)

    def _notify_added(self, model_name, path, lang=None):
        self._present[model_name] = path
        self.emit("added", model_name, path)

    def _notify_removed(self, model_name, path):
        self._present.pop(model_name, None)
        self.emit("removed", model_name, path)

    def path_available(self, model_path):
        return model_path in self._present.values() or model_path in self._custom_paths

    def get_best_path_for_model(self, model_name):
        if model_name is None:
            return None
        return self._present.get(model_name, None)

    def get_arch_for_model(self, model_name):
        if not MOONSHINE_AVAILABLE:
            return None
        for name, lang, info in _all_model_infos():
            if name == model_name:
                return info["model_arch"]
        return None

    def get_lang_for_model(self, model_name):
        if not MOONSHINE_AVAILABLE:
            return None
        for name, lang, info in _all_model_infos():
            if name == model_name:
                return lang
        return None

    @staticmethod
    def _infer_arch_for_folder(model_path):
        root = Path(model_path)
        if (root / "streaming_config.json").is_file():
            name = root.name.lower()
            for token, arch in (("medium", ModelArch.MEDIUM_STREAMING),
                                ("small", ModelArch.SMALL_STREAMING),
                                ("base", ModelArch.BASE_STREAMING),
                                ("tiny", ModelArch.TINY_STREAMING)):
                if token in name:
                    return arch
            return ModelArch.SMALL_STREAMING
        if "tiny" in root.name.lower():
            return ModelArch.TINY
        return ModelArch.BASE

    def register_custom_model_path(self, model_path, locale_str):
        self._custom_paths[model_path] = locale_str

    def unregister_custom_model_path(self, model_path):
        self._custom_paths.pop(model_path, None)

    def custom_path_available(self, model_path):
        return Path(model_path).is_dir() and (Path(model_path) / "tokenizer.bin").is_file()


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
        self._build_catalog()

        local = stt_moonshine_local_model_manager()
        local.connect("added", self._model_path_added_cb)
        local.connect("removed", self._model_path_removed_cb)

    def _build_catalog(self):
        if not MOONSHINE_AVAILABLE:
            return
        for model_name, lang, info in _all_model_infos():
            arch = info["model_arch"]
            arch_str = _arch_to_string(arch)

            desc = STTMoonshineModelDescription()
            desc.name = model_name
            desc.lang = lang
            desc.locale = lang
            desc.arch = arch
            desc.type = arch_str
            desc.url = info["download_url"]
            desc.size = _ARCH_SIZES.get(arch_str, "")
            desc.quality = _ARCH_QUALITY.get(arch_str, "")

            path = stt_moonshine_local_model_manager().get_best_path_for_model(model_name)
            if path is not None:
                desc.paths = [path]

            self._models[model_name] = desc
            self._locales_dict.setdefault(lang, []).append(desc)

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
        return self._models.get(model_name, None)

    def get_models_for_locale(self, locale_str):
        lang = _lang_of_locale(locale_str)
        return list(self._locales_dict.get(lang, []))

    def supported_locales(self):
        if not MOONSHINE_AVAILABLE:
            return []
        return list(self._locales_dict.keys())

_GLOBAL_ONLINE_MANAGER = None

def stt_moonshine_online_model_manager():
    global _GLOBAL_ONLINE_MANAGER
    if _GLOBAL_ONLINE_MANAGER is None:
        _GLOBAL_ONLINE_MANAGER = STTMoonshineOnlineModelManager()
    return _GLOBAL_ONLINE_MANAGER
