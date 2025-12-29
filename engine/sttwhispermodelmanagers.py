import os
import json
import logging
from re import search
from pathlib import Path
import urllib.request
from enum import Enum
import tempfile
import shutil
import uuid
import threading

from gi.repository import GObject, Gio, GLib

LOG_MSG = logging.getLogger()

# Whisper model directories
MODEL_DIRS = [
    os.getenv('WHISPER_MODEL_PATH'),
    Path.home() / '.cache/whisper',
    Path('/usr/share/whisper'),
    Path('/usr/local/share/whisper')
]

# Hugging Face model repository
MODEL_PRE_URL = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/'
WHISPER_MODELS = {
    'tiny': 'ggml-tiny.bin',
    'tiny.en': 'ggml-tiny.en.bin',
    'base': 'ggml-base.bin',
    'base.en': 'ggml-base.en.bin',
    'small': 'ggml-small.bin',
    'small.en': 'ggml-small.en.bin',
    'medium': 'ggml-medium.bin',
    'medium.en': 'ggml-medium.en.bin',
    'large-v1': 'ggml-large-v1.bin',
    'large-v2': 'ggml-large-v2.bin',
    'large-v3': 'ggml-large-v3.bin'
}

DOWNLOADED_MODEL_SUFFIX = ".downloaded_model_tmp"

def _helper_locale_normalize(locale_str):
    lang = locale_str[0:2].lower()
    if len(locale_str) < 5:
        return lang

    lang2 = locale_str[3:5]
    return lang + "_" + lang2.upper()

class STTDownloadState(float, Enum):
    STOPPED = -1.0
    UNKNOWN_PROGRESS = -0.5
    UNPACKING = -0.6
    ONGOING = 0.0

class STTWhisperModelDescription(GObject.Object):
    __gtype_name__ = "STTWhisperModelDescription"

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

        self._operation = None
        self.download_progress = STTDownloadState.STOPPED

    def _download_finished(self):
        if self._operation.is_cancelled():
            self._operation = None

    def _download_model_thread(self, download_link, destination, status):
        with urllib.request.urlopen(download_link) as response:
            length_str = response.getheader('content-length')
            blocksize = 4096
            if length_str:
                length = int(length_str)
                blocksize = max(blocksize, length // 20)
            else:
                length = 0

            destination.parent.mkdir(parents=True, exist_ok=True)
            copy_id = uuid.uuid4()
            tmp_dst = Path(str(destination) + str(copy_id) + DOWNLOADED_MODEL_SUFFIX)

            try:
                with open(tmp_dst, 'wb') as tmp_file:
                    size = 0
                    while True:
                        if status.is_cancelled():
                            tmp_file.close()
                            if tmp_dst.exists():
                                tmp_dst.unlink()
                            return

                        buffer = response.read(blocksize)
                        if buffer is None or len(buffer) == 0:
                            break

                        tmp_file.write(buffer)
                        size += len(buffer)
                        if length != 0:
                            self.download_progress = size / length
                        else:
                            self.download_progress = STTDownloadState.UNKNOWN_PROGRESS

                os.rename(tmp_dst, destination)

                if status.is_cancelled():
                    if destination.exists():
                        destination.unlink()

            except Exception as e:
                LOG_MSG.error("Download error: %s", e)
                if tmp_dst.exists():
                    tmp_dst.unlink()

        self.download_progress = STTDownloadState.STOPPED
        GLib.idle_add(self._download_finished) 
        
    def stop_downloading(self):
        if self._operation is not None:
            self._operation.cancel()

    def start_downloading(self):
        if self._operation is not None:
            return

        LOG_MSG.debug("start downloading model (%s)", self.url)

        self.download_progress = STTDownloadState.ONGOING
        self._operation = Gio.Cancellable()

        download_thread = threading.Thread(
            target=self._download_model_thread,
            args=(self.url, Path(MODEL_DIRS[1], self.name), self._operation)
        )
        download_thread.start()

    def get_best_path_for_model(self):
        if self.paths in [None, []]:
            return None

        return self.paths[0]

    def delete_paths(self):
        if self.custom is True:
            return

        for path in self.paths:
            if Path(path).parent == MODEL_DIRS[1] and self.url is not None:
                try:
                    Path(path).unlink()
                except Exception as e:
                    LOG_MSG.error("Failed to delete %s: %s", path, e)

        self._operation = None
        self.download_progress = STTDownloadState.STOPPED
        self.paths = []

class STTWhisperLocalModelManager(GObject.Object):
    __gtype_name__ = "STTWhisperLocalModelManager"

    __gsignals__ = {
        "added": (GObject.SIGNAL_RUN_FIRST, None, (str, str,)),
        "removed": (GObject.SIGNAL_RUN_FIRST, None, (str, str,)),
    }

    def __init__(self):
        super().__init__()
        self._monitors = []
        self._models_dict = {}
        self._locales_dict = {}
        self._model_paths_dict = {}
        self._get_available_local_models()
        self._custom_paths = {}

    def _add_model_description_to_locale(self, model_desc):
        if model_desc.locale is None:
            return

        models_list = self._locales_dict.get(model_desc.locale, None)
        if models_list is None:
            self._locales_dict[model_desc.locale] = [model_desc]
        else:
            models_list.append(model_desc)

    def _new_model_available(self, model_path):

        if str(model_path).endswith(DOWNLOADED_MODEL_SUFFIX):
            LOG_MSG.debug("model path is a temporary file (%s)", model_path)
            return None

        if not model_path.is_file():
            LOG_MSG.debug("model path is not a file (%s)", model_path)
            return None

        if not model_path.name.endswith('.bin'):
            LOG_MSG.debug("model path is not a .bin file (%s)", model_path)
            return None

        if not os.access(model_path, os.R_OK):
            LOG_MSG.debug("access rights are wrong (%s)", model_path)
            return None

        if self.path_available(str(model_path)):
            LOG_MSG.debug("model file already in list (%s)", model_path)
            return None

        locale_str = None
        model_type = None
        model_name = model_path.name

        if model_name.startswith("ggml-"):
            parts = model_name.replace("ggml-", "").split(".")
            model_type = parts[0]

            if ".en" in model_name:
                locale_str = "en"
                model_type += ".en"
            else:
                locale_str = "multilingual"
        else:
            LOG_MSG.debug("non-standard name format (%s)", model_path)

        if model_path.parent not in MODEL_DIRS:
            model_desc = STTWhisperModelDescription()
            model_desc.paths = [str(model_path)]
            model_desc.name = model_name
            model_desc.custom = True
            model_desc.locale = locale_str
            model_desc.type = model_type

            self._models_dict[str(model_path)] = model_desc
            self._model_paths_dict[str(model_path)] = model_desc

            LOG_MSG.debug("custom model file is valid (%s)", model_path)
            return model_desc

        model_desc = self._models_dict.get(model_name, None)
        if model_desc is None:
            model_desc = STTWhisperModelDescription()
            model_desc.paths = [str(model_path)]
            model_desc.locale = locale_str
            model_desc.type = model_type
            model_desc.name = model_name

            self._add_model_description_to_locale(model_desc)
            self._models_dict[model_desc.name] = model_desc
            self._model_paths_dict[str(model_path)] = model_desc

            LOG_MSG.debug("model file is valid (%s) - name not known yet", model_path)
            self.emit("added", model_name, str(model_path))
            return model_desc

        model_desc.paths.append(str(model_path))
        model_desc.paths.sort(key=lambda element: MODEL_DIRS.index(Path(element).parent))

        LOG_MSG.debug("model file is valid (%s) - name already known", model_path)
        self.emit("added", model_name, str(model_path))
        return model_desc

    def _remove_model_description(self, model_path):
        model_desc = self._model_paths_dict.pop(model_path, None)
        if model_desc is None:
            return

        LOG_MSG.debug("model file removed (%s)", model_path)

        model_desc.paths.remove(model_path)
        if not any(model_desc.paths):
            models_list = self._locales_dict.get(model_desc.locale, [])
            if model_desc in models_list:
                models_list.remove(model_desc)
            if not any(models_list):
                self._locales_dict.pop(model_desc.locale, None)

            key = model_desc.name if model_desc.custom is False else model_path
            self._models_dict.pop(key, None)

        model_name = model_desc.name if model_desc.custom is False else None
        self.emit("removed", model_name, model_path)

    def _model_file_changed_cb(self, monitor, file, other_file, event_type):
        LOG_MSG.debug("a file changed (source = %s) %s %s", self, file.get_path(), event_type)

        if file.get_path() in [str(d) for d in MODEL_DIRS if d]:
            LOG_MSG.debug("change does not concern a child of a top directory. Ignoring.")
            return

        LOG_MSG.info("a model file changed (%s) (event=%s)", file.get_path(), event_type)
        if event_type == Gio.FileMonitorEvent.CHANGES_DONE_HINT:
            if file.get_path().endswith(DOWNLOADED_MODEL_SUFFIX):
                LOG_MSG.debug("temporary file ignored (%s)", file.get_path())
                return

            self._new_model_available(Path(file.get_path()))
        elif event_type == Gio.FileMonitorEvent.DELETED:
            self._remove_model_description(file.get_path())

    def _get_available_local_models(self):
        for directory in MODEL_DIRS:
            LOG_MSG.debug("scanning %s for models", directory)

            if directory is None:
                continue

            monitor = Gio.File.new_for_path(str(directory)).monitor(Gio.FileMonitorFlags.NONE, None)
            monitor.connect("changed", self._model_file_changed_cb)
            self._monitors.append(monitor)

            directory_path = Path(directory)
            if not directory_path.is_dir():
                continue

            for child in directory_path.iterdir():
                LOG_MSG.debug("scanning file (%s)", str(child))
                self._new_model_available(child)

    def path_available(self, model_path):
        return model_path in self._model_paths_dict

    def get_models_for_locale(self, locale_str):
        models = self._locales_dict.get(locale_str, []).copy()
        multilingual = self._locales_dict.get("multilingual", [])
        models.extend(multilingual)
        return models

    def get_best_path_for_model(self, model_name):
        if model_name is None:
            return None

        model = self._models_dict.get(model_name, None)
        if model is None:
            return None

        if model.paths in [None, []]:
            return None

        return model.paths[0]

    def get_model_description(self, model_name):
        return self._models_dict.get(model_name, None)

    def get_supported_locales(self):
        return list(self._locales_dict.keys())

    def _custom_model_file_changed_cb(self, monitor, file, other_file, event_type):
        LOG_MSG.info("a custom model file changed (%s) (event=%s)", file.get_path(), event_type)
        if event_type == Gio.FileMonitorEvent.CHANGES_DONE_HINT:
            self._new_model_available(Path(file.get_path()))
        elif event_type == Gio.FileMonitorEvent.DELETED:
            model = self._model_paths_dict.get(file.get_path(), None)
            if model is None:
                return

            LOG_MSG.debug("custom model file removed (%s)", file.get_path())
            self._model_paths_dict.pop(file.get_path(), None)
            self.emit("removed", None, file.get_path())

    def register_custom_model_path(self, model_path_str, locale_str):
        if Path(model_path_str).parent in MODEL_DIRS:
            LOG_MSG.debug("registered a path in default directories (%s)", model_path_str)
            return

        monitor = self._custom_paths.get(model_path_str, None)
        if monitor is not None:
            monitor.refcount += 1
            LOG_MSG.debug("custom path already registered (%s). Increasing refcount (%i).",
                         model_path_str, monitor.refcount)
            return

        monitor = Gio.File.new_for_path(model_path_str).monitor_file(Gio.FileMonitorFlags.NONE, None)
        monitor.connect("changed", self._custom_model_file_changed_cb)
        self._custom_paths[model_path_str] = monitor
        monitor.refcount = 1

        model_desc = self._new_model_available(Path(model_path_str))
        if model_desc:
            model_desc.locale = locale_str
            self._add_model_description_to_locale(model_desc)
            self.emit("added", None, model_path_str.rstrip("/"))

    def unregister_custom_model_path(self, model_path_str):
        monitor = self._custom_paths.get(model_path_str, None)
        if monitor is None:
            LOG_MSG.debug("trying to unregister a path not in custom model paths (%s)", model_path_str)
            return

        if monitor.refcount != 1:
            LOG_MSG.debug("refcount of custom path not 0 yet (%s)", model_path_str)
            monitor.refcount -= 1
            return

        self._custom_paths.pop(model_path_str, None)
        self._remove_model_description(model_path_str)

_GLOBAL_LOCAL_MANAGER = None

def stt_whisper_local_model_manager():
    global _GLOBAL_LOCAL_MANAGER
    if _GLOBAL_LOCAL_MANAGER is None:
        _GLOBAL_LOCAL_MANAGER = STTWhisperLocalModelManager()
    return _GLOBAL_LOCAL_MANAGER

class STTWhisperOnlineModelManager(GObject.Object):
    __gtype_name__ = "STTWhisperOnlineModelManager"
    __gsignals__ = {
        "added": (GObject.SIGNAL_RUN_FIRST, None, (object,)),
        "changed": (GObject.SIGNAL_RUN_FIRST, None, (object,)),
        "removed": (GObject.SIGNAL_RUN_FIRST, None, (object,)),
    }

    def __init__(self):
        super().__init__()

        self._locales_dict = {}
        self._online_models = {}

        local_manager = stt_whisper_local_model_manager()
        local_manager.connect("added", self._model_path_added_cb)
        local_manager.connect("removed", self._model_path_removed_cb)
        self._populate_with_whisper_models()

    def _populate_with_whisper_models(self):
        model_sizes = {
            'tiny': '75 MB',
            'tiny.en': '75 MB',
            'base': '142 MB',
            'base.en': '142 MB',
            'small': '466 MB',
            'small.en': '466 MB',
            'medium': '1.5 GB',
            'medium.en': '1.5 GB',
            'large-v1': '2.9 GB',
            'large-v2': '2.9 GB',
            'large-v3': '2.9 GB'
        }

        for model_name, filename in WHISPER_MODELS.items():
            model_desc = STTWhisperModelDescription()
            model_desc.name = filename
            model_desc.url = MODEL_PRE_URL + filename
            model_desc.size = model_sizes.get(model_name, 'Unknown')

            if model_name.endswith('.en'):
                model_desc.locale = 'en'
                model_desc.type = model_name.replace('.en', '')
            else:
                model_desc.locale = 'multilingual'
                model_desc.type = model_name

            LOG_MSG.debug("adding online model (%s)", model_desc.name)

            local_desc = stt_whisper_local_model_manager().get_model_description(model_desc.name)
            if local_desc is not None:
                model_desc.paths = local_desc.paths

            if model_desc.name in self._online_models:
                existing = self._online_models[model_desc.name]
                if not existing.paths and model_desc.paths:
                    existing.paths = model_desc.paths
                continue

            self._online_models[model_desc.name] = model_desc
            self._add_model_description_to_locale(model_desc)

        for locale in stt_whisper_local_model_manager().get_supported_locales():
            model_list = stt_whisper_local_model_manager().get_models_for_locale(locale)
            LOG_MSG.debug("adding local models for locale (%s)", locale)
            for model_desc in model_list:
                key = model_desc.name if model_desc.custom is False else model_desc.paths[0]
                if key in self._online_models:
                    continue

                if model_desc.custom is False:
                    bin_key = key if key.endswith('.bin') else key + '.bin'
                    if bin_key in self._online_models or key.replace('.bin', '') in self._online_models:
                        continue

                LOG_MSG.debug("adding local model to online dict (%s)", key)
                self._online_models[key] = model_desc
                self._add_model_description_to_locale(model_desc)

    def _add_model_description_to_locale(self, model_desc):
        locale_models = self._locales_dict.get(model_desc.locale, None)

        if locale_models is None:
            self._locales_dict[model_desc.locale] = [model_desc]
        else:
            locale_models.append(model_desc)

    def _model_path_added_cb(self, manager, model_name, model_path):
        if model_name is not None:
            online_model_desc = self._online_models.get(model_name, None)
            local_model_desc = manager.get_model_description(model_name)
        else:
            online_model_desc = self._online_models.get(model_path, None)
            local_model_desc = manager.get_model_description(model_path)

        if online_model_desc is not None:
            if online_model_desc.paths in [None, []]:
                online_model_desc.paths = local_model_desc.paths

            self.emit("changed", online_model_desc)
            return

        key = local_model_desc.name if local_model_desc.custom is False else local_model_desc.paths[0]
        self._online_models[key] = local_model_desc
        self._add_model_description_to_locale(local_model_desc)
        self.emit("added", local_model_desc)

    def _remove_model_description_from_locale(self, model_desc):
        locale_models = self._locales_dict.get(model_desc.locale, None)
        if locale_models and model_desc in locale_models:
            locale_models.remove(model_desc)
        if not any(locale_models):
            self._locales_dict.pop(model_desc.locale, None)

    def _model_path_removed_cb(self, manager, model_name, model_path):
        if model_name is None:
            online_model_desc = self._online_models.pop(model_path, None)
            if online_model_desc:
                self._remove_model_description_from_locale(online_model_desc)
                self.emit("removed", online_model_desc)
            return

        online_model_desc = self._online_models.get(model_name, None)
        if online_model_desc is None:
            return

        if any(online_model_desc.paths):
            self.emit("changed", online_model_desc)
            return

        if online_model_desc.url is not None:
            self.emit("changed", online_model_desc)
            return

        self._online_models.pop(model_name, None)
        self._remove_model_description_from_locale(online_model_desc)
        self.emit("removed", online_model_desc)

    def get_model_description(self, model_name):
        return self._online_models.get(model_name, None)

    def get_models_for_locale(self, locale_str):
        models = self._locales_dict.get(locale_str, []).copy()
        if locale_str != 'multilingual':
            multilingual = self._locales_dict.get('multilingual', [])
            models.extend(multilingual)
        return models

    def supported_locales(self):
        return list(self._locales_dict.keys())

_GLOBAL_ONLINE_MANAGER = None

def stt_whisper_online_model_manager():
    global _GLOBAL_ONLINE_MANAGER
    if _GLOBAL_ONLINE_MANAGER is None:
        _GLOBAL_ONLINE_MANAGER = STTWhisperOnlineModelManager()
    return _GLOBAL_ONLINE_MANAGER
