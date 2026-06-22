# vim:set et sts=4 sw=4:
#
# ibus-stt - Speech To Text engine for IBus
# Copyright (C) 2022 Philippe Rouquier <bonfire-app@wanadoo.fr>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import logging
import threading
from pathlib import Path
from enum import Enum

from gi.repository import GObject, Gio, GLib

LOG_MSG = logging.getLogger()

# HuggingFace cache directory
HF_CACHE_DIR = Path(os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))) / 'hub'

# onnx-asr model catalog: name -> (locale, size, repo_id)
ONNXASR_MODELS = {
    "nemo-parakeet-tdt-0.6b-v3": ("multilingual", "~300 MB", "istupakov/parakeet-tdt-0.6b-v3-onnx"),
    "nemo-parakeet-tdt-0.6b-v2": ("en", "~300 MB", "istupakov/parakeet-tdt-0.6b-v2-onnx"),
    "nemo-parakeet-ctc-0.6b": ("en", "~250 MB", "istupakov/parakeet-ctc-0.6b-onnx"),
    "nemo-parakeet-rnnt-0.6b": ("en", "~250 MB", "istupakov/parakeet-rnnt-0.6b-onnx"),
    "nemo-canary-1b-v2": ("multilingual", "~600 MB", "istupakov/canary-1b-v2-onnx"),
    "nemo-fastconformer-ru-ctc": ("ru", "~150 MB", "istupakov/nemo-fastconformer-ru-onnx"),
    "nemo-fastconformer-ru-rnnt": ("ru", "~150 MB", "istupakov/nemo-fastconformer-ru-onnx"),
    "gigaam-v2-ctc": ("ru", "~200 MB", "istupakov/gigaam-v2-onnx"),
    "gigaam-v2-rnnt": ("ru", "~200 MB", "istupakov/gigaam-v2-onnx"),
    "gigaam-v3-ctc": ("ru", "~200 MB", "istupakov/gigaam-v3-onnx"),
    "gigaam-v3-rnnt": ("ru", "~200 MB", "istupakov/gigaam-v3-onnx"),
    "gigaam-v3-e2e-ctc": ("ru", "~200 MB", "istupakov/gigaam-v3-onnx"),
    "gigaam-v3-e2e-rnnt": ("ru", "~200 MB", "istupakov/gigaam-v3-onnx"),
    "whisper-base": ("multilingual", "~150 MB", "istupakov/whisper-base-onnx"),
}


def _repo_cache_path(repo_id):
    """Get the HuggingFace cache path for a repo ID."""
    return HF_CACHE_DIR / ("models--" + repo_id.replace("/", "--"))


def _is_repo_cached(repo_id):
    """Check if a HuggingFace repo is cached locally."""
    snapshots = _repo_cache_path(repo_id) / "snapshots"
    if not snapshots.is_dir():
        return False
    try:
        return any(snapshots.iterdir())
    except OSError:
        return False


class STTDownloadState(float, Enum):
    STOPPED = -1.0
    UNKNOWN_PROGRESS = -0.5
    UNPACKING = -0.6
    ONGOING = 0.0


class STTOnnxAsrModelDescription(GObject.Object):
    __gtype_name__ = "STTOnnxAsrModelDescription"

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
        self.repo_id = init_model.repo_id if init_model is not None else ""

        self._operation = None
        self.download_progress = STTDownloadState.STOPPED

    def _download_finished(self):
        if self._operation is not None and self._operation.is_cancelled():
            self._operation = None

    def _download_model_thread(self, model_name, status):
        try:
            import onnx_asr

            self.download_progress = STTDownloadState.UNKNOWN_PROGRESS

            if status.is_cancelled():
                self.download_progress = STTDownloadState.STOPPED
                GLib.idle_add(self._download_finished)
                return

            # onnx_asr.load_model downloads from HuggingFace if not cached
            onnx_asr.load_model(model_name)

            if status.is_cancelled():
                self.download_progress = STTDownloadState.STOPPED
                GLib.idle_add(self._download_finished)
                return

            # Update paths after successful download
            if self.repo_id:
                cache_path = _repo_cache_path(self.repo_id)
                if cache_path.is_dir():
                    self.paths = [str(cache_path)]

        except Exception as e:
            LOG_MSG.error("Download error for %s: %s", model_name, e)

        self.download_progress = STTDownloadState.STOPPED
        GLib.idle_add(self._download_finished)

    def stop_downloading(self):
        if self._operation is not None:
            self._operation.cancel()

    def start_downloading(self):
        if self._operation is not None:
            return

        LOG_MSG.debug("start downloading onnx-asr model (%s)", self.name)

        self.download_progress = STTDownloadState.ONGOING
        self._operation = Gio.Cancellable()

        download_thread = threading.Thread(
            target=self._download_model_thread,
            args=(self.name, self._operation)
        )
        download_thread.start()

    def get_best_path_for_model(self):
        if self.paths in [None, []]:
            return None
        return self.paths[0]

    def delete_paths(self):
        if self.custom is True:
            return

        if not self.repo_id:
            return

        import shutil
        cache_path = _repo_cache_path(self.repo_id)
        if cache_path.is_dir():
            try:
                shutil.rmtree(cache_path)
            except Exception as e:
                LOG_MSG.error("Failed to delete %s: %s", cache_path, e)

        self._operation = None
        self.download_progress = STTDownloadState.STOPPED
        self.paths = []


class STTOnnxAsrLocalModelManager(GObject.Object):
    __gtype_name__ = "STTOnnxAsrLocalModelManager"

    __gsignals__ = {
        "added": (GObject.SIGNAL_RUN_FIRST, None, (str, str,)),
        "removed": (GObject.SIGNAL_RUN_FIRST, None, (str, str,)),
    }

    def __init__(self):
        super().__init__()
        self._monitors = []
        self._models_dict = {}
        self._locales_dict = {}
        self._custom_models = {}
        self._scan_cached_models()
        self._setup_monitor()

    def _add_model_description_to_locale(self, model_desc):
        if model_desc.locale is None:
            return

        models_list = self._locales_dict.get(model_desc.locale, None)
        if models_list is None:
            self._locales_dict[model_desc.locale] = [model_desc]
        else:
            models_list.append(model_desc)

    def _scan_cached_models(self):
        """Scan HuggingFace cache for known onnx-asr models."""
        # Scan HuggingFace cache for downloaded models
        for model_name, (locale, size, repo_id) in ONNXASR_MODELS.items():
            if model_name in self._models_dict:
                continue

            if _is_repo_cached(repo_id):
                cache_path = str(_repo_cache_path(repo_id))
                model_desc = STTOnnxAsrModelDescription()
                model_desc.name = model_name
                model_desc.locale = locale
                model_desc.size = size
                model_desc.repo_id = repo_id
                model_desc.paths = [cache_path]
                model_desc.type = model_name.split("-")[0] if "-" in model_name else model_name

                self._models_dict[model_name] = model_desc
                self._add_model_description_to_locale(model_desc)
                LOG_MSG.debug("found cached onnx-asr model: %s at %s", model_name, cache_path)

    def _setup_monitor(self):
        """Monitor the HuggingFace cache directory for changes."""
        try:
            HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except OSError:
            LOG_MSG.warning("Cannot create HuggingFace cache directory: %s", HF_CACHE_DIR)
            return

        monitor = Gio.File.new_for_path(str(HF_CACHE_DIR)).monitor_directory(
            Gio.FileMonitorFlags.NONE, None)
        monitor.connect("changed", self._cache_dir_changed_cb)
        self._monitors.append(monitor)

    def _cache_dir_changed_cb(self, monitor, file, other_file, event_type):
        if event_type not in (Gio.FileMonitorEvent.CREATED,
                              Gio.FileMonitorEvent.DELETED,
                              Gio.FileMonitorEvent.CHANGES_DONE_HINT):
            return

        dir_name = file.get_basename()

        for model_name, (locale, size, repo_id) in ONNXASR_MODELS.items():
            expected_dir = "models--" + repo_id.replace("/", "--")
            if dir_name != expected_dir:
                continue

            if event_type in (Gio.FileMonitorEvent.CREATED,
                              Gio.FileMonitorEvent.CHANGES_DONE_HINT):
                if model_name not in self._models_dict and _is_repo_cached(repo_id):
                    cache_path = str(_repo_cache_path(repo_id))
                    model_desc = STTOnnxAsrModelDescription()
                    model_desc.name = model_name
                    model_desc.locale = locale
                    model_desc.size = size
                    model_desc.repo_id = repo_id
                    model_desc.paths = [cache_path]
                    model_desc.type = model_name.split("-")[0] if "-" in model_name else model_name

                    self._models_dict[model_name] = model_desc
                    self._add_model_description_to_locale(model_desc)
                    self.emit("added", model_name, cache_path)
                    LOG_MSG.info("onnx-asr model now available: %s", model_name)

            elif event_type == Gio.FileMonitorEvent.DELETED:
                model_desc = self._models_dict.pop(model_name, None)
                if model_desc is not None:
                    models_list = self._locales_dict.get(model_desc.locale, [])
                    if model_desc in models_list:
                        models_list.remove(model_desc)
                    if not any(models_list):
                        self._locales_dict.pop(model_desc.locale, None)

                    self.emit("removed", model_name, str(_repo_cache_path(repo_id)))
                    LOG_MSG.info("onnx-asr model removed: %s", model_name)

    def path_available(self, model_name):
        return model_name in self._models_dict

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

        # For onnx-asr, the "path" is the model name identifier
        return model_name

    def get_model_description(self, model_name):
        return self._models_dict.get(model_name, None)

    def get_supported_locales(self):
        return list(self._locales_dict.keys())

    def register_custom_model_path(self, model_name, locale_str):
        """Register a custom HuggingFace model repo ID."""
        if model_name in ONNXASR_MODELS:
            LOG_MSG.debug("registered a catalog model name (%s)", model_name)
            return

        if model_name in self._custom_models:
            self._custom_models[model_name] += 1
            LOG_MSG.debug("custom model already registered (%s). Increasing refcount (%i).",
                         model_name, self._custom_models[model_name])
            return

        self._custom_models[model_name] = 1

        # For custom models, repo_id is the model name itself (a HF repo path)
        repo_id = model_name
        model_desc = STTOnnxAsrModelDescription()
        model_desc.name = model_name
        model_desc.custom = True
        model_desc.locale = locale_str
        model_desc.repo_id = repo_id

        if _is_repo_cached(repo_id):
            model_desc.paths = [str(_repo_cache_path(repo_id))]

        self._models_dict[model_name] = model_desc
        self._add_model_description_to_locale(model_desc)
        self.emit("added", None, model_name)

    def unregister_custom_model_path(self, model_name):
        if model_name not in self._custom_models:
            LOG_MSG.debug("trying to unregister unknown custom model (%s)", model_name)
            return

        if self._custom_models[model_name] != 1:
            LOG_MSG.debug("refcount of custom model not 0 yet (%s)", model_name)
            self._custom_models[model_name] -= 1
            return

        self._custom_models.pop(model_name, None)

        model_desc = self._models_dict.pop(model_name, None)
        if model_desc is not None:
            models_list = self._locales_dict.get(model_desc.locale, [])
            if model_desc in models_list:
                models_list.remove(model_desc)
            if not any(models_list):
                self._locales_dict.pop(model_desc.locale, None)

            self.emit("removed", None, model_name)


_GLOBAL_LOCAL_MANAGER = None

def stt_onnxasr_local_model_manager():
    global _GLOBAL_LOCAL_MANAGER
    if _GLOBAL_LOCAL_MANAGER is None:
        _GLOBAL_LOCAL_MANAGER = STTOnnxAsrLocalModelManager()
    return _GLOBAL_LOCAL_MANAGER


class STTOnnxAsrOnlineModelManager(GObject.Object):
    __gtype_name__ = "STTOnnxAsrOnlineModelManager"
    __gsignals__ = {
        "added": (GObject.SIGNAL_RUN_FIRST, None, (object,)),
        "changed": (GObject.SIGNAL_RUN_FIRST, None, (object,)),
        "removed": (GObject.SIGNAL_RUN_FIRST, None, (object,)),
    }

    def __init__(self):
        super().__init__()

        self._locales_dict = {}
        self._online_models = {}

        local_manager = stt_onnxasr_local_model_manager()
        local_manager.connect("added", self._model_added_cb)
        local_manager.connect("removed", self._model_removed_cb)
        self._populate_with_onnxasr_models()

    def _populate_with_onnxasr_models(self):
        for model_name, (locale, size, repo_id) in ONNXASR_MODELS.items():
            model_desc = STTOnnxAsrModelDescription()
            model_desc.name = model_name
            model_desc.locale = locale
            model_desc.size = size
            model_desc.repo_id = repo_id
            model_desc.type = model_name.split("-")[0] if "-" in model_name else model_name
            # STTModelRow only shows the download/delete button when url is non-empty,
            # so expose the HuggingFace repo page as the URL marker.
            model_desc.url = "https://huggingface.co/" + repo_id

            LOG_MSG.debug("adding online onnx-asr model (%s)", model_name)

            local_desc = stt_onnxasr_local_model_manager().get_model_description(model_name)
            if local_desc is not None:
                model_desc.paths = local_desc.paths

            self._online_models[model_name] = model_desc
            self._add_model_description_to_locale(model_desc)

        # Add locally-available custom models not in catalog
        for locale in stt_onnxasr_local_model_manager().get_supported_locales():
            model_list = stt_onnxasr_local_model_manager().get_models_for_locale(locale)
            for model_desc in model_list:
                key = model_desc.name
                if key in self._online_models:
                    continue

                LOG_MSG.debug("adding local custom model to online dict (%s)", key)
                self._online_models[key] = model_desc
                self._add_model_description_to_locale(model_desc)

    def _add_model_description_to_locale(self, model_desc):
        locale_models = self._locales_dict.get(model_desc.locale, None)

        if locale_models is None:
            self._locales_dict[model_desc.locale] = [model_desc]
        else:
            locale_models.append(model_desc)

    def _model_added_cb(self, manager, model_name, model_path):
        if model_name is not None:
            online_model_desc = self._online_models.get(model_name, None)
            local_model_desc = manager.get_model_description(model_name)
        else:
            online_model_desc = self._online_models.get(model_path, None)
            local_model_desc = manager.get_model_description(model_path)

        if online_model_desc is not None:
            if online_model_desc.paths in [None, []]:
                online_model_desc.paths = local_model_desc.paths if local_model_desc else []

            self.emit("changed", online_model_desc)
            return

        if local_model_desc is not None:
            key = local_model_desc.name
            self._online_models[key] = local_model_desc
            self._add_model_description_to_locale(local_model_desc)
            self.emit("added", local_model_desc)

    def _remove_model_description_from_locale(self, model_desc):
        locale_models = self._locales_dict.get(model_desc.locale, None)
        if locale_models and model_desc in locale_models:
            locale_models.remove(model_desc)
        if locale_models is not None and not any(locale_models):
            self._locales_dict.pop(model_desc.locale, None)

    def _model_removed_cb(self, manager, model_name, model_path):
        if model_name is None:
            online_model_desc = self._online_models.pop(model_path, None)
            if online_model_desc:
                self._remove_model_description_from_locale(online_model_desc)
                self.emit("removed", online_model_desc)
            return

        online_model_desc = self._online_models.get(model_name, None)
        if online_model_desc is None:
            return

        online_model_desc.paths = []

        # Catalog models stay listed even when not downloaded
        if model_name in ONNXASR_MODELS:
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

def stt_onnxasr_online_model_manager():
    global _GLOBAL_ONLINE_MANAGER
    if _GLOBAL_ONLINE_MANAGER is None:
        _GLOBAL_ONLINE_MANAGER = STTOnnxAsrOnlineModelManager()
    return _GLOBAL_ONLINE_MANAGER
