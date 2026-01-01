import json
import logging

from pathlib import Path
from gi.repository import GObject, Gio
from sttwhispermodelmanagers import stt_whisper_local_model_manager

LOG_MSG = logging.getLogger()

class STTWhisperModel(GObject.Object):
    __gtype_name__ = "STTWhisperModel"
    __gsignals__ = {
        "changed": (GObject.SIGNAL_RUN_FIRST, None, ()),
    }

    def __init__(self, locale_str=None):
        super().__init__()

        self._locale_str = locale_str
        self._settings = Gio.Settings.new("org.freedesktop.ibus.engine.stt")
        self._settings_id = self._settings.connect("changed::whisper-models", self._models_changed)

        self._model_name = None
        self._model_path = None
        self._valid_model = False

        model = self._get_model_from_settings()
        self._set_model(model)

        self._model_path_added_id = stt_whisper_local_model_manager().connect("added", self._model_added_cb)
        self._model_path_removed_id = stt_whisper_local_model_manager().connect("removed", self._model_removed_cb)

    def __del__(self):
        stt_whisper_local_model_manager().disconnect(self._model_path_added_id)
        stt_whisper_local_model_manager().disconnect(self._model_path_removed_id)
        if self._model_name is None and self._model_path is not None:
            stt_whisper_local_model_manager().unregister_custom_model_path(self._model_path)

    def _get_model_from_settings(self):
        models_json_string = self._settings.get_string("whisper-models")
        if models_json_string in (None, "None", ""):
            return None

        models_dict = json.loads(models_json_string)
        return models_dict.get(self._locale_str, None)

    def _set_model(self, model):
        LOG_MSG.debug("new model (%s, current path=%s / current name=%s)",
                     model, self._model_path, self._model_name)
        if model is None:
            if self._model_name is None and self._model_path is None:
                return

            self._model_name = None
            self._model_path = None
            self._valid_model = False

            self.emit("changed")
            return

        model = model.rstrip("/")
        model_name = self._model_name
        model_path = self._model_path

        if Path(model).is_absolute() is True:
            if self._model_name is None and self._model_path == model:
                return

            self._model_name = None
            self._model_path = model
            stt_whisper_local_model_manager().register_custom_model_path(model, self._locale_str)
            self._valid_model = stt_whisper_local_model_manager().path_available(model)
        else:
            tmp_model_path = stt_whisper_local_model_manager().get_best_path_for_model(model)
            if self._model_name == model and tmp_model_path == model_path:
                return

            self._model_name = model
            self._model_path = tmp_model_path
            self._valid_model = bool(tmp_model_path is not None)

        if model_path not in [self._model_path, None] and model_name is None:
            stt_whisper_local_model_manager().unregister_custom_model_path(model_path)

        LOG_MSG.debug("model changed (valid=%i, current path=%s - current name=%s)",
                     self._valid_model, self._model_path, self._model_name)
        self.emit("changed")

    def _models_changed(self, settings, key):
        model = self._get_model_from_settings()
        self._set_model(model)

    def _model_added_cb(self, manager, name, path):
        if self._model_name is not None:
            if name != self._model_name:
                return

            model_path = stt_whisper_local_model_manager().get_best_path_for_model(name)
            if self._model_path == model_path:
                return

            self._model_path = model_path
        elif self._model_path != path:
            return

        self._valid_model = True
        self.emit("changed")

    def _model_removed_cb(self, manager, name, path):
        if self._model_name is not None:
            if name != self._model_name:
                return

            if self._model_path != path:
                return

            self._model_path = stt_whisper_local_model_manager().get_best_path_for_model(name)
            self._valid_model = bool(self._model_path is not None)
        elif self._model_path == path:
            self._valid_model = False
        else:
            return

        self.emit("changed")

    def available(self):
        return self._valid_model

    def get_locale(self):
        return self._locale_str

    def get_name(self):
        return self._model_name

    def get_path(self):
        return self._model_path

    def set_name(self, model_name):
        self._set_model(model_name)

        models_json_string = self._settings.get_string("whisper-models")
        if models_json_string in (None, "None", ""):
            models_dict = {}
        else:
            models_dict = json.loads(models_json_string)

        models_dict[self._locale_str] = model_name
        models_json_string = json.dumps(models_dict)

        self._settings.disconnect(self._settings_id)
        self._settings.set_string("whisper-models", models_json_string)
        self._settings_id = self._settings.connect("changed::whisper-models", self._models_changed)
