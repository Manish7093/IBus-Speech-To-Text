import json
import logging

from pathlib import Path
from gi.repository import GObject, Gio

from sttmoonshinemodelmanagers import (
    stt_moonshine_local_model_manager,
    MOONSHINE_AVAILABLE,
)

LOG_MSG = logging.getLogger()

class STTMoonshineModel(GObject.Object):
    __gtype_name__ = "STTMoonshineModel"
    __gsignals__ = {
        "changed": (GObject.SIGNAL_RUN_FIRST, None, ()),
    }

    def __init__(self, locale_str=None):
        super().__init__()

        self._locale_str = locale_str
        self._settings = Gio.Settings.new("org.freedesktop.ibus.engine.stt")
        self._settings_id = self._settings.connect("changed::moonshine-models", self._models_changed)

        self._model_name = None
        self._model_path = None
        self._model_arch = None
        self._valid_model = False

        self._model_added_id = stt_moonshine_local_model_manager().connect("added", self._model_added_cb)
        self._model_removed_id = stt_moonshine_local_model_manager().connect("removed", self._model_removed_cb)

        model = self._get_model_from_settings()
        self._set_model(model)

    def __del__(self):
        try:
            manager = stt_moonshine_local_model_manager()
            if manager is not None:
                manager.disconnect(self._model_added_id)
                manager.disconnect(self._model_removed_id)
                if self._model_name is None and self._model_path is not None:
                    manager.unregister_custom_model_path(self._model_path)
        except (TypeError, AttributeError, NameError):
            pass

    def _get_model_from_settings(self):
        models_json_string = self._settings.get_string("moonshine-models")
        if models_json_string in (None, "None", ""):
            return None
        try:
            models_dict = json.loads(models_json_string)
            if not isinstance(models_dict, dict):
                return None
        except json.JSONDecodeError:
            return None
        return models_dict.get(self._locale_str, None)

    def _set_model(self, model):
        LOG_MSG.debug("new moonshine model (%s, current path=%s / current name=%s)",
                      model, self._model_path, self._model_name)
        if model is None:
            if self._model_name is None and self._model_path is None:
                return
            self._model_name = None
            self._model_path = None
            self._model_arch = None
            self._valid_model = False
            self.emit("changed")
            return

        model = model.rstrip("/")
        local = stt_moonshine_local_model_manager()

        if Path(model).is_absolute():
            if self._model_name is None and self._model_path == model:
                return
            self._model_name = None
            self._model_path = model
            self._model_arch = local._infer_arch_for_folder(model) if MOONSHINE_AVAILABLE else None
            local.register_custom_model_path(model, self._locale_str)
            self._valid_model = local.custom_path_available(model)
        else:
            tmp_path = local.get_best_path_for_model(model)
            if self._model_name == model and tmp_path == self._model_path:
                return
            self._model_name = model
            self._model_path = tmp_path
            self._model_arch = local.get_arch_for_model(model)
            self._valid_model = bool(tmp_path is not None)

        LOG_MSG.debug("moonshine model changed (valid=%s, path=%s, name=%s, arch=%s)",
                      self._valid_model, self._model_path, self._model_name, self._model_arch)
        self.emit("changed")

    def _models_changed(self, settings, key):
        self._set_model(self._get_model_from_settings())

    def _model_added_cb(self, manager, name, path):
        if self._model_name is not None:
            if name != self._model_name:
                return
            self._model_path = path
        elif self._model_path != path:
            return
        self._valid_model = True
        self.emit("changed")

    def _model_removed_cb(self, manager, name, path):
        if self._model_name is not None:
            if name != self._model_name:
                return
            self._model_path = manager.get_best_path_for_model(name)
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

    def get_arch(self):
        return self._model_arch

    def set_name(self, model_name):
        self._set_model(model_name)

        models_json_string = self._settings.get_string("moonshine-models")
        if models_json_string in (None, "None", ""):
            models_dict = {}
        else:
            try:
                models_dict = json.loads(models_json_string)
                if not isinstance(models_dict, dict):
                    models_dict = {}
            except json.JSONDecodeError:
                models_dict = {}

        models_dict[self._locale_str] = model_name
        models_json_string = json.dumps(models_dict)

        self._settings.disconnect(self._settings_id)
        self._settings.set_string("moonshine-models", models_json_string)
        self._settings_id = self._settings.connect("changed::moonshine-models", self._models_changed)
