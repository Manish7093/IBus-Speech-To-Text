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

import logging
import locale

from gettext import gettext as _
from babel import Locale, UnknownLocaleError

import gi

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

from gi.repository import Gtk, Gio, Gdk, GObject, Adw

from sttutils import *
from sttshortcutrow import STTShortcutRow
from sttshortcutdialog import STTShortcutDialog

from sttcurrentlocale import stt_current_locale
from sttmodelchooserdialog import STTModelChooserDialog

from sttbackenddeps import (STT_DEFAULT_BACKEND,
                            stt_backend_component,
                            stt_backend_display_name,
                            stt_backend_install_command,
                            stt_backend_is_available,
                            stt_backend_missing_dependencies,
                            stt_backend_model_manager,
                            stt_invalidate_availability_cache)

LOG_MSG=logging.getLogger()


@Gtk.Template(resource_path="/org/freedesktop/ibus/engine/stt/config/sttconfigdialog.ui")
class STTConfigDialog (Adw.Window):
    __gtype_name__="STTConfigDialog"

    main_stack    = Gtk.Template.Child()
    toast_overlay = Gtk.Template.Child()

    vosk_check    = Gtk.Template.Child()
    whisper_check = Gtk.Template.Child()
    onnxasr_check = Gtk.Template.Child()
    moonshine_check = Gtk.Template.Child()

    vosk_action_row      = Gtk.Template.Child()
    whisper_action_row   = Gtk.Template.Child()
    onnxasr_action_row   = Gtk.Template.Child()
    moonshine_action_row = Gtk.Template.Child()

    tab_stack    = Gtk.Template.Child()
    tab_switcher = Gtk.Template.Child()

    language_dropdown     = Gtk.Template.Child()
    system_language_switch = Gtk.Template.Child()

    model_info_row      = Gtk.Template.Child()
    change_model_button = Gtk.Template.Child()

    preload_model_switch   = Gtk.Template.Child()
    active_on_start_switch = Gtk.Template.Child()

    vc_whisper_warning   = Gtk.Template.Child()
    voice_commands_group = Gtk.Template.Child()

    commands_row    = Gtk.Template.Child()
    case_row        = Gtk.Template.Child()
    diacritics_row  = Gtk.Template.Child()
    punctuation_row = Gtk.Template.Child()
    custom_row      = Gtk.Template.Child()

    category_stack     = Gtk.Template.Child()
    commandslistbox    = Gtk.Template.Child()
    caselistbox        = Gtk.Template.Child()
    diacriticslistbox  = Gtk.Template.Child()
    punctuationlistbox = Gtk.Template.Child()
    customlistbox      = Gtk.Template.Child()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._valid_formatting_file_path = False
        self._valid_formatting_file = False
        self._valid_override_file = False
        self._values_dict = {}
        self._utterances_dict = {}
        self._no_model_toast = None
        self._unsupported_locale_toast = None
        self._missing_deps_toast = None
        self._model = None
        self._suppress_language_cb = False
        self._suppress_engine_cb = False
        self._engine = None

        self._settings=Gio.Settings.new("org.freedesktop.ibus.engine.stt")
        self._settings.bind("preload", self.preload_model_switch, "active", Gio.SettingsBindFlags.DEFAULT)
        self._settings.bind("active-on-start", self.active_on_start_switch, "active", Gio.SettingsBindFlags.DEFAULT)

        self._backend_subtitles = {
            backend: (row.get_subtitle() or "")
            for backend, row in self._backend_action_rows().items()
        }

        # Load current locale
        self._current_locale = stt_current_locale()
        self._locale_sig_id=self._current_locale.connect("changed", self._locale_changed_cb)
        self._override_file_changed_id=self._current_locale.connect("override-file-changed", self._override_file_changed_cb)
        self._override_file_written=False

        self._select_backend_button(self._backend())

        self._locale_list = []
        self._locale_names = Gtk.StringList()
        self._refresh_locale_dropdown()

        if self._current_locale.default_locale:
            self._suppress_language_cb = True
            self.system_language_switch.set_active(True)
            self._suppress_language_cb = False
            self.language_dropdown.set_sensitive(False)

        self._init_model()

        # This updates _valid_formatting_file and _valid_override_file
        self._load_utterances()

        self._create_engine()

        self._update_voice_commands_visibility()

        self._toast_action=Gio.SimpleAction.new("manage_model", None)
        self._install_action=Gio.SimpleAction.new("install_backend", None)
        action_group=Gio.SimpleActionGroup.new()
        action_group.insert(self._toast_action)
        action_group.insert(self._install_action)
        self.insert_action_group("toast", action_group)
        self._toast_action.connect("activate",
                                   self._manage_model_action_activated)
        self._install_action.connect("activate",
                                     self._install_action_activated)

        self._update_backend_rows()
        self._refresh_status_messages()
        self._backend_was_available = stt_backend_is_available(self._backend())
        self.connect("notify::is-active", self._window_active_cb)

    def _window_active_cb(self, *_args):
        if not self.get_property("is-active"):
            return

        stt_invalidate_availability_cache()
        available = stt_backend_is_available(self._backend())
        if available == self._backend_was_available:
            self._update_backend_rows()
            return

        self._backend_was_available = available
        self._reload_backend()

    def _backend(self):
        backend = self._settings.get_string("backend")
        if backend in ("", None):
            return STT_DEFAULT_BACKEND
        return backend

    def _backend_check_buttons(self):
        return {
            "vosk":      self.vosk_check,
            "whisper":   self.whisper_check,
            "onnxasr":   self.onnxasr_check,
            "moonshine": self.moonshine_check,
        }

    def _backend_action_rows(self):
        return {
            "vosk":      self.vosk_action_row,
            "whisper":   self.whisper_action_row,
            "onnxasr":   self.onnxasr_action_row,
            "moonshine": self.moonshine_action_row,
        }

    def _backend_for_check_button(self, button):
        for backend, check in self._backend_check_buttons().items():
            if check == button:
                return backend
        return None

    def _select_backend_button(self, backend):
        button = self._backend_check_buttons().get(backend)
        if button is None:
            button = self.vosk_check

        self._suppress_engine_cb = True
        button.set_active(True)
        self._suppress_engine_cb = False

    def _update_backend_rows(self):
        for backend, row in self._backend_action_rows().items():
            subtitle = self._backend_subtitles.get(backend, "")
            if stt_backend_is_available(backend):
                row.set_subtitle(subtitle)
            elif subtitle:
                row.set_subtitle(_("%s · Not installed") % subtitle)
            else:
                row.set_subtitle(_("Not installed"))


    def _create_engine(self):
        # Instantiate the recognition engine matching the current backend.
        # Tear down existing engine if present
        if self._engine is not None:
            try:
                self._engine.disconnect_by_func(self._engine_model_changed_cb)
            except TypeError:
                pass
            self._engine.destroy()
            self._engine = None

        backend = self._backend()
        if not stt_backend_is_available(backend):
            LOG_MSG.warning("backend %s is missing its dependencies, "
                            "no engine created", backend)
            return

        engine_class = stt_backend_component(backend, "engine")
        if engine_class is None:
            return

        try:
            self._engine = engine_class(current_locale=self._current_locale)
        except Exception as error:
            LOG_MSG.error("cannot create engine for backend %s (%s)",
                          backend, error)
            self._engine = None
            return

        self._engine.connect("model-changed", self._engine_model_changed_cb)
        self._engine.preload()
        LOG_MSG.debug("engine created (backend=%s), has_model=%s",
                      backend, self._engine.has_model())

    def _populate_locale_list(self):
        self._locale_list.clear()
        self._locale_names = Gtk.StringList()
        system_locale_str = locale.getlocale()[0]
        self._append_locale_option(system_locale_str)
        if (self._current_locale.locale != system_locale_str
                and self._current_locale.locale not in self._locale_list):
            self._append_locale_option(self._current_locale.locale)

        backend = self._backend()
        if backend == "moonshine":
            manager = stt_backend_model_manager("moonshine")
        else:
            manager = stt_backend_model_manager("vosk")

        supported = manager.supported_locales() if manager is not None else []

        _EXCLUDED = {"multilingual"}

        for loc in sorted(supported):
            if loc in _EXCLUDED:
                continue
            if loc not in self._locale_list:
                self._append_locale_option(loc)

    def _refresh_locale_dropdown(self):
        # Rebuild the language list and hand it to the dropdown.
        self._populate_locale_list()

        self._suppress_language_cb = True
        self.language_dropdown.set_model(self._locale_names)
        self._suppress_language_cb = False

        self._select_current_locale_in_dropdown()

    def _append_locale_option(self, locale_str):
        if locale_str in (None, "", "None", "multilingual"):
            return
        if locale_str in self._locale_list:
            return

        system_locale_str = locale.getlocale()[0]
        try:
            babel_locale = Locale.parse(locale_str)
            name = babel_locale.get_display_name(system_locale_str)
        except (UnknownLocaleError, ValueError):
            name = locale_str

        if locale_str == system_locale_str:
            name = _("%s (system)") % name

        self._locale_list.append(locale_str)
        self._locale_names.append(name)

    def _select_current_locale_in_dropdown(self):
        try:
            idx = self._locale_list.index(self._current_locale.locale)
        except ValueError:
            idx = 0

        self._suppress_language_cb = True
        self.language_dropdown.set_selected(idx)
        self._suppress_language_cb = False

    def _init_model(self):
        if self._model != None:
            try:
                self._model.disconnect_by_func(self._model_changed_cb)
            except TypeError:
                pass
            self._model = None

        backend = self._backend()
        locale_str = self._current_locale.locale

        if stt_backend_is_available(backend):
            model_class = stt_backend_component(backend, "model")
            if model_class is not None:
                try:
                    self._model = model_class(locale_str=locale_str)
                except Exception as error:
                    LOG_MSG.error("cannot create model for backend %s (%s)",
                                  backend, error)

        if self._model is not None:
            self._model.connect("changed", self._model_changed_cb)

        self._update_model_info()

    def _update_model_info(self):
        backend = self._backend()

        if stt_backend_missing_dependencies(backend):
            self.model_info_row.set_title(
                _("%s is not installed") % stt_backend_display_name(backend))
            self.model_info_row.set_subtitle(
                stt_backend_install_command(backend))
            self.change_model_button.set_label(_("Install…"))
            return

        if self._model is None or not self._model.available():
            self.model_info_row.set_title(_("No model downloaded"))
            self.model_info_row.set_subtitle(
                _("Download a model to get started"))
            self.change_model_button.set_label(_("Download"))
            return

        self.change_model_button.set_label(_("Change"))
        model_name = self._model.get_name()

        if model_name in (None, ""):
            model_path = self._model.get_path()
            self.model_info_row.set_title(_("Custom model"))
            self.model_info_row.set_subtitle(
                model_path if (model_path and model_path != "")
                else _("Installed manually"))
            return

        manager = stt_backend_model_manager(backend)
        desc = manager.get_model_description(model_name) if manager else None

        self.model_info_row.set_title(model_name)

        if backend == "moonshine":
            if desc is None:
                self.model_info_row.set_subtitle(_("Unknown model"))
            else:
                mtype = (desc.type or "").replace("_", " ").title()
                size  = desc.size or _("unknown size")
                quality = desc.quality or ""
                if quality:
                    self.model_info_row.set_subtitle(
                        _("%s · %s · %s") % (mtype, quality, size))
                else:
                    self.model_info_row.set_subtitle(
                        _("%s · %s") % (mtype, size))

        elif backend in ("whisper", "onnxasr"):
            if desc is None:
                self.model_info_row.set_subtitle(_("Unknown model"))
            else:
                mtype = (desc.type or "").capitalize()
                size  = desc.size or _("unknown size")
                if desc.locale == "en":
                    self.model_info_row.set_subtitle(
                        _("%s · English only · %s") % (mtype, size))
                elif desc.locale == "multilingual":
                    self.model_info_row.set_subtitle(
                        _("%s · Multilingual · %s") % (mtype, size))
                else:
                    self.model_info_row.set_subtitle(
                        _("%s · %s") % (mtype, size))
        else:
            if desc is None:
                self.model_info_row.set_subtitle(_("Unknown size"))
            else:
                size = desc.size or _("unknown size")
                if desc.is_obsolete:
                    self.model_info_row.set_subtitle(_("Obsolete · %s") % size)
                elif desc.type is not None and desc.type.startswith("big"):
                    self.model_info_row.set_subtitle(
                        _("Large model · %s") % size)
                elif desc.type is not None:
                    self.model_info_row.set_subtitle(
                        _("Lightweight · %s") % size)
                else:
                    self.model_info_row.set_subtitle(size)

    def _auto_prompt_model_download(self):
        if not stt_backend_is_available(self._backend()):
            return

        if self._model is not None and not self._model.available():
            dialog = STTModelChooserDialog(model=self._model)
            dialog.set_transient_for(self)
            dialog.present()

    def _model_changed_cb(self, model):
        self._update_model_info()

    def _copy_to_clipboard(self, text):
        clipboard = self.get_clipboard()
        if clipboard is None:
            return

        try:
            clipboard.set_content(
                Gdk.ContentProvider.new_for_value(GObject.Value(str, text)))
        except Exception as error:
            LOG_MSG.warning("cannot copy to clipboard (%s)", error)

    def _install_dialog_body(self, backend, missing):
        modules = "\n".join("    • %s" % dep.requirement for dep in missing)
        return _("The %(backend)s backend needs python modules that are not "
                 "installed on this system:\n\n"
                 "%(modules)s\n\n"
                 "Install them in a terminal with the following command, "
                 "then press “Check Again”:\n\n"
                 "    %(command)s") % {
                     "backend": stt_backend_display_name(backend),
                     "modules": modules,
                     "command": stt_backend_install_command(backend)}

    def _present_install_dialog(self, backend):
        missing = stt_backend_missing_dependencies(backend, refresh=True)
        if not missing:
            return False

        heading = _("%s Is Not Installed") % stt_backend_display_name(backend)
        body = self._install_dialog_body(backend, missing)
        command = stt_backend_install_command(backend)

        # Adw.AlertDialog needs libadwaita 1.5, fall back on Adw.MessageDialog.
        if hasattr(Adw, "AlertDialog"):
            dialog = Adw.AlertDialog(heading=heading, body=body)
            self._setup_install_dialog(dialog, backend, command)
            dialog.present(self)
        else:
            dialog = Adw.MessageDialog(transient_for=self, modal=True,
                                       heading=heading, body=body)
            self._setup_install_dialog(dialog, backend, command)
            dialog.present()

        return True

    def _setup_install_dialog(self, dialog, backend, command):
        dialog.add_response("close", _("Close"))
        dialog.add_response("copy", _("Copy Command"))
        dialog.add_response("recheck", _("Check Again"))
        dialog.set_response_appearance("recheck",
                                       Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("recheck")
        dialog.set_close_response("close")
        dialog.connect("response", self._install_dialog_response_cb,
                       backend, command)

    def _install_dialog_response_cb(self, dialog, response, backend, command):
        if response == "copy":
            self._copy_to_clipboard(command)
            self.toast_overlay.add_toast(
                Adw.Toast(title=_("Command copied to the clipboard"),
                          timeout=3))
            return

        if response != "recheck":
            return

        if not stt_backend_is_available(backend, refresh=True):
            self.toast_overlay.add_toast(
                Adw.Toast(title=_("%s is still not installed")
                          % stt_backend_display_name(backend), timeout=5))
            return

        self._update_backend_rows()
        if backend != self._backend():
            self._apply_backend(backend)
        else:
            self._reload_backend()

    def _install_action_activated(self, _action, _param):
        self._present_install_dialog(self._backend())

    def _backend_not_installed(self):
        if self._missing_deps_toast != None:
            return

        if self._no_model_toast != None:
            self._no_model_toast.dismiss()
            self._no_model_toast = None
        if self._unsupported_locale_toast != None:
            self._unsupported_locale_toast.dismiss()
            self._unsupported_locale_toast = None

        self._missing_deps_toast = Adw.Toast(
            title=_("%s is not installed")
                  % stt_backend_display_name(self._backend()),
            timeout=0,
            button_label=_("How to Install"),
            action_name="toast.install_backend")
        self._missing_deps_toast.connect("dismissed", self._toast_dismissed)
        self.toast_overlay.add_toast(self._missing_deps_toast)

    def _refresh_status_messages(self):
        if not stt_backend_is_available(self._backend()):
            self._backend_not_installed()
            return

        if self._missing_deps_toast != None:
            self._missing_deps_toast.dismiss()
            self._missing_deps_toast = None

        if (self._model is None or not self._model.available()
                or self._engine is None or not self._engine.has_model()):
            self._engine_has_no_model()
            return

        if self._no_model_toast != None:
            self._no_model_toast.dismiss()
            self._no_model_toast = None

        if self._valid_formatting_file == False:
            self._unsupported_locale()
        elif self._unsupported_locale_toast != None:
            self._unsupported_locale_toast.dismiss()
            self._unsupported_locale_toast = None

    def _update_voice_commands_visibility(self):
        is_vosk = (self._backend() == "vosk")
        self.vc_whisper_warning.set_visible(not is_vosk)
        self.voice_commands_group.set_visible(is_vosk)

        if (not is_vosk
                and self.tab_stack.get_visible_child_name() == "voice_commands"):
            self.tab_stack.set_visible_child_name("setup")

    @Gtk.Template.Callback()
    def engine_toggled_cb(self, button):
        if not button.get_active():
            return
        if getattr(self, '_suppress_engine_cb', False):
            return

        backend = self._backend_for_check_button(button)
        if backend is None:
            return

        current = self._backend()
        if backend == current:
            return

        if not stt_backend_is_available(backend, refresh=True):
            self._select_backend_button(current)
            self._update_backend_rows()
            self._present_install_dialog(backend)
            return

        self._apply_backend(backend)

    def _apply_backend(self, backend):
        self._settings.set_string("backend", backend)
        self._select_backend_button(backend)
        self._reload_backend(prompt_download=True)

    def _reload_backend(self, prompt_download=False):
        self._refresh_locale_dropdown()
        self._init_model()
        self._create_engine()
        self._update_voice_commands_visibility()
        self._empty_shortcut_page()
        self._load_utterances()
        self._update_backend_rows()
        self._refresh_status_messages()

        if (prompt_download
                and self._model is not None
                and not self._model.available()):
            self._auto_prompt_model_download()

    @Gtk.Template.Callback()
    def language_selected_cb(self, dropdown, _param):
        if self._suppress_language_cb:
            return

        idx = dropdown.get_selected()
        if idx == Gtk.INVALID_LIST_POSITION or idx >= len(self._locale_list):
            return

        locale_str = self._locale_list[idx]
        if locale_str == self._current_locale.locale:
            return

        self._current_locale.locale = locale_str

    @Gtk.Template.Callback()
    def system_language_switched_cb(self, switch, _param):
        if self._suppress_language_cb:
            return

        active = switch.get_active()
        self.language_dropdown.set_sensitive(not active)

        if active:
            self._current_locale.locale = "None"
        else:
            idx = self.language_dropdown.get_selected()
            if (idx != Gtk.INVALID_LIST_POSITION
                    and idx < len(self._locale_list)):
                self._current_locale.locale = self._locale_list[idx]

    @Gtk.Template.Callback()
    def change_model_clicked_cb(self, *_args):
        backend = self._backend()
        if not stt_backend_is_available(backend, refresh=True):
            self._present_install_dialog(backend)
            return

        if self._model is None:
            self._reload_backend()

        if self._model != None:
            dialog = STTModelChooserDialog(model=self._model)
            dialog.set_transient_for(self)
            dialog.present()

    def _locale_changed_cb(self, _current_locale):
        if self.system_language_switch.get_active() != self._current_locale.default_locale:
            self._suppress_language_cb = True
            self.system_language_switch.set_active(
                self._current_locale.default_locale)
            self._suppress_language_cb = False

        self.language_dropdown.set_sensitive(
            not self._current_locale.default_locale)
        self._select_current_locale_in_dropdown()

        if self._current_locale.locale not in self._locale_list:
            self._refresh_locale_dropdown()

        self._init_model()
        self._load_current_locale()

        if self._model is not None and not self._model.available():
            self._auto_prompt_model_download()

    def _override_file_changed_cb(self, _current_locale, deleted):
        if not deleted and not self._override_file_written:
            LOG_MSG.debug("override file changed")
            self._load_current_locale()
        self._override_file_written = False

    @Gtk.Template.Callback()
    def new_formatting_file_button_clicked_cb(self, _button):
        dialog = Gtk.FileChooserDialog(transient_for=self, title=_("Open Formatting File"), modal=True, action=Gtk.FileChooserAction.OPEN)
        dialog.add_buttons(_("Cancel"), Gtk.ResponseType.CANCEL, _("Open"), Gtk.ResponseType.ACCEPT)
        dialog.connect("response", self._open_locale_file_cb)
        dialog.set_transient_for(self)
        dialog.present()

    def _open_locale_file_cb(self, dialog, response):
        if response != Gtk.ResponseType.ACCEPT:
            dialog.destroy()
            return
        f = dialog.get_file()
        dialog.destroy()
        self._current_locale.formatting_file_path(f.get_path())

    @Gtk.Template.Callback()
    def commands_row_activated_cb(self, _row):
        self.category_stack.set_visible_child_name("commands")
        self.main_stack.set_visible_child_name("category")

    @Gtk.Template.Callback()
    def case_row_activated_cb(self, _row):
        self.category_stack.set_visible_child_name("case")
        self.main_stack.set_visible_child_name("category")

    @Gtk.Template.Callback()
    def diacritics_row_activated_cb(self, _row):
        self.category_stack.set_visible_child_name("diacritics")
        self.main_stack.set_visible_child_name("category")

    @Gtk.Template.Callback()
    def punctuation_row_activated_cb(self, _row):
        self.category_stack.set_visible_child_name("punctuation")
        self.main_stack.set_visible_child_name("category")

    @Gtk.Template.Callback()
    def custom_row_activated_cb(self, _row):
        self.category_stack.set_visible_child_name("custom")
        self.main_stack.set_visible_child_name("category")

    @Gtk.Template.Callback()
    def back_button_clicked_cb(self, _button):
        self.main_stack.set_visible_child_name("main")

    def _empty_shortcut_page(self):
        self._valid_formatting_file_path = False
        self._valid_formatting_file = False
        self._valid_override_file = False

        for row in self._values_dict.values():
            listbox = row.pref_group
            listbox.remove(row)

        self.commands_row.set_visible(False)
        self.case_row.set_visible(False)
        self.diacritics_row.set_visible(False)
        self.punctuation_row.set_visible(False)

        self._values_dict = {}
        self._utterances_dict = {}

    def _load_current_locale(self):
        self._empty_shortcut_page()
        self._load_utterances()
        self._refresh_status_messages()

    def _apply_change(self):
        LOG_MSG.debug("override file being written")
        self._override_file_written=True

        json_data = {
            "commands":    [],
            "case":        [],
            "diacritics":  [],
            "punctuation": [],
            "custom":      [],
        }

        write_changes=False
        for row in self._values_dict.values():
            value=row.get_json_data()
            if value == None:
                continue
            write_changes=True
            if row.pref_group == self.commandslistbox:
                json_data["commands"].append(value)
            elif row.pref_group == self.caselistbox:
                json_data["case"].append(value)
            elif row.pref_group == self.diacriticslistbox:
                json_data["diacritics"].append(value)
            elif row.pref_group == self.punctuationlistbox:
                json_data["punctuation"].append(value)
            elif row.pref_group == self.customlistbox:
                json_data["custom"].append(value)

        if write_changes == True:
            self._current_locale.overriding=json_data

    def shortcut_row_reset_cb(self, row):
        # After a row is reset remove extra utterances from global dictionary
        for utterance in row._extra_utterances:
            self._utterances_dict.pop(utterance, None)
        self._apply_change()

    def shortcut_row_deleted_cb(self, row):
        self._values_dict.pop(row.value, None)
        for utterance in row.utterances:
            self._utterances_dict.pop(utterance, None)
        for utterance in row._extra_utterances:
            self._utterances_dict.pop(utterance, None)
        parent = row.get_parent()
        parent.remove(row)
        self._apply_change()

    def shortcut_row_activated_cb(self, row):
        self._present_shortcut_dialog(row)

    def _present_shortcut_dialog(self, row):
        if self._engine is None:
            LOG_MSG.warning("no engine available, cannot edit shortcuts")
            return

        dialog = STTShortcutDialog(
            row=row, engine=self._engine, transient_for=self)
        dialog.connect("response", self._shortcut_dialog_response_cb)
        dialog.present()

    @Gtk.Template.Callback()
    def new_shortcut_clicked_cb(self, _button):
        self._present_shortcut_dialog(None)

    def _shortcut_dialog_response_cb(self, dialog, response):
        if response == Gtk.ResponseType.APPLY:
            (added, removed) = dialog.apply_to_row()
            for u in added:
                self._utterances_dict[u] = True
            for u in removed:
                self._utterances_dict.pop(u, None)
            self._apply_change()
        elif response == Gtk.ResponseType.OK:
            # Addition
            row = dialog.get_new_row()
            row.pref_group=self.customlistbox
            self.customlistbox.add(row)
            row.connect("activated", self.shortcut_row_activated_cb)
            row.connect("delete", self.shortcut_row_deleted_cb)
            row.connect("reset", self.shortcut_row_reset_cb)

            # It can't be a diacritic sign as the shortcut was created
            self._values_dict[row.value]=row

            # Only _extra_utterances can be added
            for u in row._extra_utterances:
                self._utterances_dict[u] = True
            self._apply_change()
        dialog.destroy()

    def _load_section(self, json_data, section, listbox, cat_row):
        item_list=json_data.get(section)
        if item_list  in (None,[]):
            if cat_row != None:
                cat_row.set_visible(False)
            return

        if cat_row != None:
            cat_row.set_visible(True)

        for item in item_list:
            value=item.get("value")
            # It may happen with diacritics that it's a list
            # The [0] is the non combining unicode (ie U+005E for circumflex)
            # and [1] is the combining unicode character (ie U+0302 for circumflex)

            utterances=item.get("utterances")
            description=item.get("description")

            # In case there is only one
            if isinstance(utterances,str):
                utterances = [utterances]

            # Each occurrence has to be unique
            for utterance in utterances[:]:
                if self._utterances_dict.get(utterance, False):
                    LOG_MSG.error("utterance already exists (%s)", utterance)
                    utterances.remove(utterance)
                    continue
                self._utterances_dict[utterance] = True

            key = value[0] if isinstance(value, list) else value
            row = self._values_dict.get(key)

            if row == None:
                if utterances == []:
                    continue
                row=STTShortcutRow(value=value,
                                   utterances=utterances,
                                   description=description,
                                   editable=False,
                                   pref_group=listbox)
                listbox.add(row)
                row.connect("activated", self.shortcut_row_activated_cb)
                row.connect("reset", self.shortcut_row_reset_cb)
                self._values_dict[key] = row
            else:
                row.utterances  = list(set(row.utterances) | set(utterances))
                row.description = description

    def _load_section_override(self, item_list, listbox):
        if item_list == None:
            return

        for item in item_list:
            value=item.get("value")
            # It may happen with diacritics that it's a list
            # The [0] is the non combining unicode (ie U+005E for circumflex)
            # and [1] is the combining unicode character (ie U+0302 for circumflex)

            utterances=item.get("utterances")
            description=item.get("description")
            if utterances not in (None,[]):
                # In case there is only one
                if isinstance(utterances, str):
                    utterances = [utterances]

                # Each occurrence has to be unique
                for utterance in utterances[:]:
                    if self._utterances_dict.get(utterance, False):
                        LOG_MSG.error("utterance already exists (%s)", utterance)
                        utterances.remove(utterance)
                        continue

                    self._utterances_dict[utterance] = True

            key = value[0] if isinstance(value, list) else value
            row = self._values_dict.get(key)

            if row != None:
                if description != None:
                    row.description = description

                if utterances not in (None,[]):
                    row.add_extra_utterances(utterances)
            elif utterances not in (None,[]):
                row=STTShortcutRow(value=value,
                                   extra_utterances=utterances,
                                   description=description,
                                   editable=True,
                                   pref_group=listbox)
                listbox.add(row)
                row.connect("delete", self.shortcut_row_deleted_cb)
                row.connect("reset", self.shortcut_row_reset_cb)
                row.connect("activated", self.shortcut_row_activated_cb)
                self._values_dict[key]=row

    def _load_formatting_file(self):
        # Load custom formatting file as well. Note: it overrides existing keys
        LOG_MSG.debug("loading formatting file")
        json_data = self._current_locale.formatting
        if json_data is None:
            return

        self._load_section(json_data, "commands",    self.commandslistbox,    self.commands_row)
        self._load_section(json_data, "case",        self.caselistbox,        self.case_row)
        self._load_section(json_data, "diacritics",  self.diacriticslistbox,  self.diacritics_row)
        self._load_section(json_data, "punctuation", self.punctuationlistbox, self.punctuation_row)
        self._load_section(json_data, "custom",      self.customlistbox,      None)

        self._valid_formatting_file = True

    def _load_overriding_file(self):
        LOG_MSG.debug("loading overriding file")
        json_data=self._current_locale.overriding
        if json_data == None:
            return

        # Now add overrides
        self._load_section_override(json_data.get("commands"), self.commandslistbox)
        self._load_section_override(json_data.get("case"), self.caselistbox)
        self._load_section_override(json_data.get("diacritics"), self.diacriticslistbox)
        self._load_section_override(json_data.get("punctuation"), self.punctuationlistbox)
        self._load_section_override(json_data.get("custom"), self.customlistbox)

        self._valid_override_file=True

    def _load_utterances(self):
        self._load_formatting_file()
        if self._valid_formatting_file == False:
            self._empty_shortcut_page()

        self._load_overriding_file()

    def _manage_model_action_activated(self, _action, _param):
        self._auto_prompt_model_download()

    def _toast_dismissed(self, toast):
        if toast == self._missing_deps_toast:
            self._missing_deps_toast=None
        elif toast == self._no_model_toast:
            self._no_model_toast=None

            # Display the other message if needed
            if self._valid_formatting_file == False:
                self._unsupported_locale()
        else:
            self._unsupported_locale_toast=None

    def _unsupported_locale(self):
        # Careful: we can have no formatting file but an overriding one !!
        if self._missing_deps_toast != None:
            return

        if self._no_model_toast != None:
            return

        # Formatting files only exist for vosk; other engines have no files to find
        if self._backend() != "vosk":
            if self._unsupported_locale_toast is not None:
                self._unsupported_locale_toast.dismiss()
                self._unsupported_locale_toast = None
            return

        if self._unsupported_locale_toast != None:
            return

        if self._valid_formatting_file_path:
            msg = _("The formatting file for your locale has an invalid format.")
        else:
            msg = _("No formatting file found for your locale. "
                    "You can add one manually.")

        self._unsupported_locale_toast = Adw.Toast(title=msg, timeout=0)
        self._unsupported_locale_toast.connect("dismissed", self._toast_dismissed)
        self.toast_overlay.add_toast(self._unsupported_locale_toast)

    def _engine_has_no_model(self):
        if self._missing_deps_toast != None:
            return

        if self._no_model_toast != None:
            return

        if self._unsupported_locale_toast != None:
            self._unsupported_locale_toast.dismiss()
            self._unsupported_locale_toast = None

        self._no_model_toast=Adw.Toast(title=_("No model available for current locale"), timeout=0, button_label=_("Download"), action_name="toast.manage_model")
        self._no_model_toast.connect("dismissed", self._toast_dismissed)
        self.toast_overlay.add_toast(self._no_model_toast)

    def _engine_model_changed_cb(self, engine):
        if engine.has_model() == False:
            self._engine_has_no_model()
        elif self._no_model_toast != None:
            self._no_model_toast.dismiss()
            self._no_model_toast = None
