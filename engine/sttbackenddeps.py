import importlib
import importlib.machinery
import importlib.util
import logging
import sys

LOG_MSG = logging.getLogger()

class STTBackendDependency:

    def __init__(self, module, distribution, extras=None):
        self.module = module
        self.distribution = distribution
        self.extras = tuple(extras) if extras is not None else ()

    @property
    def requirement(self):
        if not self.extras:
            return self.distribution
        return "%s[%s]" % (self.distribution, ",".join(self.extras))

    @property
    def pip_argument(self):
        if not self.extras:
            return self.distribution
        return "'%s'" % self.requirement

    def available(self):
        #True when the module can be found on sys.path right now
        try:
            spec = importlib.machinery.PathFinder.find_spec(self.module,
                                                            sys.path)
        except (ImportError, ValueError, TypeError, AttributeError) as error:
            LOG_MSG.debug("cannot look up module %s (%s)", self.module, error)
            return False

        return spec is not None

    def __repr__(self):
        return "<STTBackendDependency %s>" % self.requirement

# NOTE: keep the keys in sync with the "backend" GSettings key and with the
# check buttons of sttconfigdialog.ui.
_BACKENDS = {
    "vosk": {
        "name": "Vosk",
        "dependencies": (),
        "components": {
            "engine":  ("sttgstvosk", "STTGstVosk"),
            "model":   ("sttvoskmodel", "STTVoskModel"),
            "manager": ("sttvoskmodelmanagers", "stt_vosk_online_model_manager"),
        },
    },
    "whisper": {
        "name": "Whisper",
        "dependencies": (),
        "components": {
            "engine":  ("sttgstwhisper", "STTGstWhisper"),
            "model":   ("sttwhispermodel", "STTWhisperModel"),
            "manager": ("sttwhispermodelmanagers", "stt_whisper_online_model_manager"),
        },
    },
    "onnxasr": {
        "name": "onnx-asr",
        "dependencies": (
            STTBackendDependency("onnx_asr", "onnx-asr", ("cpu", "hub")),
        ),
        "components": {
            "engine":  ("sttgstonnxasr", "STTGstOnnxAsr"),
            "model":   ("sttonnxasrmodel", "STTOnnxAsrModel"),
            "manager": ("sttonnxasrmodelmanagers", "stt_onnxasr_online_model_manager"),
        },
    },
    "moonshine": {
        "name": "Moonshine",
        "dependencies": (
            STTBackendDependency("moonshine_voice", "moonshine-voice"),
        ),
        "components": {
            "engine":  ("sttgstmoonshine", "STTGstMoonshine"),
            "model":   ("sttmoonshinemodel", "STTMoonshineModel"),
            "manager": ("sttmoonshinemodelmanagers", "stt_moonshine_online_model_manager"),
        },
    },
}

STT_DEFAULT_BACKEND = "vosk"

# Cache the lookups: they are queried from callbacks that can run on every
# model/locale change. Dropped by stt_invalidate_availability_cache().
_availability_cache = {}

def stt_backends():
    return tuple(_BACKENDS.keys())

def stt_backend_display_name(backend):
    entry = _BACKENDS.get(backend)
    if entry is None:
        return backend
    return entry["name"]

def stt_backend_dependencies(backend):
    entry = _BACKENDS.get(backend)
    if entry is None:
        return ()
    return entry["dependencies"]

def stt_invalidate_availability_cache():
    _availability_cache.clear()
    importlib.invalidate_caches()

def stt_backend_missing_dependencies(backend, refresh=False):
    if refresh:
        stt_invalidate_availability_cache()

    missing = []
    for dependency in stt_backend_dependencies(backend):
        available = _availability_cache.get(dependency.module)
        if available is None:
            available = dependency.available()
            _availability_cache[dependency.module] = available
            LOG_MSG.debug("module %s available: %s",
                          dependency.module, available)
        if not available:
            missing.append(dependency)

    return missing

def stt_backend_is_available(backend, refresh=False):
    return not stt_backend_missing_dependencies(backend, refresh=refresh)

def stt_backend_install_command(backend):
    missing = stt_backend_missing_dependencies(backend)
    if not missing:
        return ""

    return "pip install %s" % " ".join(
        dependency.pip_argument for dependency in missing)

def stt_backend_component(backend, component):
    entry = _BACKENDS.get(backend)
    if entry is None:
        LOG_MSG.warning("unknown backend (%s)", backend)
        return None

    location = entry["components"].get(component)
    if location is None:
        return None

    module_name, attribute = location
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        LOG_MSG.warning("cannot import %s for backend %s (%s)",
                        module_name, backend, error)
        return None

    value = getattr(module, attribute, None)
    if value is None:
        LOG_MSG.warning("%s has no attribute %s", module_name, attribute)
    return value

def stt_backend_model_manager(backend):
    manager = stt_backend_component(backend, "manager")
    if manager is None:
        return None

    try:
        return manager()
    except Exception as error:
        LOG_MSG.warning("cannot create model manager for %s (%s)",
                        backend, error)
        return None
