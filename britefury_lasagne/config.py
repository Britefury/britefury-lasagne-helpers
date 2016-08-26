import json, os

_SETTINGS_PATH = os.path.expanduser(os.path.join('~', '.britefury_lasagne_config.json'))
_DEFAULT_DATA_PATH = os.path.expanduser(os.path.join('~', '.britefury_lasagne'))


_settings__ = None
def get_settings():
    global _settings__
    if _settings__ is None:
        if os.path.exists(_SETTINGS_PATH):
            try:
                f = open(_SETTINGS_PATH, 'r')
            except:
                print('britefury_lasagne: WARNING: error trying to open settings file from {}'.format(
                    _SETTINGS_PATH))
                _settings__ = {}
            else:
                s = json.load(f)
                if isinstance(s, dict):
                    _settings__ = s
                else:
                    print('britefury_lasagne: WARNING: settings file should contain an object at the '
                          'root, not a {}'.format(type(s)))
                    _settings__ = {}
        else:
            _settings__ = {}
    return _settings__

_data_dir_path__ = None
def get_data_dir_path():
    global _data_dir_path__
    if _data_dir_path__ is None:
        _data_dir_path__ = get_settings().get('data_dir', None)
        if _data_dir_path__ is None:
            # Use the default
            _data_dir_path__ = _DEFAULT_DATA_PATH
            if os.path.exists(_data_dir_path__):
                if not os.path.isdir(_data_dir_path__):
                    raise RuntimeError('britefury_lasasgne: the DATA directory path ({}) is not a '
                                       'directory'.format(_data_dir_path__))
            else:
                os.makedirs(_data_dir_path__)
        else:
            if not os.path.exists(_data_dir_path__):
                raise RuntimeError('britefury_lasasgne: the DATA directory path ({}) specified in the '
                                   'settings file does not exist'.format(_data_dir_path__))
    return _data_dir_path__