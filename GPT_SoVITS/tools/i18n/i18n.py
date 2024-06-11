import json
import locale
import os

cwd = os.path.dirname(os.path.abspath(__file__))


def load_language_list(language):
    file_path = os.path.join(cwd, f"locale{os.path.sep}{language}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        language_list = json.load(f)
    return language_list


class I18nAuto:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[
                0
            ]  # getlocale can't identify the system's language ((None, None))
        file_path = os.path.join(cwd, f"locale{os.path.sep}{language}.json")
        if not os.path.exists(file_path):
            language = "en_US"
        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language
