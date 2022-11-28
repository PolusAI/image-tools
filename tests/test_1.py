from polus.plugins import plugins, submit_plugin

BF_URL = "https://raw.githubusercontent.com/PolusAI/polus-plugins/master/regression/polus-basic-flatfield-correction-plugin/plugin.json"


class TestLoadPlugin:
    def submit(self):
        p = submit_plugin(BF_URL, refresh=True)
        assert "BasicFlatfieldCorrectionPlugin" in plugins.list
