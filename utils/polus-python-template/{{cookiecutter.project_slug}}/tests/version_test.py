import unittest, json
from pathlib import Path
import urllib.request as request

class VersionTest(unittest.TestCase):
    """Verify the version matches in VERSION and plugin.json """
    
    version_path = Path(__file__).parent.parent.joinpath("VERSION")
    json_path = Path(__file__).parent.parent.joinpath("plugin.json")
    url = 'https://hub.docker.com/v2/repositories/labshare/{{ cookiecutter.project_slug }}/tags/?page_size=1&page=1&ordering=last_updated'
    
    def test_plugin_manifest(self):
        
        # Get the plugin version
        with open(self.version_path,'r') as file:
            version = file.readline()
            
        # Load the plugin manifest
        with open(self.json_path,'r') as file:
            plugin_json = json.load(file)
        
        self.assertEqual(plugin_json['version'],version)
        self.assertTrue(plugin_json['containerId'].endswith(version))
    
    def test_docker_hub(self):
        
        # Get the plugin version
        with open(self.version_path,'r') as file:
            version = file.readline()
            
        response = json.load(request.urlopen(self.url))
        if len(response['results']) == 0:
            self.fail('Could not find repository or no containers are in the repository.')
        latest_tag = json.load(response)['results'][0]['name']
        
        self.assertEqual(latest_tag,version)
        
if __name__=="__main__":
    
    unittest.main()