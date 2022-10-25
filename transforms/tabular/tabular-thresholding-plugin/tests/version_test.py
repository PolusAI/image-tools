import unittest, json
from pathlib import Path
import urllib.request as request

class VersionTest(unittest.TestCase):
    """ Verify VERSION is correct """
    
    version_path = Path(__file__).parent.parent.joinpath("VERSION")
    json_path = Path(__file__).parent.parent.joinpath("plugin.json")
    url = 'https://hub.docker.com/repository/docker/polusai/tabular-thresholding-plugin/tags?page=1&ordering=last_updated'
    
    def test_plugin_manifest(self):
        """ Tests VERSION matches the version in the plugin manifest """
        
        # Get the plugin version
        with open(self.version_path,'r') as file:
            version = file.readline()
            
        # Load the plugin manifest
        with open(self.json_path,'r') as file:
            plugin_json = json.load(file)
        
        self.assertEqual(plugin_json['version'],version)
        self.assertTrue(plugin_json['containerId'].endswith(version))

    def test_docker_hub(self):
        """ Tests VERSION matches the latest docker container tag """
        
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