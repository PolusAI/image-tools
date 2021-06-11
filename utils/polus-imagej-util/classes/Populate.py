#%%

import re, json, logging, copy
import imagej, scyjava, jpype
from pathlib import Path

class Plugin:
    
    def __init__(self, name, library, fullPath, inputs, outputs):
        self.name = name
        self.library = library
        self.fullPath = fullPath
        self.__dataMap(inputs, outputs)
        self.__support()
    
    @property
    def inputs(self):
        return self._inputs
    
    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs
        
    @property
    def outputs(self):
        return self._outputs
    
    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs
        
    @property
    def wippTypeInputs(self):
        return [var[1] for var in self._inputs]
    
    @property
    def wippTypeOutputs(self):
        return self._outputs[0][1]
    
    
    # Define the imagej data types that map to collection
    COLLECTION_TYPES = [
        'Iterable',
        'Interval',
        'IterableInterval',
        # 'IterableRegion',
        'RandomAccessibleInterval',
        'ImgPlus',
        'PlanarImg',
        # 'ImgFactory',
        # 'ImgLabeling',
        'ArrayImg',
        'Img'
    ]

    # Define the imagej data types that map to number
    NUMBER_TYPES = [
        'RealType',
        'NumericType',
        'byte', 'ByteType', 'UnsignedByteType',
        'short','ShortType','UnsignedShortType',
        'int','Integer','IntegerType',
        'long', 'Long', 'LongType', 'UnsignedLongType',
        'float','FloatType',
        'double','Double','DoubleType'
    ]

    # Define the imagej data types that map to boolean
    BOOLEAN_TYPES = [
        'boolean','Boolean','BooleanType'
    ]

    # Define the imagej data types that map to array
    ARRAY_TYPES = [
        # 'double[][]',
        'List',
        'double[]',
        'long[]',
        'ArrayList',
        # 'Object[]',
        'int[]'
    ]

    # Define the imagej data types that map to string
    STRING_TYPES = [
        'RealLocalizable',
        'String'
    ]

    # Save all imagej data types as key and corresponding WIPP data type as value in dictionary
    imagej_to_Wipp_map = {imagej_data_type: 'collection' for imagej_data_type in COLLECTION_TYPES}
    imagej_to_Wipp_map.update({imagej_data_type: 'number' for imagej_data_type in NUMBER_TYPES})
    imagej_to_Wipp_map.update({imagej_data_type: 'boolean' for imagej_data_type in BOOLEAN_TYPES})
    imagej_to_Wipp_map.update({imagej_data_type: 'array' for imagej_data_type in ARRAY_TYPES})
    imagej_to_Wipp_map.update({imagej_data_type: 'string' for imagej_data_type in STRING_TYPES})
    
    def __dataMap(self, inputs, outputs):

        # Create empty lists to store input and output data types
        self._inputs = []
        self._outputs = []
        
        # Iterate over all inputs
        for imagejDataType in inputs:
            # Try to map from imagej data type to WIPP data type
            try:
                self._inputs.append((imagejDataType, Plugin.imagej_to_Wipp_map[imagejDataType]))
                
            # Place WIPP data type as unknown if not currently supported
            except:
                self._inputs.append((imagejDataType, 'unknown'))
        
        # Try to map output imagej data type to WIPP data type
        try:
            self._outputs.append((outputs, Plugin.imagej_to_Wipp_map[outputs]))
            
        # Place WIPP data type as unknown if not currently supported
        except:
            self._outputs.append((outputs, 'unknown'))
            
            
    def __support(self):
        if 'collection' in self.wippTypeInputs and 'collection' in self.wippTypeOutputs:
            self.support = True
        else:
            self.support = False
        


class Populate:
    
    def __init__(self, imagej_help_docs, logfile='full.log'):
        
        # Create logger for class member
        self.__logger(logfile)
        
        # Create imagej plug in by calling the parser member method
        self.plugins = self._parser(imagej_help_docs)
    
    def _parser(self, imagej_help_docs):
        
        # Split each plugin's data into its own string and save in list
        split_plugins = re.split(r'\t(?=\()', imagej_help_docs)
        
        # Complile the regular expression search pattern for the library and name
        re_paths = re.compile(r'(?:[A-z]*\.){3}(?P<library>.*)(?:\.)(?P<name>.*)(?=\()')
        
        # Coompile the regular expression search pattern for the input data types
        re_inputs = re.compile(r'(?<=\t\t)(.*?)(?=[^A-z0-9])')
        
        # Complile the regular expression search pattern for the outputs
        re_outputs = re.compile(r'(?<=^\()(.*?)(?=\s.*\)|\s.*,)')
        
        # Create a dictionary of Plugin object to store resutls of parser
        plugin_dic = {}
        
        # Create plugin counter
        plugin_counter = 0
        
        # Iterate over every imagej plugin in help docs and parse name, library,
        # input data types and output data types
        for plugin in split_plugins[1:]:
            
            plugin_counter += 1
            
            # Search for the plugin name and library
            paths = re_paths.search(plugin).groupdict()
            
            # Save the name and library
            library = paths['library']
            name = paths['name']
            
            # Create the full path
            fullPath = library + '.' +name
            
            # Search for the input data type
            inputs = re_inputs.findall(plugin)
            
            # Search for the output data type
            outputs = re_outputs.search(plugin).group()
            
            # Instantiate a Plugin object and add to dictionary
            plugin_dic[name] = Plugin(name, library, fullPath, inputs, outputs)
            
            if plugin_dic[name].support:
                support_msg = plugin_dic[name].support
            else:
                support_msg = 'The current plug in is not supported, no inputs are a WIPP collection data type or the output is not a WIPP collection data type'
            
            # Log the plugin info to the main log
            self._logger.info(
                self._msg.format(    
                    counter = plugin_counter,
                    name = plugin_dic[name].name,
                    library = plugin_dic[name].library,
                    fullpath = plugin_dic[name].fullPath,
                    inputs = plugin_dic[name].inputs,
                    outputs = plugin_dic[name].outputs,
                    support = support_msg
                )
            )
        
        # Return the dictionary of plug in paths
        return plugin_dic
        
        
    def __logger(self, logfile):
        
        # Check if excluded log exists
        if Path(logfile).exists():
            # Unlink excluded log
            Path(logfile).unlink()
        
        # Create a logger object with name of module
        self._logger = logging.getLogger(__name__)
        
        # Set the logger level
        self._logger.setLevel(logging.INFO)
        
        # Create a log formatter
        self._logFormatter = logging.Formatter('%(message)s')
        
        # Create handler with log file name
        self._fileHandler = logging.FileHandler(logfile)
        
        # Set format of logs
        self._fileHandler.setFormatter(self._logFormatter)
        
        # Add the handler to the class logger
        self._logger.addHandler(self._fileHandler)
        
        self._logger.info(
'This log documents the information obtained from parsing each imagej plug in.\n\
Log is specified in the following format:\n\n\
Plugin Number: The count of plugins parsed\n\
Name: The imagej name of the plug in (e.g. "ConvertImages$Int32")\n\
Library: The plugin library source (e.g. "convert")\n\
Full Path: Full path of plug in (e.g. "convert.ConvertImages$Int32")\n\
Inputs: A list of imagej and WIPP input data types (e.g. [(imagej data type of var1, WIPP data type of var1), (imagej data type of var2, WIPP data type var2)...])\n\
Outputs: The output data type [(imagej data type, WIPP data type)]\n\
Support: Can the current plugin be converted from immagej to WIPP\n\n')
        
        # Create default message for logger
        self._msg = 'Plugin Number: {counter}\nName: {name}\nLibrary: {library}\nFull Path: {fullpath}\nInputs: {inputs}\nOutputs: {outputs}\nSupported: {support}\n\n'
        
        

if __name__ == '__main__':
    import imagej
    
    # Disable warning message
    def disable_loci_logs():
        DebugTools = scyjava.jimport("loci.common.DebugTools")
        DebugTools.setRootLevel("WARN")
    scyjava.when_jvm_starts(disable_loci_logs)
    
    print('Starting JVM\n')
    
    # Start JVM
    ij = imagej.init("sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4",headless=True)
    
    # Retreive all available operations from pyimagej
    imagej_help_docs = scyjava.to_python(ij.op().help())
    
    print('Parsing imagej op help\n')
    
    populater = Populate(imagej_help_docs)

    print('Shutting down JVM')
    
    # Shut down JVM
    jpype.shutdownJVM()
    

# %%
