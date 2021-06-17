#%%

import re, json, logging, copy
import imagej, scyjava, jpype
from pathlib import Path

class Op:
    
    def __init__(self, namespace, name, fullPath, inputs, output):

        # Define class attributes
        self.namespace = namespace
        self.name = name
        self.fullPath = fullPath
        
        # Map the inputs and output from imageJ data type to WIPP data type
        self.__dataMap(inputs, output)
              
        # Define required and optional inputs by testing last character in each input title
        self._requiredInputs = [_input for _input in self._inputs if _input[0][1][-1] != '?']
        self._optionalInputs = [_input for _input in self._inputs if _input[0][1][-1] == '?']
        
        # Determine if the op is currently supported
        self.__support()
    
    @property
    def inputs(self):
        return self._inputs
    
    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs
        
    @property
    def output(self):
        return self._output
    
    @output.setter
    def output(self, output):
        self._output = output
        
    @property
    def imagejInputDataTypes(self):
        return [var[0][0] for var in self._inputs]
    
    @property
    def imagejInputTitles(self):
        return [var[0][1] for var in self._inputs]
    
    @property
    def wippTypeInputs(self):
        return [var[1] for var in self._inputs]
    
    @property
    def wippTypeOutput(self):
        return self._output[0][1]
    
    @property
    def imagejTypeOutput(self):
        return self._output[0][0]
    
    @property
    def wippTypeRequiredInputs(self):
        return [var[1] for var in self._requiredInputs]
    
    
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
    
    
    def __dataMap(self, inputs, output):

        # Create empty lists to store input and output data types
        self._inputs = []
        self._output = []
        
        # Iterate over all inputs
        for _input in inputs:
            
            # Try to map from imagej data type to WIPP data type
            try:
                self._inputs.append((_input, Op.imagej_to_Wipp_map[_input[0]]))
                
            # Place WIPP data type as unknown if not currently supported
            except:
                self._inputs.append((_input, 'unknown'))
        
        # Try to map output imagej data type to WIPP data type
        try:
            self._output.append((output, Op.imagej_to_Wipp_map[output[0]]))
            
        # Place WIPP data type as unknown if not currently supported
        except:
            self._output.append((output, 'unknown'))
        

            
    def __support(self):
        
        # Check if any inputs and the output contains collection data type and all inputs can be mapped from imagej data type to WIPP data type
        if 'collection' in self.wippTypeInputs and 'collection' in self.wippTypeOutput and 'unknown' not in self.wippTypeInputs:
            
            # Set the support attribute as true (imagej op is supported)
            self.fullSupport = True
            self.partialSupport = True
        
        elif 'collection' in self.wippTypeRequiredInputs and 'collection' in self.wippTypeOutput and 'unknown' not in self.wippTypeRequiredInputs:
            self.partialSupport = True
            self.fullSupport = False
        
        else:
            # Set the support attribute as false (imagej op is NOT supported)
            self.fullSupport = False
            self.partialSupport = False
            
        

class Library:
    def __init__(self, library):
        self._library = library
        self._ops = {}
        self._allInputs = {}
        self._allOutputs = {}
        self.supportedOps = {}
        
    def addOp(self, name, op):
        # Add the op to the _ops dicitonary attribute
        self._ops[name] = op
        
        # Check if the op is currently supported
        if op.support:
            
            # Add op to list of supported ops
            self.supportedOps[name] = op
            
            # Add each var to Library's input dictionary
            for title, dtype, wippType in zip(op.imagejInputTitles, op.imagejInputDataTypes, op.wippTypeInputs):
                
                # Check if variable exists in input dicitonary
                if title not in self._allInputs:
                    self._allInputs[title] = {
                        'type':wippType, 
                        'title':title, 
                        'description':title, 
                        'required':False, 
                        'call_types':{name:dtype}
                        }
            
                # If variable key exists update it
                else:
                    self._allInputs[title]['call_types'].update({name:dtype})
                    if self._allInputs[title]['type'] != wippType:
                        #raise Exception
                        print('The', self._library, 'library has multiple input data types for the same input title across different ops')
            
            # Check if the output dictionary has been created
            if 'out' not in self._allOutputs:
            
                # Add the output to Library's output dictionary
                self._allOutputs = {
                    'out':{         
                        'type': op.wippTypeOutput, 
                        'title':'out', 
                        'description':'out',
                        'call_types': {
                            name:op.imagejTypeOutput
                            }
                        }
                    }
            
            else:
                self._allOutputs['out']['call_types'][name] = op.imagejTypeOutput

class Populate:
    
    def __init__(self, imagej_help_docs, logfile='full.log'):
        
        # Create logger for class member
        self.__logger(logfile)
        
        # Create imagej plug in by calling the parser member method
        self._parser(imagej_help_docs)
    
    def _parser(self, imagej_help_docs):
        
        # Get list of all available op namespaces
        opsNameSpace = list(ij.op().ops().iterator())
        
        # Complile the regular expression search pattern for available ops in the namespace
        re_path = re.compile(r'\t(?P<path>.*\.)(?P<name>.*)(?=\()')
        
        # Coompile the regular expression search pattern for the input data types and title
        re_inputs = re.compile(r'(?<=\t\t)(.*?)\s(.*)(?=,|\))')
        
        # Complile the regular expression search pattern for the outputs
        re_output = re.compile(r'^\((.*?)\s(.*)\)')
        
        # Iterate over all ops
        for namespace in opsNameSpace:
            
            # Get the help info about available ops for the namespace
            opDocs = scyjava.to_python(ij.op().help(namespace))
            
            # Split the help string into seperate ops
            splitOps = re.split(r'\t(?=\()', opDocs)
            
            # Create a counter for number of ops parsed
            ops_count = 0
            
            for opDoc in splitOps[1:]:
                
                # Increment the ops parsed count
                ops_count += 1
                
                # Search for op path and name
                opPath = re_path.search(opDoc).groupdict()
                
                # Save op name and full path
                name = opPath['name']
                fullPath = opPath['path'] + name
                
                # Find all inputs
                inputs = re_inputs.findall(opDoc)
                
                # Search for output
                output = re_output.findall(opDoc)[0]
                
                # Create an Op object to store the op data
                op = Op(namespace, name, fullPath, inputs, output)

                if op.partialSupport:
                    support_msg = True
                else:
                    support_msg = 'The current plug in is not supported, no inputs are a WIPP collection data type or the output is not a WIPP collection data type'
                
                # Log the plugin info to the main log
                self._logger.info(
                    self._msg.format(    
                        counter = ops_count,
                        namespace = namespace,
                        name = name,
                        fullpath = fullPath,
                        inputs = str(op.inputs),
                        output = str(op.output),
                        support = support_msg
                    )
                )
                
                # Split each plugin's data into its own string and save in list
                #split_plugins = re.split(r'\t(?=\()', imagej_help_docs)
                
                # Complile the regular expression search pattern for the library and name
                #re_paths = re.compile(r'(?:[A-z]*\.){3}(?P<library>.*)(?:\.)(?P<name>.*)(?=\()')
                
                # Coompile the regular expression search pattern for the input data types and title
                #re_inputs = re.compile(r'(?<=\t\t)(.*?)\s(.*)(?=,|\))')
                
                # Complile the regular expression search pattern for the outputs
                #re_output = re.compile(r'(?<=\().*?(?=\s.*\)|\s.*,)')
            
                # Create a dictionary of Plugin object to store resutls of parser
                #self.libraries = {}
                
                # Create plugin counter
                #plugin_counter = 0
            
            # Iterate over every imagej plugin in help docs and parse name, library,
            # input data types and output data types
            # for plugin in split_plugins[1:]:
                
            #     plugin_counter += 1
                
            #     # Search for the plugin name and library
            #     paths = re_paths.search(plugin).groupdict()
                
            #     # Save the name and library
            #     library = paths['library']
            #     name = paths['name']
                
            #     # Create the full path
            #     fullPath = library + '.' +name
                
            #     # Search for the input data type
            #     inputs = re_inputs.findall(plugin)
                
            #     # Search for the output data type
            #     output = re_output.search(plugin).group()
                
            #     op = Op(library, name, fullPath, inputs, output)
                
            #     # Check if the library exists
            #     if library in self.libraries:
                    
            #         # Add the op to the library
            #         self.libraries[library].addOp(name, op)
                    
            #     else:
            #         # Create the library
            #         self.libraries[library] = Library(library)
                    
            #         # Add the op to the library
            #         self.libraries[library].addOp(name, op)
                    
                    
            #     if self.libraries[library]._ops[name].support:
            #         support_msg = True
            #     else:
            #         support_msg = 'The current plug in is not supported, no inputs are a WIPP collection data type or the output is not a WIPP collection data type'
                
            #     # Log the plugin info to the main log
            #     self._logger.info(
            #         self._msg.format(    
            #             counter = plugin_counter,
            #             name = self.libraries[library]._ops[name].name,
            #             library = self.libraries[library]._ops[name].library,
            #             fullpath = self.libraries[library]._ops[name].fullPath,
            #             inputs = self.libraries[library]._ops[name].inputs,
            #             output = self.libraries[library]._ops[name].output,
            #             support = support_msg
            #         )
            #     )
        
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
        
        # Create header info for the main log
        loginfo = ''
        
        # Open the main log info template
        with open('logtemplates/mainlog.txt') as fhand:
            for line in fhand:
                loginfo += line
                
        # Close the file connection
        fhand.close()
        
        # Set the header info
        self._logger.info(loginfo)

        
        # Create default message for logger
        self._msg = 'Op Number: {counter}\nNamespace: {namespace}\nName: {name}\nFull Path: {fullpath}\nInputs: {inputs}\nOutput: {output}\nSupported: {support}\n\n'
    
        # Create a new logger to log input warnings
        
        
        
    def buildJSON(self, author, email, github_username, version):
        
        # Instantiate empty dictionary to store the dictionary to be converted to json
        self.jsonDic = {}
            
        
        # Iterate over all imagej libraries that were parsed
        for libName, lib, in self.libraries.items():
            
            # Check if any ops are suppported
            if len(lib.supportedOps) > 0:
                
                # Add the json "template" for the library to the dictionary containing all library "templates"
                self.jsonDic[libName] = {
                    'author': author,
                    'email': email,
                    'github_username': github_username,
                    'version': version,
                    'project_name': 'ImageJ ' + libName.replace('.', ' '),
                    'project_short_description': str([op for op in lib.supportedOps.keys()]).replace("'", '')[1:-1],
                    'plugin_namespace':{
                        op.name: 'out = ij.op().' + op.library.replace('.', '().') + '().' + op.name + str(tuple(op.imagejInputTitles)) for op in lib.supportedOps.values()
                        },
                    '_inputs':{
                        'opName':{
                            'title': 'Operation',
                            'type': 'enum',
                            'options':[
                                op.name for op in lib.supportedOps.values()
                                ],
                            'description': 'Operation to peform',
                            'required': 'false'
                            }
                        },
                    'outputs':
                        lib._allOutputs
                    }
                
                # Update the _inputs section dictionary with the inputs dictionary stored in the Library attribute
                self.jsonDic[libName]['_inputs'].update(lib._allInputs)
            
if __name__ == '__main__':
    import imagej
    
    # Disable warning message
    def disable_loci_logs():
        DebugTools = scyjava.jimport('loci.common.DebugTools')
        DebugTools.setRootLevel('WARN')
    scyjava.when_jvm_starts(disable_loci_logs)
    
    print('Starting JVM\n')
    
    # Start JVM
    ij = imagej.init('sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4',headless=True)
    
    
    # Retreive all available operations from pyimagej
    imagej_help_docs = scyjava.to_python(ij.op().help())
    #print(imagej_help_docs)
    
    print('Parsing imagej op help\n')
    
    # Populate ops by parsing the imagej operations help
    populater = Populate(imagej_help_docs)
    
    print('Building json template')
    
    #Build the json dictionary to be passed to the cookiecutter module 
    #populater.buildJSON('Benjamin Houghton', 'benjamin.houghton@axleinfo.com', 'bthoughton', '0.1.1')
    
    print('Shutting down JVM')
    
    del ij
    
    # Shut down JVM
    jpype.shutdownJVM()
    

# %%
