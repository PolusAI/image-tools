#%%

import re, json, logging, copy
import scyjava
from pathlib import Path

"""
This file parses the imagej ops to create cookiecutter json templates
"""

# Disable warning message
def disable_loci_logs():
    DebugTools = scyjava.jimport('loci.common.DebugTools')
    DebugTools.setRootLevel('WARN')
scyjava.when_jvm_starts(disable_loci_logs)


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
        return self._output[0][0][0]
    
    @property
    def imagejTitleOutput(self):
        return self._output[0][0][1]
    
    @property
    def wippTypeRequiredInputs(self):
        return [var[1] for var in self._requiredInputs]
    
    @property
    def imagejTypeRequiredInputs(self):
        return [var[0][0] for var in self._requiredInputs]
    
    @ property
    def imagejTitleRequiredInputs(self):
        return [var[0][1] for var in self._requiredInputs]
    
    
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
        
        # Check if any inputs or the output contains collection data type and ALL inputs/output can be mapped from imagej data type to WIPP data type
        if ('collection' in self.wippTypeInputs or 'collection' in self.wippTypeOutput) and 'unknown' not in self.wippTypeInputs + [self.wippTypeOutput]:
            
            # Set the support attribute as true (imagej op is supported)
            self.fullSupport = True
            self.partialSupport = True
        
        # Check if the required inputs satisfy the requirements 
        elif ('collection' in self.wippTypeRequiredInputs or 'collection' in self.wippTypeOutput) and 'unknown' not in self.wippTypeRequiredInputs + [self.wippTypeOutput]:
            self.partialSupport = True
            self.fullSupport = False
        
        else:
            # Set the support attribute as false (imagej op is NOT supported)
            self.fullSupport = False
            self.partialSupport = False
            
            # Determine why the op is not supported (check if either the input or output is a collection)
            if 'collection' not in self.wippTypeRequiredInputs and 'collection' not in self.wippTypeOutput:
                self.supportmsg = "None of the required inputs is a 'collection' and the output is not a 'collection'"
                
                # Test if all of the required data types can be converted from imagej to WIPP
                if 'unknown' in self.wippTypeInputs + [self.wippTypeOutput]:
                    self.supportmsg = "None of the required inputs is a 'collection' and the output is not a 'collection' and one of the required inputs and/or the output cannot currently be mapped to a WIPP data type"
                    
            # Test if all of the required data types can be converted from imagej to WIPP
            elif 'unknown' in self.wippTypeInputs + [self.wippTypeOutput]:
                self.supportmsg = "One of the required inputs and/or the output cannot currently be mapped to a WIPP data type"
        

class Namespace:
    def __init__(self, name):
        self._name = name
        self._ops = {}
        self._allRequiredInputs = {}
        self._allOutputs = {}
        self.supportedOps = {}
        
    def addOp(self, op):

        # Add the op to the _ops dicitonary attribute
        self._ops[op.name] = op
        
        # Check if the op is currently supported
        if op.partialSupport:
            
            # Add op to list of supported ops
            self.supportedOps[op.name] = op
            
            # Add each var to namespace's input dictionary
            for title, dtype, wippType in zip(op.imagejTitleRequiredInputs, op.imagejTypeRequiredInputs, op.wippTypeRequiredInputs):
                
                # Check if variable exists in input dicitonary
                if title not in self._allRequiredInputs:
                    self._allRequiredInputs[title] = {
                        'type':wippType, 
                        'title':title, 
                        'description':title, 
                        'required':False, 
                        'call_types':{op.name:dtype}
                        }
            
                # If variable key exists update it
                else:
                    self._allRequiredInputs[title]['call_types'].update({op.name:dtype})
                    if self._allRequiredInputs[title]['type'] != wippType:
                        #raise Exception
                        print('The', self._name, 'namespace has multiple input data types for the same input title across different ops')
            
            # Check if the output dictionary has been created
            if op.imagejTitleOutput not in self._allOutputs:
            
                # Add the output to Library's output dictionary
                self._allOutputs = {
                    'out':{         
                        'type': op.wippTypeOutput, 
                        'title': op.imagejTitleOutput, 
                        'description':'out',
                        'call_types': {
                            op.name:op.imagejTypeOutput
                            }
                        }
                    }
            
            else:
                self._allOutputs['out']['call_types'][op.name] = op.imagejTypeOutput

class Populate:
    
    def __init__(self, ij, logFile='./utils/polus-imagej-util/full.log', logTemplate='./utils/polus-imagej-util/classes/logtemplates/mainlog.txt'):
        
        # Store the imagej instance
        self._ij = ij
        
        # Store the log output file and log template file path
        self.logFile = logFile
        self.logTemplate = logTemplate
        
        # Create dictionary to store all namespaces
        self._namespaces = {}
        
        # Create logger for class member
        self.__logger(self.logFile, self.logTemplate)
        
        # Create imagej plug in by calling the parser member method
        self._parser()
    
    def _parser(self):
        
        # Get list of all available op namespaces
        opsNameSpace = scyjava.to_python(self._ij.op().ops().iterator())
        
        # Complile the regular expression search pattern for available ops in the namespace
        re_path = re.compile(r'\t(?P<path>.*\.)(?P<name>.*)(?=\()')
        
        # Coompile the regular expression search pattern for the input data types and title
        re_inputs = re.compile(r'(?<=\t\t)(.*?)\s(.*)(?=,|\))')
        
        # Complile the regular expression search pattern for the outputs
        re_output = re.compile(r'^\((.*?)\s(.*)\)')
        
        # Create a counter for number of ops parsed
        ops_count = 0
        
        # Iterate over all ops
        for namespace in opsNameSpace:
            
            # Add the namespace to the dictionary
            self._namespaces[namespace] = Namespace(namespace)
            
            # Get the help info about available ops for the namespace
            opDocs = scyjava.to_python(self._ij.op().help(namespace))
            
            # Split the help string into seperate ops
            splitOps = re.split(r'\t(?=\()', opDocs)
            
            # Iterate over all ops in the namespace
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

                # Check if the op is supported
                if op.partialSupport:
                    support_msg = True
                else:
                    support_msg = op.supportmsg
                    
                
                # Log the plugin info to the main log
                self._logger.info(
                    self._msg.format(    
                        counter = ops_count,
                        namespace = namespace,
                        name = name,
                        fullpath = fullPath,
                        inputs = op.inputs,
                        output = op.output,
                        support = support_msg
                    )
                )
                
                # Add the op to the namespace
                self._namespaces[namespace].addOp(op)
        
    def __logger(self, logFile, logTemplate):
        
        # Check if excluded log exists
        if Path(logFile).exists():
            # Unlink excluded log
            Path(logFile).unlink()
        
        # Create a logger object with name of module
        self._logger = logging.getLogger(__name__)
        
        # Set the logger level
        self._logger.setLevel(logging.INFO)
        
        # Create a log formatter
        self._logFormatter = logging.Formatter('%(message)s')
        
        # Create handler with log file name
        self._fileHandler = logging.FileHandler(logFile)
        
        # Set format of logs
        self._fileHandler.setFormatter(self._logFormatter)
        
        # Add the handler to the class logger
        self._logger.addHandler(self._fileHandler)
        
        # Create header info for the main log
        loginfo = ''
        
        # Open the main log info template
        with open(logTemplate) as fhand:
            for line in fhand:
                loginfo += line
                
        # Close the file connection
        fhand.close()
        
        # Set the header info
        self._logger.info(loginfo)

        
        # Create default message for logger
        self._msg = 'Op Number: {counter}\nNamespace: {namespace}\nName: {name}\nFull Path: {fullpath}\nInputs: {inputs}\nOutput: {output}\nSupported: {support}\n\n'
    
        # Create a new logger to log input warnings
        
        
        
    def buildJSON(self, author, email, github_username, version, cookietin_path):
        
        # Instantiate empty dictionary to store the dictionary to be converted to json
        self.jsonDic = {}
            
        
        # Iterate over all imagej libraries that were parsed
        for name, namespace, in self._namespaces.items():
            
            # Check if any ops are suppported
            if len(namespace.supportedOps) > 0:
                
                # Add the json "template" for the library to the dictionary containing all library "templates"
                self.jsonDic[name] = {
                    'author': author,
                    'email': email,
                    'github_username': github_username,
                    'version': version,
                    'project_name': 'ImageJ ' + name.replace('.', ' '),
                    'project_short_description': str([op for op in namespace.supportedOps.keys()]).replace("'", '')[1:-1],
                    'plugin_namespace':{
                        op.name: 'out = ij.op().' + op.namespace.replace('.', '().') + str(tuple(op.imagejTitleRequiredInputs)).replace("'", "").replace(' ', '') for op in namespace.supportedOps.values()
                        },
                    '_inputs':{
                        'opName':{
                            'title': 'Operation',
                            'type': 'enum',
                            'options':[
                                op.name for op in namespace.supportedOps.values()
                                ],
                            'description': 'Operation to peform',
                            'required': 'false'
                            }
                        },
                    '_outputs':
                        namespace._allOutputs,
                    'project_slug': "polus-{{ cookiecutter.project_name|lower|replace(' ', '-') }}-plugin"
                    }
                
                # Update the _inputs section dictionary with the inputs dictionary stored in the Library attribute
                self.jsonDic[name]['_inputs'].update(namespace._allRequiredInputs)
                
                print('\n')
                print(self.jsonDic[name])
                
                # Create Path object with directory path to store cookiecutter.json file for each namespace
                file_path = Path(cookietin_path).with_name('cookietin').joinpath(namespace._name.replace('.','-'))
                
                # Create the directory
                file_path.mkdir(exist_ok=True,parents=True)

                # Open the directory and place json file in directory
                with open(file_path.joinpath('cookiecutter.json'),'w') as fw:
                    json.dump(self.jsonDic[name], fw,indent=4)


if __name__ == '__main__':
    
    import imagej, jpype
    from pathlib import Path
    
    # Disable warning message
    def disable_loci_logs():
        DebugTools = scyjava.jimport('loci.common.DebugTools')
        DebugTools.setRootLevel('WARN')
    scyjava.when_jvm_starts(disable_loci_logs)
    
    print('Starting JVM\n')
    
    # Start JVM
    ij = imagej.init('sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4', headless=True)
    
    
    # Retreive all available operations from pyimagej
    #imagej_help_docs = scyjava.to_python(ij.op().help())
    #print(imagej_help_docs)
    
    print('Parsing imagej ops help\n')
    
    # Populate ops by parsing the imagej operations help
    populater = Populate(ij, logFile = 'full.log', logTemplate='logtemplates/mainlog.txt')
    
    print('Building json template\n')
    
    # Build the json dictionary to be passed to the cookiecutter module 
    populater.buildJSON('Benjamin Houghton', 'benjamin.houghton@axleinfo.com', 'bthoughton', '0.1.1', __file__)
    
    print('Shutting down JVM\n')
    
    del ij
    
    # Shut down JVM
    jpype.shutdownJVM()
    

# %%
