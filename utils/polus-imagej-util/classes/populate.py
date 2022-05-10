import re
import json
import logging
from matplotlib.font_manager import json_load
from numpy import False_
import scyjava
import imagej
from pathlib import Path

"""
This file provides classes to parse the imagej ops help and create cookiecutter 
json templates. This file is should be run before generating ops with 
generate.py or using the ImageJ UI. 
"""

# Disable warning message
def disable_loci_logs():
    DebugTools = scyjava.jimport('loci.common.DebugTools')
    DebugTools.setRootLevel('WARN')
scyjava.when_jvm_starts(disable_loci_logs)


class Op:
    
    """A class to represent each Imagej overload method with corresponding 
    inputs and outputs.
    
    The Op class is intended to be used in conjunction with the Plugin and 
    Populator classes. Altogether the three classes parse and store the imagej 
    ops help and finally construct the json template files used to construct the 
    main program and unit testing. Each Op represents a single imagej 
    overloading method. The attributes of the op store the various input and 
    output titles and their corresponding WIPP and imagej data types. The class 
    also stores the required and optional inputs as indicated by a '?' directly 
    following an input title in the imagej ops help (e.g. ?optional_input1).
    
    Attributes:
        name: A string representing the imagej name of the overloading method
        plugin: A Plugin class member representing the imagej op of which 
            the overloading method belongs.
        _inputs: A list of tuples containing the input title, imagej data type 
            and WIPP data type (see full.log for structure of _inputs list).
        _output: A list containing a single tuple of the output title, imagej 
            data type and imagej data type (see full.log for structure of 
            _output list).
        _required_inputs: A list of tuples containing input title, imagej data 
            type and WIPP data type of the required inputs of the method.
        _optional_inputs: A list of tuples containing input title, imagej data 
            type and WIPP data type of the optional inputs of the method.
        full_support: A boolean indicating if the overloading method is 
            supported using required and optional inputs. At this time no 
            optional inputs are supported. 
        partial_support: A boolean indicating if the overloading method is 
            supported using the required inputs. Additionally a method must take 
            a collection as input and output a collection to be partially 
            supported.
        support_msg: A list of booleans indicating why an op method is or is not
            supported. The first value indicates if all required inputs and 
            output can be mapped to a WIPP data type and the second value 
            indcates if both the required inputs and output contain a collection 
            data type.
        imagej_input_data_types: A list of strings representing the imagej data 
            types of the method's inputs.
        imagej_input_titles: A list of strings representing the imagej input 
            titles of the method.
        wipp_type_inputs: A list of strings representing the WIPP data types of 
            the method's inputs.
        wipp_type_output: A string representing the WIPP data type of the 
            method's output.
        imagej_type_output: A string representing the imagej data type of the 
            method's output.
        imagej_title_output: A string representing the imagej output title of 
            the method.
        wipp_type_required_inputs: A list of strings representing the WIPP data 
            type of the required inputs.
        imagej_type_required_inputs: A list of strings representing the imagej 
            data type of the required inputs.
        imagej_title_required_inputs A list of strings representing the imagej 
            input titles the required inputs.
        
    """
    
    def __init__(self, 
                 plugin: 'Plugin', 
                 name: str, 
                 full_path: str, 
                 inputs: list, 
                 output: tuple):
        
        """A method to instantiate an Op class member
        
        Args:
            plugin: The Plugin object representing the imagej op that the 
                overloading method belongs. Plugin instance.
            name: A string of representing the overloading method name.
            full_path: A string representing the full Java call for the op and 
                its overloading method.
            inputs: A list of tuples containing the imagej input titles and 
                imagej data types.
            output: A tuple containing the imagej output title and imagej data 
                type.
        
        Raises:
            TypeError: Raises if inputs is not a list.
        """
        
        if not isinstance(inputs, list):
            raise TypeError('inputs must be an instance of a list')
        
        # Define class attributes
        self.plugin = plugin
        self.name = name
        self.full_path = full_path
        self._inputs = []
        self._output = []
        
        # Check and update if any input titles
        for input_index, input in enumerate(inputs):
            # Check if input title will interfere with reserved python keyword
            if input[1] == 'in':
                # Change the input name from "in" to "in1" 
                inputs[input_index] = (input[0], 'in1')
            
            # Change the name of the out as input argument so it does not 
            # interfere with output from op
            elif input[1] == 'out':
                # Change name from "out" to "out_input"
                inputs[input_index] = (input[0], 'out_input')

            elif input[1] == 'out?':
                # Change name from "out" to "out_input"
                inputs[input_index] = (input[0], 'out_input?')
                
        # Check if the output is not titled 'out' and change to 'out' if 
        # neccessary
        if output[1] != 'out':
            output = list(output)
            output[1] = 'out'
            output = tuple(output)
        
        # Map the inputs and output from ImageJ data type to WIPP data type
        #self.__dataMap(inputs, output)
        self._inputs.extend([(_input,Op.imagej_to_Wipp_map.get(_input[0],'unknown')) for _input in inputs])
        self._output.extend([(output,Op.imagej_to_Wipp_map.get(output[0],'unknown'))])
              
        # Define required and optional inputs by testing last character in each 
        # input title
        self._required_inputs = [
            _input for _input in self._inputs if _input[0][1][-1] != '?'
        ]
        self._optional_inputs = [
            _input for _input in self._inputs if _input[0][1][-1] == '?'
        ]
        
        # Determine if the op is currently supported and define member 
        # attributes for partial and full support
        self.__support()
    
    @property
    def imagej_input_data_types(self):
        return [var[0][0] for var in self._inputs]
    
    @property
    def imagej_input_titles(self):
        return [var[0][1] for var in self._inputs]
    
    @property
    def wipp_type_inputs(self):
        return [var[1] for var in self._inputs]
    
    @property
    def wipp_type_output(self):
        return self._output[0][1]
    
    @property
    def imagej_type_output(self):
        return self._output[0][0][0]
    
    @property
    def imagej_title_output(self):
        return self._output[0][0][1].replace('?', '')
    
    @property
    def wipp_type_required_inputs(self):
        return [var[1] for var in self._required_inputs]
    
    @property
    def imagej_type_required_inputs(self):
        return [var[0][0] for var in self._required_inputs]
    
    @ property
    def imagej_title_required_inputs(self):
        return [var[0][1] for var in self._required_inputs]
    
    
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
    
    
    def __dataMap(
        self, 
        inputs: list, 
        output: tuple
        ) -> None:
        
        """This method is DEPRACTED - A method to map each imagej input data 
        type to a WIPP data type.
        
        This method is called when parsing the imagej ops help and is not 
        intended to be called directly. The method attempts to map all inputs 
        and the output from an imagej data type to a WIPP data type. Note that 
        the method does not create a WIPP data object, the data type is only 
        stored as a string in the input and output attributes of each member 
        method. If a data type conversion is not currently supported the method 
        will store 'unknown' for the data type.
        
        
        Args:
            inputs: A list of tuples containing the imagej input titles and data 
                types.
            output: A tuple containing the imagej output title and data type.
        
        Returns:
            None
        
        Raises:
            None
        """
        
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
        
        """A method to determine if the imagej op is currently supported by the 
        op generation pipeline.
        
        This method uses the input and output data types to determine if an op 
        is currently supported. For an op to be supported is must have 
        collection as one of the required inputs and the output must also be a 
        collection. Additionally, all the required inputs and the output 
        must be able to map from imagej to WIPP for partial support. For full 
        support all of the inputs and output must be able to map from imagej to 
        WIPP. If the data type conversion is not supported 'unknown' will be 
        stored as the WIPP type. At this time, this pipeline only supports 
        required inputs. Therefore, full support is arbitrary for the purposes 
        of plugin generation, this feature was only added for future 
        development.
        
        Args:
            None
        
        Returns:
            None
        
        Raises:
            None
        """
        # Create initial support message for partial
        self.support_msg = [
            True, 
            True
        ]
        
        """Check for full support"""
        
        # Initially set full support True
        self.full_support = True
        
        # If inputs or output cannot be mapped to WIPP data type
        if 'unknown' in self.wipp_type_inputs + [self.wipp_type_output]:
            self.full_support = False
        
        # Check if the input and output both contain collection data types
        elif 'collection' not in self.wipp_type_inputs or 'collection' not in self.wipp_type_output:
            self.full_support = False
            
        """Check for partial support"""
        
        # Initially set partial support to True
        self.partial_support = True
        
        # Check that required inputs and ouput can be mapped to WIPP data type
        if 'unknown' in self.wipp_type_required_inputs + [self.wipp_type_output]:
            self.partial_support = False
            self.support_msg[0] = False
            
        # Check if the input and output both contain collection data types
        if 'collection' not in self.wipp_type_required_inputs or 'collection' not in self.wipp_type_output:
            self.partial_support = False
            self.support_msg[1] = False
        
        
            
        
class Plugin:
    
    """A class to represent imagej ops and plugins.
    
    The Plugin class is used to store all the information about each plugin, 
    which is later used to build the plugin directory and files. Each Plugin 
    can be thought of as a single imagej op. Each op in turn has a number of 
    overloading methods for different data types. The attributes of a Plugin 
    object store the relevant information about the op and its child overloading 
    methods. The Populate class also uses to build the cookiecutter json files 
    for plugin generation.
    
    Attributes:
        _name: A string representing the imagej op
        _ops: A dictionary containing the overloading methods of the op as keys 
            and class Op objects as values.
        _all_required_inputs: A dictionary containing information about the 
            required inputs of all overloading methods.
        _all_outputs: A dictionary containing information about the outputs of 
            all overloading methods.
        supported_ops: A dictionary containing the supported overloading methods 
            as keys and the corresponding class Op objects as values.
    """
    
    def __init__(self, 
                 name: str):
        
        """A method to instantiate a Plugin object.
        
        Args:
            name: A string representing imagej op name.
            
        Raises:
            None
        """
        
        self._name = name
        self._ops = {}
        self._all_required_inputs = {}
        self._all_outputs = {}
        self.supported_ops = {}
        
    def add_op(self, 
              op: 'Op') -> None:
        
        """A method to store information about an overloading method in the 
        class member's attributes.
        
        This method's function is to store information about an imagej op and 
        its overloading methods. As overloading methods are parsed from the 
        imagej ops help, class Ops objects are instantiated and referenced in 
        the _ops attribute. The method also stores information about the op 
        which is used to build cookiecutter json template files. 
        
        Args:
            op: An object of class Op, representing one of the ops imagej 
                overloading methods. 
        
        Returns:
            None
        
        Raises:
            None
        """
        
        # Add the op to the _ops dicitonary attribute
        self._ops[op.name] = op
        
        # Check if the op is currently supported
        if op.partial_support:
            
            # Add op to list of supported ops
            self.supported_ops[op.name] = op
            
            # Add each var to plugin's input dictionary
            for title, dtype, wippType in zip(op.imagej_title_required_inputs, 
                                              op.imagej_type_required_inputs, 
                                              op.wipp_type_required_inputs):
                
                # Check if variable exists in input dictionary
                if title not in self._all_required_inputs:
                    self._all_required_inputs[title] = {
                        'type':wippType, 
                        'title':title, 
                        'description':title, 
                        'required':False, 
                        'call_types':{op.name:dtype},
                        'wipp_type':{op.name:wippType}
                        }
            
                # If variable key exists update it
                else:
                    self._all_required_inputs[title]['wipp_type'].update({op.name:wippType})
                    self._all_required_inputs[title]['call_types'].update({op.name:dtype})
                    if self._all_required_inputs[title]['type'] != wippType:
                        #raise Exception
                        #print('The', self._name, 'plugin has multiple input data types for the same input title across different op overloading calls')
                        pass
                    
            # Check if the output dictionary is empty
            if self._all_outputs == {}:
            
                # Add the output to Library's output dictionary
                self._all_outputs = {
                    op.imagej_title_output:{         
                        'type': op.wipp_type_output, 
                        'title': op.imagej_title_output, 
                        'description':'out',
                        'call_types': {
                            op.name:op.imagej_type_output
                            }
                        }
                    }
            
            # Check if the output title is not in dictionary
            elif op.imagej_title_output not in self._all_outputs:
                self._all_outputs.update({
                    op.imagej_title_output:{         
                        'type': op.wipp_type_output, 
                        'title': op.imagej_title_output, 
                        'description':'out',
                        'call_types': {
                            op.name:op.imagej_type_output
                        }
                    }
                })
            
            else:
                self._all_outputs[op.imagej_title_output]['call_types'][op.name] = op.imagej_type_output

class Populate:
    """A class to parse imagej ops information and build json templates for 
    plugin generation.
    
    The Populate class has several methods that utilize the Op and Plugin 
    classes to parse store, and finally build cookiecutter json templates from 
    the imagej ops help. The attributes of a class Populate member store the 
    information about all imagej ops and their overloading methods. Note that 
    this class is not intended to be called directly; instead, a class member is 
    instantiated with Generate.py. The populate class also instantiates an
    ImageJ instance for op parsing.
    
    Attributes:
        _ij: A net.imagej.Imagej instance from which to parse the imagej ops 
            help.
        log_file: A str representing the path to the log file.
        log_template: A str representing the path to a txt file which is used as 
            the log header. This file should explain the format of the final log 
            file.
        _logger: A logging.Logger object which logs information about all imagej 
            ops and methods.
        _log_formatter: A logging.Formatter object to set to format of the log 
            file.
        _file_handler: A logging.FileHandler object to handle to log file.
        _plugins: A dic with op names as keys and class Plugin objects as 
            values. This dic contains the information about all imagej ops and 
            their overloading methods.
        json_dic: A dictionary with op names as keys and the cookiecutter json 
            dictionaries to be used for plugin generation.
        scale (dict): Plugins represented as keys and scale type/class 
            represented as values.
    
    """
    
    def __init__(
        self,
        log_file='./utils/polus-imagej-util/full.log',
        log_template='./utils/polus-imagej-util/classes/logtemplates/mainlog.txt'
        ):
        
        """A method to instantiate a class Populate object
        
        Args:
            log_file: A str representing the path to the log file.
            log_template: A str representing the path to a txt file which is 
                used as the log header. This file should explain the format of 
                the final log file.
            
        Raises:
            None
        
        """
        
        # Instantiate the imagej instance
        self._ij = imagej.init('sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4', headless=True)
        
        # Store the log output file and log template file path
        self.log_file = log_file
        self.log_template = log_template
        
        # Load the scalability configuration file
        with open(Path(__file__).parents[1].joinpath('scale.json'), 'r') as f:
            self.scale = json.load(f)
        
        # Create dictionary to store all plugins
        self._plugins = {}
        
        # Create logger for class member
        self.__logger(self.log_file, self.log_template)
        
        # Create imagej plug in by calling the parser member method
        self._parser()
    
    def _parser(self) -> None:
        """"A method to parse imagej ops help and extract imagej op information.
        
        This method utilizes the python re module to parse the imagej instance 
        ops help. The method then instantiates class Op and class Plugin 
        objects to store information about the ops and methods. Finally relevant 
        information about the ops and methods is written to the log file.
        
        Args:
            None
        
        Returns:
            None
        
        Raises:
            None
        """
        
        # Get list of all available ops to be converted to plugins
        plugins = scyjava.to_python(self._ij.op().ops().iterator())
        
        # Complile the regular expression search pattern for op overloading 
        # methods
        re_path = re.compile(r'\t(?P<path>.*\.)(?P<name>.*)(?=\()')
        
        # Coompile the regular expression search pattern for the input data 
        # types and title
        re_inputs = re.compile(r'(?<=\t\t)(.*?)\s(.*)(?=,|\))')
        
        # Complile the regular expression search pattern for the outputs
        re_output = re.compile(r'^\((.*?)\s(.*)\)')
        
        # Create a counter for number of ops parsed
        ops_count = 0
        
        # Iterate over all ops
        for plugin in plugins:
            
            # Add the plugin to the dictionary
            self._plugins[plugin] = Plugin(plugin)
            
            # Get the help info about the plugin/op
            op_docs = scyjava.to_python(self._ij.op().help(plugin))
            
            # Split the help string into seperate ops
            split_ops = re.split(r'\t(?=\()', op_docs)
            
            # Iterate over all op methods in the plugin/op
            for op_doc in split_ops[1:]:
                
                # Increment the ops parsed count
                ops_count += 1
                
                # Search for op path and name
                op_path = re_path.search(op_doc).groupdict()
                
                # Save op name and full path
                name = op_path['name']
                full_path = op_path['path'] + name
                
                # Find all inputs
                inputs = re_inputs.findall(op_doc)
                
                # Search for output
                output = re_output.findall(op_doc)[0]
                
                # Create an Op object to store the op data
                op = Op(plugin, name, full_path, inputs, output)

                # Check if the op is supported
                if op.partial_support:
                    support_msg = True
                else:
                    support_msg = op.support_msg
                    
                
                # Log the plugin info to the main log
                self._logger.info(
                    self._msg.format(    
                        counter = ops_count,
                        plugin = plugin,
                        name = name,
                        full_path = full_path,
                        inputs = op._inputs,
                        output = op._output,
                        support = support_msg
                    )
                )
                
                # Add the overlaoding method to the plugin/op
                self._plugins[plugin].add_op(op)
        
    def __logger(self, 
                 log_file: str, 
                 log_template: str) -> None:
        
        """A method to initialize a logger and log information about the imagej 
        ops and overloading methods.
        
        The logger makes use of python's built-in logger module to log relevant 
        information about each op and its overloading methods as they are parsed 
        from the imagej ops help.
        
        Args:
            log_file: A str representing the path to the log file.
            log_template: A str representing the path to a txt file which is 
                used as the log header. This file should explain the format of 
                the final log file.
        
        Returns:
            None
        
        Raises:
            None
        """

        # Check if excluded log exists
        if Path(log_file).exists():
            # Unlink excluded log
            Path(log_file).unlink()
        
        # Create a logger object with name of module
        self._logger = logging.getLogger(__name__)
        
        # Set the logger level
        self._logger.setLevel(logging.INFO)
        
        # Create a log formatter
        self._log_formatter = logging.Formatter('%(message)s')
        
        # Create handler with log file name
        self._file_handler = logging.FileHandler(log_file)
        
        # Set format of logs
        self._file_handler.setFormatter(self._log_formatter)
        
        # Add the handler to the class logger
        self._logger.addHandler(self._file_handler)
        
        # Create header info for the main log
        loginfo = ''
        
        # Open the main log info template
        with open(log_template) as fhand:
            for line in fhand:
                loginfo += line
                
        # Close the file connection
        fhand.close()
        
        # Set the header info
        self._logger.info(loginfo)

        
        # Create default message for logger
        self._msg = \
            'Op Number: {counter}\n' +\
            'Op Name: {plugin}\n' +\
            'Op Method: {name}\n' +\
            'Full Path: {full_path}\n' +\
            'Inputs: {inputs}\n' +\
            'Output: {output}\n' +\
            'Supported: {support}\n\n'
        
        
    def build_json(self, 
                  author: str, 
                  email: str, 
                  github_username: str, 
                  version: str, 
                  cookietin_path: str) -> None:
        
        """A method to create cookiecutter json dictionaries for plugin 
        generation.
        
        This method uses the information stored in each class Op object and 
        class Plugin object to create the final cookiecutter json 
        dictionaries to be used for plugin directories and files. Upon creation
        of the json dictionary the method utilizes the json module to write 
        ( json.dump() ) the dictionary contents of each op into a json file in 
        the cookietin directory.
        
        Args:
            author: A string representing the author of the plugin.
            email: A string representing the email of the author of the plugin.
            github_username: A string representing the GitHub username of the 
                author of the plugin. 
            version: A string representing the version number of the plugin. 
            cookietin_path: A str representing the path to the cookietin 
                directory.
            
        Returns:
            None
        
        Raises:
            None
        """
        
        
        # Instantiate empty dictionary to store the dictionary to be converted 
        # to json
        self.json_dic = {}
            
        # Create dic of characters to be replaced in overloading method call
        # (plugin_namespace)
        char_to_replace = {
            '[':'(',
            ']':')',
            "'":'',
            ' ':''
            }
        
        # Iterate over all imagej libraries that were parsed
        for name, plugin, in self._plugins.items():
                        
            # Check if any ops are suppported
            if len(plugin.supported_ops) > 0:
                
                # Add the json "template" for the library to the dictionary 
                # containing all library "templates"
                self.json_dic[name] = {
                    'author': author,
                    'email': email,
                    'github_username': github_username,
                    'version': version,
                    'project_name': 'ImageJ ' + name.replace('.', ' '),
                    'project_short_description': 
                        str([op for op in plugin.supported_ops.keys()]).replace("'", '')[1:-1],
                    'citation': '',
                    'plugin_namespace':{
                        op.name: 
                            'out = ij.op().' + \
                            op.plugin.replace('.', '().') + \
                            re.sub(r"[\s'\[\]]", 
                                   lambda x: char_to_replace[x.group(0)], 
                                   str(op.imagej_title_required_inputs)) 
                            for op in plugin.supported_ops.values()
                        },
                    '_inputs':{
                        'opName':{
                            'title': 'opName',
                            'type': 'enum',
                            'options':[
                                op.name for op in plugin.supported_ops.values()
                                ],
                            'description': 'Operation to perform',
                            'required': 'False'
                            }
                        },
                    '_outputs':
                        plugin._all_outputs,
                    'project_slug': "polus-{{ cookiecutter.project_name|lower|replace(' ', '-') }}-plugin",
                    'docker_repo' : "{{ cookiecutter.project_name|lower|replace(' ', '-') }}-plugin",
                    'scalability': self.scale.get(name.replace('.', '-'), "independent")
                    }
                
                
                # If threhsolding op add compute threhsold call to template
                if self.json_dic[name]['scalability'] == 'threshold':
                    self.json_dic[name]['compute_threshold'] = 'threshold = ij.op().' \
                        + name.replace('.', '().') + '(histogram)'
              
                # Update the _inputs section dictionary with the inputs 
                # dictionary stored in the Library attribute
                self.json_dic[name]['_inputs'].update(plugin._all_required_inputs)
            
                # Create Path object with directory path to store 
                # cookiecutter.json file for each plugin
                file_path = Path(cookietin_path).with_name('cookietin').joinpath(plugin._name.replace('.','-'))
                
                # Create the directory
                file_path.mkdir(exist_ok=True,parents=True)
                
                # Open the directory and place json file in directory
                with open(file_path.joinpath('cookiecutter.json'),'w') as fw:
                    json.dump(self.json_dic[name], fw,indent=4)


class GeneratedParser:
    """A class to update cookiecutter templates with previously generated ImageJ
    plugin manifests.
    
    The advantage to using this class is it allows the user to update
    cookiecutter template files with plugin descriptions, input names and input
    descriptions which have to be determined through manual research about the
    ImageJ op. Before using this class's update method all ImageJ plugins which 
    have erroneous manifests should be stashed or removed from polus-plugins
    directory to avoid overwriting clean cookiecutter templates. Alternatively,
    pass a list for which plugins to specifically target when updating 
    templates. 
    
    Attributes:
        polus_plugins (pathlib.Path): The sysytem file path to the polus-plugins
            directory.
        cookietin (pathlib.Path): The system file path to the cookiecutter 
            template directory.
        manifests (dict): The previously generated ImageJ plugin manifests with
            short plugin names as keys (i.e. "filter-convolve").
        cookiecutter (dict): The cookiecutter templates with short plugin names
            as keys (i.e. "threshold-maxLikelihood").
            
    Methods:
        update_templates(self, ops): A method to update the cookiecutter 
            template files with the previously generated ImageJ plugin 
            manifests.
    """
    
    def __init__(
        self, polus_plugins_directory=None, 
        cookiecutter_directory=None
        ):
        """Instantiates a member of the GeneratedParser class and parses all of
        the cookiecutter templates and plugin manifests from their respective
        directories. 
        
        Arguments:
            polus_plugins_directory (str): The sysytem file path to the 
                polus-plugins directory. If passed as None it will automatically 
                be defined assuming that populate.py is in standard location.
                (polus-plugins/utils/polus-imagej-util/classes/populate.py)
            cookiecutter_directory (str): The system file path to the 
                cookiecutter template directory. If passed as None it will 
                automatically be defined assuming that cookietin and populate.py
                are in standard locations.
                (polus-plugins/utils/polus-imagej-util/cookietin/).
        
        Raises:
            None
        """
        
        # Check if polus-plugins path passed by caller
        if polus_plugins_directory is None:
            
            # Define the polus-plugins directory as member attribute
            self.polus_plugins = Path(__file__).parents[3]
        
        else:
            # Define the polus-plugins directory as member attribute
            self.polus_plugins = Path(polus_plugins_directory)
            
        # Check if cookiecutter directory passed by caller
        if cookiecutter_directory is None:
            
            # Define the cookiecutter template directory as member attribute
            self.cookietin = Path(__file__).parents[1].joinpath('cookietin')
            
        # Get all ImageJ plugin manifests in polus-plugins
        manifest_paths = [p for p in self.polus_plugins.rglob('polus-imagej*/plugin.json')]
        
        # Instantiate dictionary to store the plugin manifests
        self.manifests = {}
        
        # Iterate over all the plugin manifest paths
        for p in manifest_paths:
            
            # Define plugin name as would be defined in cookiecutter template
            plugin_name = str(p.parent.name).replace('polus-imagej-', '').replace('-plugin', '')
            
            # Load the plugins manifest and save in the member attribute dict
            self.manifests[plugin_name] = json_load(p)
            
        
        # Define path to all cookiecutter template files
        cookietin_paths = [p for p in self.cookietin.rglob('cookiecutter.json')]
        
        # Instantiate dictionary to store the cookiecutter templates
        self.cookiecutter = {}
        
        # iterate over all cookiecutter paths
        for p in cookietin_paths:
            
            # Define plugin name as would be defined in cookiecutter template
            plugin_name = str(p.parent.name)
            
            # Load the cookiecutter template files
            self.cookiecutter[plugin_name] = json_load(p)
    
    
    def update_templates(self, ops='all', skip=[]):
        """A method to update the cookiecutter template files with the 
        previously generated ImageJ plugin manifests.
        
        This method will use the plugin manifests found in the polus-plugins
        directory to update the cookiecutter template and update all of the 
        fields which are typically manually entered when gerenating a plugin.
        This method can be used to update all templates or just a selected list.
        Indicates ops = 'all' to update all templates for which a manifest
        exists.
        
        Arguments:
            ops (list): A list of strings representing the plugin cookiecutter
                templates to update in shortened form (i.e. filter-gauss)
            skip (list): A list of plugins to not update, can be used to skip
                plugins with known varying number of inputs due to manual
                updates to the plugin.
        
        Returns:
            None
        
        Raises:
            AssertionError: Raised if the number of inputs in the cookiecutter
                template file do no match the number of inputs in the plugin
                manifest file. This accounts for ops which have a out input for 
                memory allocation which does not appear in both files. 
        """

        # Iterate over all cookiecutter template files
        for plugin, template in self.cookiecutter.items():
            
            # Check if the current op should be updated
            if plugin.lower() not in ops and ops != 'all':
                continue
            
            # Check if caller wants to skip plugin
            elif plugin.lower() in skip:
                continue
            
            # Check if a plugin manifest exists for the current plugin
            elif plugin.lower() not in self.manifests.keys():
                print('{} has no manifest'.format(plugin))
                continue
            
            else:
                print('Updating {}'.format(plugin))
            
            # Define the manifest to be used to update cookiecutter template
            manifest = self.manifests[plugin.lower()]
            
            # Determine if out_input is a cookiecutter template input
            if 'out_input' in template['_inputs'].keys():
                # Check if the input lengths match 
                if len(template['_inputs']) - 1 != len(manifest['inputs']):
                    raise AssertionError('The {} template and manifest have a varying number of inputs'.format(plugin))
                
            else:
                # Check the input lengths for other case
                if len(template['_inputs']) != len(manifest['inputs']):
                    raise AssertionError('The {} template and manifest have a varying number of inputs'.format(plugin))
                        
            
            # Update the plugins header information
            template['author'] = manifest['author']
            template['project_short_description'] = manifest['description']
            template['citation'] = manifest['citation']
            
            # Start a manifest input index
            i = 0
            
            # Iterate over the cookiecutter template inputs and thier data
            for cinp, d in template['_inputs'].items():
                
                # Skip the memory allocaiton input "out_input"
                if cinp == 'out_input':
                    continue
                
                # Update the inputs data/description
                d['title'] = manifest['inputs'][i]['name']
                d['description'] = manifest['inputs'][i]['description']
                
                # Increment the manifest input index
                i += 1
            
            # Create the  plugin's cookiecutter template directory
            self.cookietin.joinpath(plugin).mkdir(exist_ok=True, parents=True)
            
            # Define path to plugin's cookiecutter template file
            cookiecutter_path = self.cookietin.joinpath(plugin).joinpath('cookiecutter.json')
            
            # Write the cookiecutter template file
            with open(cookiecutter_path,'w') as fw:
                json.dump(template, fw,indent=4)

"""This section of uses the above classes to generate cookiecutter templates"""

if __name__ == '__main__':
    
    import jpype
    import os
    
    try:
        print('Starting JVM and parsing ops help\n')
        
        # Populate ops by parsing the imagej operations help
        populater = Populate()
        
        print('Building json templates\n')
        
        # Get the current working directory
        cwd = Path(os.getcwd())
        
        # Save a directory for the cookietin json files
        cookietin_path = cwd.joinpath('utils/polus-imagej-util/cookietin')
        
        # Get the pipeline VERSION
        version_path = Path(__file__).parents[1].joinpath('VERSION')
        with open(version_path, 'r') as fhand:
            version = next(fhand)
        
        # Build the json dictionary to be passed to the cookiecutter module 
        populater.build_json(
            'Benjamin Houghton', 
            'benjamin.houghton@axleinfo.com', 
            'bthoughton', 
            version, 
            cookietin_path)
    
    finally:
        
        print('Shutting down JVM\n')
        
        # Remove the imagej instance
        del populater._ij
        
        # Shut down JVM
        jpype.shutdownJVM()
    
    print('Updating templates with previously genreated ops')
    
    # Instantiate the generated ops parser and update templates with manifests
    parser = GeneratedParser()
    parser.update_templates('all', ['threshold-apply'])
