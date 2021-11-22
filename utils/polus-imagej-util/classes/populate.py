import re
import json
import logging
from numpy import False_
import scyjava
import imagej
from pathlib import Path

"""
This file provides classes to parse the imagej ops help and create cookiecutter 
json templates. This file is not intended to be ran directly, instead the 
classes contained here are instantiated with Generate.py.
"""

# Disable warning message
def disable_loci_logs():
    DebugTools = scyjava.jimport("loci.common.DebugTools")
    DebugTools.setRootLevel("WARN")


scyjava.when_jvm_starts(disable_loci_logs)


class Op:

    """A class to represent each Imagej overload method with corresponding inputs and outputs.

    The Op class is intended to be used in conjunction with the Namespace and Populator classes.
    Altogether the three classes parse and store the imagej ops help and finally construct the
    json template files used to construct the main program and unit testing. Each Op represents
    a single imagej overloading method. The attributes of the op store the various input and output
    titles and their corresponding WIPP and imagej data types. The class also stores the required and
    optional inputs as indicated by a '?' directly following an input title in the imagej ops help.
    Each overloading method or Op also stores

    Attributes:
        name: A string representing the imagej name of the overloading method
<<<<<<< HEAD
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
<<<<<<< HEAD
            output can be mapped to a WIPP data type and the second value 
=======
            output can be mapped to a WIPP data type and the secod value 
>>>>>>> 32c0d333bfa71d6311e616bb15d50a6e35b64c8c
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
        
=======
        namespace: A Namespace class member representing the imagej op of which the overloading method belongs.
        _inputs: A list of tuples containing the input title, imagej data type and WIPP data type (see full.log for structure of _inputs list).
        _output: A list containing a single tuple of the output title, imagej data type and imagej data type (see full.log for structure of _output list).
        _requiredInputs: A list of tuples containing input title, imagej data type and WIPP data type of the required inputs of the method.
        _optionalInputs: A list of tuples containing input title, imagej data type and WIPP data type of the optional inputs of the method.
        fullSupport: A boolean indicating if the overloading method is supported using required and optional inputs. At this time no optional inputs are supported.
        partialSupport: A boolean indicating if the overloading method is supported using the required inputs. Additionally a method must take a collection as input
            and output a collection to be partially supported.
        supportmsg: A string indicating why the overloading method is not currently supported.
        imagejInputDataTypes: A list of strings representing the imagej data types of the method's inputs.
        imagejInputTitles: A list of strings representing the imagej input titles of the method.
        wippTypeInputs: A list of strings representing the WIPP data types of the method's inputs.
        wippTypeOutput: A string representing the WIPP data type of the method's output.
        imagejTypeOutput: A string representing the imagej data type of the method's output.
        imagejTitleOutput: A string representing the imagej output title of the method.
        wippTypeRequiredInputs: A list of strings representing the WIPP data type of the required inputs.
        imagejTypeRequiredInputs: A list of strings representing the imagej data type of the required inputs.
        imagejTitleRequiredInputs A list of strings representing the imagej input titles the required inputs.

>>>>>>> origin/imagej-util-clean
    """

    def __init__(
        self,
        namespace: "Namespace",
        name: str,
        fullPath: str,
        inputs: list,
        output: tuple,
    ):

        """A method to instantiate an Op class member

        Args:
            plugin: The Plugin object representing the imagej op that the
                overloading method belongs. Plugin instance.
            name: A string of representing the overloading method name.
            fullPath: A string representing the full Java namespace call of the overloading method.
            inputs: A list of tuples containing the imagej input titles and imagej data types.
            output: A tuple containing the imagej output title and imagej data type.

        Raises:
            TypeError: Raises if inputs is not a list.
        """

        if not isinstance(inputs, list):
            raise TypeError("inputs must be an instance of a list")

        # Define class attributes
        self.plugin = plugin
        self.name = name
        self.fullPath = fullPath

        # Check and update if any inputs are named "in" which conflict with python's reserved key word
        for input_index, input in enumerate(inputs):
            if input[1] == "in":
                # Change the input name from "in" to "in1"
                inputs[input_index] = (input[0], "in1")

        # Check if the output is not titled 'out' and change to 'out' if neccessary
        if output[1] != "out":
            output = list(output)
            output[1] = "out"
            output = tuple(output)

        # Map the inputs and output from imageJ data type to WIPP data type
        self.__dataMap(inputs, output)

        # Define required and optional inputs by testing last character in each input title
        self._requiredInputs = [
            _input
            for _input in self._inputs
            if _input[0][1][-1] != "?" and _input[0][1] not in ["out"]
        ]
        self._optionalInputs = [
            _input
            for _input in self._inputs
            if _input[0][1][-1] == "?" or _input[0][1] in ["out"]
        ]

        # Determine if the op is currently supported and define member attributes for partial and full support
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
    def imagejTitleOutput(self):
        return self._output[0][0][1].replace("?", "")

    @property
    def wippTypeRequiredInputs(self):
        return [var[1] for var in self._requiredInputs]

    @property
    def imagejTypeRequiredInputs(self):
        return [var[0][0] for var in self._requiredInputs]

    @property
    def imagejTitleRequiredInputs(self):
        return [var[0][1] for var in self._requiredInputs]

    # Define the imagej data types that map to collection
    COLLECTION_TYPES = [
        "Iterable",
        "Interval",
        "IterableInterval",
        # 'IterableRegion',
        "RandomAccessibleInterval",
        "ImgPlus",
        "PlanarImg",
        # 'ImgFactory',
        # 'ImgLabeling',
        "ArrayImg",
        "Img",
    ]

    # Define the imagej data types that map to number
    NUMBER_TYPES = [
        "RealType",
        "NumericType",
        "byte",
        "ByteType",
        "UnsignedByteType",
        "short",
        "ShortType",
        "UnsignedShortType",
        "int",
        "Integer",
        "IntegerType",
        "long",
        "Long",
        "LongType",
        "UnsignedLongType",
        "float",
        "FloatType",
        "double",
        "Double",
        "DoubleType",
    ]

    # Define the imagej data types that map to boolean
    BOOLEAN_TYPES = ["boolean", "Boolean", "BooleanType"]

    # Define the imagej data types that map to array
    ARRAY_TYPES = [
        # 'double[][]',
        "List",
        "double[]",
        "long[]",
        "ArrayList",
        # 'Object[]',
        "int[]",
    ]

    # Define the imagej data types that map to string
    STRING_TYPES = ["RealLocalizable", "String"]

    # Save all imagej data types as key and corresponding WIPP data type as value in dictionary
    imagej_to_Wipp_map = {
        imagej_data_type: "collection" for imagej_data_type in COLLECTION_TYPES
    }
    imagej_to_Wipp_map.update(
        {imagej_data_type: "number" for imagej_data_type in NUMBER_TYPES}
    )
    imagej_to_Wipp_map.update(
        {imagej_data_type: "boolean" for imagej_data_type in BOOLEAN_TYPES}
    )
    imagej_to_Wipp_map.update(
        {imagej_data_type: "array" for imagej_data_type in ARRAY_TYPES}
    )
    imagej_to_Wipp_map.update(
        {imagej_data_type: "string" for imagej_data_type in STRING_TYPES}
    )

    def __dataMap(self, inputs: list, output: tuple) -> None:

        """A method to map each imagej input data type to a WIPP data type.

        This method is called when parsing the imagej ops help and is not intended to be called directly.
        The method attempts to map all inputs and the output from an imagej data type to a WIPP data
        type. Note that the method does not create a WIPP data object, the data type is only stored
        as a string in the input and output attributes of each member method. If a data type conversion is not
        currently supported the method will store 'unknown' for the data type.


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
                self._inputs.append((_input, "unknown"))

        # Try to map output imagej data type to WIPP data type
        try:
            self._output.append((output, Op.imagej_to_Wipp_map[output[0]]))

        # Place WIPP data type as unknown if not currently supported
        except:
            self._output.append((output, "unknown"))

    def __support(self):

        """A method to determine if the imagej op is currently supported by the op generation pipeline.

        This method uses the input and output data types to determine if an op is currently
        supported. For an op to be supported is must have collection as one of the required inputs
        and the output must also be a collection. Additionally, all the required inputs and the output
        must be able to map from imagej to WIPP for partial support. For full support all of the inputs
        and output must be able to map from imagej to WIPP. If the data type conversion is not
        supported 'unknown' will be stored as the WIPP type when the __dataMap() member method is called.
        At this time, this pipeline only supports required inputs. Therefore, full support is arbitrary
        for the purposes of plugin generation, this feature was only added for future development.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # Check if any inputs or the output contains collection data type and ALL inputs/output can be mapped from imagej data type to WIPP data type
        if (
            "collection" in self.wippTypeInputs and "collection" in self.wippTypeOutput
        ) and "unknown" not in self.wippTypeInputs + [self.wippTypeOutput]:

            # Set the support attribute as true (imagej op is supported)
            self.fullSupport = True
            self.partialSupport = True

        # Check if the required inputs satisfy the requirements
        elif (
            "collection" in self.wippTypeRequiredInputs
            and "collection" in self.wippTypeOutput
        ) and "unknown" not in self.wippTypeRequiredInputs + [self.wippTypeOutput]:
            self.fullSupport = False
            self.partialSupport = True

        else:
            # Set the support attribute as false (imagej op is NOT supported)
            self.fullSupport = False
            self.partialSupport = False

            # Determine why the op is not supported (check if either the input or output is a collection)
            if (
                "collection" not in self.wippTypeRequiredInputs
                or "collection" not in self.wippTypeOutput
            ):
                self.supportmsg = "None of the required inputs is a 'collection' or the output is not a 'collection'"

                # Test if all of the required data types can be converted from imagej to WIPP
                if "unknown" in self.wippTypeRequiredInputs:
                    self.supportmsg = "None of the required inputs is a colleciton or the output is not a collection AND one of the required inputs cannot currently be mapped to a WIPP data type"

            # Test if all of the required data types can be converted from imagej to WIPP
            elif "unknown" in self.wippTypeRequiredInputs:
                self.supportmsg = "One of the required inputs cannot currently be mapped to a WIPP data type"


class Namespace:

    """A class to represent imagej ops and plugins.

    The Namespace class is used to store all the information about each plugin, which is later
    used to build the plugin directory and files. Each Namespace can be thought of as a single
    imagej op. Each op in turn has a number of overloading methods for different data types.
    The attributes of a Namespace object store the relevant information about the op and its
    child overloading methods. The Populate class also uses to build the cookiecutter json files
    for plugin generation.

    Attributes:
        _name: A string representing the imagej op
        _ops: A dictionary containing the overloading methods of the op as keys and class Op objects as values.
        _allRequiredInputs: A dictionary containing information about the required inputs of all overloading methods.
        _allOutputs: A dictionary containing information about the outputs of all overloading methods.
        supportedOps: A dictionary containing the supported overloading methods as keys and the corresponding class Op
            objects as values
    """

    def __init__(self, name: str):

        """A method to instantiate a Namespace object.

        Args:
            name: A string representing imagej op name.

        Raises:
            None
        """

        self._name = name
        self._ops = {}
        self._allRequiredInputs = {}
        self._allOutputs = {}
        self.supportedOps = {}

    def addOp(self, op: "Op") -> None:

        """A method to store information about an overloading method in the class member's attributes.

        This method's function is to store information about an imagej op and its overloading methods.
        As overloading methods are parsed from the imagej ops help, class Ops objects are instantiated and
        referenced in the _ops attribute. The method also stores information about the op which is used
        to build cookiecutter json template files.

        Args:
            op: An object of class Op, representing one of the ops imagej overloading methods.

        Returns:
            None

        Raises:
            None
        """

        # Add the op to the _ops dicitonary attribute
        self._ops[op.name] = op

        # Check if the op is currently supported
        if op.partialSupport:

            # Add op to list of supported ops
            self.supportedOps[op.name] = op

            # Add each var to namespace's input dictionary
            for title, dtype, wippType in zip(
                op.imagejTitleRequiredInputs,
                op.imagejTypeRequiredInputs,
                op.wippTypeRequiredInputs,
            ):

                # Check if variable exists in input dicitonary
                if title not in self._allRequiredInputs:
                    self._allRequiredInputs[title] = {
                        "type": wippType,
                        "title": title,
                        "description": title,
                        "required": False,
                        "call_types": {op.name: dtype},
                        "wipp_type": {op.name: wippType},
                    }

                # If variable key exists update it
                else:
                    self._allRequiredInputs[title]["wipp_type"].update(
                        {op.name: wippType}
                    )
                    self._allRequiredInputs[title]["call_types"].update(
                        {op.name: dtype}
                    )
                    if self._allRequiredInputs[title]["type"] != wippType:
                        # raise Exception
                        # print('The', self._name, 'namespace has multiple input data types for the same input title across different op overloading calls')
                        pass

            # Check if the output dictionary is empty
            if self._allOutputs == {}:

                # Add the output to Library's output dictionary
                self._allOutputs = {
                    op.imagejTitleOutput: {
                        "type": op.wippTypeOutput,
                        "title": op.imagejTitleOutput,
                        "description": "out",
                        "call_types": {op.name: op.imagejTypeOutput},
                    }
                }

            # Check if the output title is not in dictionary
            elif op.imagejTitleOutput not in self._allOutputs:
                self._allOutputs.update(
                    {
                        op.imagejTitleOutput: {
                            "type": op.wippTypeOutput,
                            "title": op.imagejTitleOutput,
                            "description": "out",
                            "call_types": {op.name: op.imagejTypeOutput},
                        }
                    }
                )

            else:
                self._allOutputs[op.imagejTitleOutput]["call_types"][
                    op.name
                ] = op.imagejTypeOutput


class Populate:
    """A class to parse imagej ops information and build json templates for plugin generation.

    The Populate class has several methods that utilize the Op and Namespace classes to parse
    store, and finally build cookiecutter json templates from the imagej ops help. The attributes
    of a class Populate member store the information about all imagej ops and their overloading methods.
    Note that this class is not intended to be called directly; instead, a class member is instantiated
    with Generate.py

    Attributes:
        _ij: A net.imagej.Imagej instance from which to parse the imagej ops help.
        logFile: A str representing the path to the log file.
        logTemplate: A str representing the path to a txt file which is used as the log header. This file
            should explain the format of the final log file.
        _logger: A logging.Logger object which logs information about all imagej ops and methods.
        _logFormatter: A logging.Formatter object to set to format of the log file.
        _fileHandler: A logging.FileHandler object to handle to log file.
        _namespaces: A dic with op names as keys and class Namespace objects as values. This dic contains
            the information about all imagej ops and their overloading methods.
        jsonDic: A dictionary with op names as keys and the cookiecutter json dictionaries to be used for plugin generation.

    """

    def __init__(
        self,
        ij: "imagej.Imagej",
        logFile="./utils/polus-imagej-util/full.log",
        logTemplate="./utils/polus-imagej-util/classes/logtemplates/mainlog.txt",
    ):

        """A method to instantiate a class Populate object

        Args:
            ij: A net.imagej.Imagej instance from which to parse the imagej ops help.
            logFile: A str representing the path to the log file.
            logTemplate: A str representing the path to a txt file which is used as the log header. This file
                should explain the format of the final log file.

        Raises:
            None

        """

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

    def _parser(self) -> None:
        """ "A method to parse imagej ops help and extract imagej op information.

        This method utilizes the python re module to parse the imagej instance ops help. The method then
        instantiates class Op and class Namespace objects to store information about the ops and methods.
        Finally relevant information about the ops and methods is written to the log file.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # Get list of all available op namespaces
        opsNameSpace = scyjava.to_python(self._ij.op().ops().iterator())

        # Complile the regular expression search pattern for available ops in the namespace
        re_path = re.compile(r"\t(?P<path>.*\.)(?P<name>.*)(?=\()")

        # Coompile the regular expression search pattern for the input data types and title
        re_inputs = re.compile(r"(?<=\t\t)(.*?)\s(.*)(?=,|\))")

        # Complile the regular expression search pattern for the outputs
        re_output = re.compile(r"^\((.*?)\s(.*)\)")

        # Create a counter for number of ops parsed
        ops_count = 0

        # Iterate over all ops
        for namespace in opsNameSpace:

            # Add the namespace to the dictionary
            self._namespaces[namespace] = Namespace(namespace)

            # Get the help info about available ops for the namespace
            opDocs = scyjava.to_python(self._ij.op().help(namespace))

            # Split the help string into seperate ops
            splitOps = re.split(r"\t(?=\()", opDocs)

            # Iterate over all ops in the namespace
            for opDoc in splitOps[1:]:

                # Increment the ops parsed count
                ops_count += 1

                # Search for op path and name
                opPath = re_path.search(opDoc).groupdict()

                # Save op name and full path
                name = opPath["name"]
                fullPath = opPath["path"] + name

                # Find all inputs
                inputs = re_inputs.findall(opDoc)

                # Search for output
                output = re_output.findall(opDoc)[0]

                # Create an Op object to store the op data
                op = Op(plugin, name, full_path, inputs, output)

                # Check if the op is supported
                if op.partial_support:
                    support_msg = True
                else:
                    support_msg = op.supportmsg

                # Log the plugin info to the main log
                self._logger.info(
                    self._msg.format(
                        counter=ops_count,
                        namespace=namespace,
                        name=name,
                        fullpath=fullPath,
                        inputs=op.inputs,
                        output=op.output,
                        support=support_msg,
                    )
                )

                # Add the op to the namespace
                self._namespaces[namespace].addOp(op)

    def __logger(self, logFile: str, logTemplate: str) -> None:

        """A method to initialize a logger and log information about the imagej ops and overloading methods.

        The logger makes use of python's built-in logger module to log relevant information about each op
        and its overloading methods as they are parsed from the imagej ops help.

        Args:
            logFile: A str representing the path to the log file.
            logTemplate: A str representing the path to a txt file which is used as the log header. This file
                should explain the format of the final log file.

        Returns:
            None

        Raises:
            None
        """

        # Check if excluded log exists
        if Path(log_file).exists():
            # Unlink excluded log
            Path(logFile).unlink()

        # Create a logger object with name of module
        self._logger = logging.getLogger(__name__)

        # Set the logger level
        self._logger.setLevel(logging.INFO)

        # Create a log formatter
        self._logFormatter = logging.Formatter("%(message)s")

        # Create handler with log file name
        self._fileHandler = logging.FileHandler(logFile)

        # Set format of logs
        self._fileHandler.setFormatter(self._logFormatter)

        # Add the handler to the class logger
        self._logger.addHandler(self._fileHandler)

        # Create header info for the main log
        loginfo = ""

        # Open the main log info template
        with open(log_template) as fhand:
            for line in fhand:
                loginfo += line

        # Close the file connection
        fhand.close()

        # Set the header info
        self._logger.info(loginfo)

        # Create default message for logger
        self._msg = "Op Number: {counter}\nNamespace: {namespace}\nName: {name}\nFull Path: {fullpath}\nInputs: {inputs}\nOutput: {output}\nSupported: {support}\n\n"

    def buildJSON(
        self,
        author: str,
        email: str,
        github_username: str,
        version: str,
        cookietin_path: str,
    ) -> None:

        """A method to create cookiecutter json dictionaries for plugin generation.

        This method uses the information stored in each class Op object and class Namespace object to create the final
        cookiecutter json dictionaries to be used for plugin directories and files. Upon creation of the json dictionary
        the method utilizes the json module to write ( json.dump() ) the dictionary contents of each op into a json file in the
        cookietin directory.

        Args:
            author: A string representing the author of the plugin.
            email: A string representing the email of the author of the plugin.
            github_username: A string representing the GitHub username of the author of the plugin.
            version: A string representing the version number of the plugin.
            cookietin_path: A str representing the path to the cookietin directory.

        Returns:
            None

        Raises:
            None
        """

        # Instantiate empty dictionary to store the dictionary to be converted to json
        self.jsonDic = {}

        # Create dic of characters to be replaced in namespace
        char_to_replace = {"[": "(", "]": ")", "'": "", " ": ""}

        # Iterate over all imagej libraries that were parsed
        for (
            name,
            namespace,
        ) in self._namespaces.items():

            # Check if any ops are suppported
            if len(namespace.supportedOps) > 0:

                # Add the json "template" for the library to the dictionary containing all library "templates"
                self.jsonDic[name] = {
                    "author": author,
                    "email": email,
                    "github_username": github_username,
                    "version": version,
                    "project_name": "ImageJ " + name.replace(".", " "),
                    "project_short_description": str(
                        [op for op in namespace.supportedOps.keys()]
                    ).replace("'", "")[1:-1],
                    "plugin_namespace": {
                        op.name: "out = ij.op()."
                        + op.namespace.replace(".", "().")
                        + re.sub(
                            r"[\s'\[\]]",
                            lambda x: char_to_replace[x.group(0)],
                            str(op.imagejTitleRequiredInputs),
                        )
                        for op in namespace.supportedOps.values()
                    },
                    "_inputs": {
                        "opName": {
                            "title": "Operation",
                            "type": "enum",
                            "options": [
                                op.name for op in namespace.supportedOps.values()
                            ],
                            "description": "Operation to peform",
                            "required": "False",
                        }
                    },
                    "_outputs": namespace._allOutputs,
                    "project_slug": "polus-{{ cookiecutter.project_name|lower|replace(' ', '-') }}-plugin",
                }

                # Update the _inputs section dictionary with the inputs dictionary stored in the Library attribute
                self.jsonDic[name]["_inputs"].update(namespace._allRequiredInputs)

                # Create Path object with directory path to store cookiecutter.json file for each namespace
                file_path = (
                    Path(cookietin_path)
                    .with_name("cookietin")
                    .joinpath(namespace._name.replace(".", "-"))
                )

                # Create the directory
                file_path.mkdir(exist_ok=True, parents=True)

                # Open the directory and place json file in directory
                with open(file_path.joinpath("cookiecutter.json"), "w") as fw:
                    json.dump(self.jsonDic[name], fw, indent=4)


"""This section is for testing only, the classes contained in this file were 
intended to be instantiated in generate.py"""

if __name__ == "__main__":

    import imagej
    import jpype
    from pathlib import Path

    # Disable warning message
    def disable_loci_logs():
        DebugTools = scyjava.jimport("loci.common.DebugTools")
        DebugTools.setRootLevel("WARN")

    scyjava.when_jvm_starts(disable_loci_logs)

    print("Starting JVM\n")

    # Start JVM
    ij = imagej.init(
        "sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4", headless=True
    )

    # Retreive all available operations from pyimagej
    # imagej_help_docs = scyjava.to_python(ij.op().help())
    # print(imagej_help_docs)

    print("Parsing imagej ops help\n")

    # Populate ops by parsing the imagej operations help
    populater = Populate(
        ij,
        logFile="full.log",
        logTemplate="utils/polus-imagej-util/classes/logtemplates/mainlog.txt",
    )

    print("Building json template\n")

    # Build the json dictionary to be passed to the cookiecutter module
    populater.buildJSON(
        "Benjamin Houghton",
        "benjamin.houghton@axleinfo.com",
        "bthoughton",
        "0.1.1",
        __file__,
    )

    print("Shutting down JVM\n")

    del ij

    # Shut down JVM
    jpype.shutdownJVM()
