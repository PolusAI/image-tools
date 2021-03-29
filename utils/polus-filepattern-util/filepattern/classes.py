import copy, pathlib, typing, abc
from filepattern.functions import get_regex, get_matching, parse_directory, \
                                  parse_vector, logger, VARIABLES, output_name, \
                                  _parse, parse_filename

class PatternObject():
    """ Abstract base class for handling filepatterns
    
    Most of the functions in filepattern return complicated variable
    structures that might be difficult to use in an abstract way. This class
    provides tools to streamline usage of the filepattern functions. In
    particular, the iterate function is an iterable that permits simple
    iteration over filenames with specific values and grouped by any variable.
    """

    def __init__(self,
                 file_path: typing.Union[pathlib.Path,str],
                 pattern: str,
                 var_order: str = "rtczyxp"):
        """Initialize a Pattern object
        
        Args:
            file_path: Path to directory or file to parse
            pattern: A filepattern string
            var_order: Defines the dictionary nesting order. The list of
                characters is limited to :any:`VARIABLES`. *Defaults to
                "rtczyxp".*
        """
        self.files = {}
        self.uniques = {}
        
        # Define iteration variables
        self._kwargs = None
        self._group_by = None
        
        self.pattern = pattern
        self.regex, self.variables = get_regex(pattern)
        self.path = file_path

        self.var_order =  var_order
        
        self.var_order = "".join([v for v in self.var_order if v in self.variables])

        self.files, self.uniques = self.parse_data(file_path)
        
    def __call__(self,group_by: list = [],**kwargs) -> typing.Iterable[typing.List[dict]]:
        """Iterate through files parsed using a filepattern
        
        This function is an iterable. On each call, it returns a list of
        filenames that matches a set of variable values. It iterates through
        every combination of variable values.

        Variables designated in the group_by input argument are grouped
        together. So, if ``group_by="zc"``, then each iteration will return all
        filenames that have constant values for each variable except z and c.

        In addition to the group_by variable, specific variable arguments can
        also be included as with the :any:`get_matching` function.

        Args:
            group_by: String of variables by which the output filenames will be
                grouped
            **kwargs: Each keyword argument must be a valid uppercase letter
                from :any:`VARIABLES`. The value can be one integer or a list of
                integers.
                
        Returns:
            Iterable that returns a list of files with matching variables
        """
        
        self._group_by = group_by
        self._kwargs = kwargs
        
        return self
    
    @abc.abstractmethod
    def parse_data(self,file_path: str) -> dict:
        """Parse data in a directory
        
        This is where all the logic for the parsing the data should live. It
        must return a nested dictionary in the same format as
        :any:`parse_directory`.
            
        Args:
            file_path: Path to target file directory to parse
            
        Returns:
            A nested dictionary of file dictionaries
        """

    def output_name(self,files:typing.List[dict] = []) -> str:
        """Determine an output name for a list of files
        
        See the :any:`output_name` method for more details.
        
        This method uses the ``filepattern`` used to initialize the object to
        determine an output file name that summarizes the range of variables
        included in the ``file_path`` list of dictionaries. If ``file_path`` is
        empty, this method returns an output file name that summarizes the range
        of all variables parsed by the object.

        Args:
            files: A list of file dictionaries

        Returns:
            An output file name
        """
        
        if len(files) == 0:
            files = self.files
            files = get_matching(files,self.var_order,**{k.upper():v for k,v in self.uniques.items()})
        
        vals = {v:set() for v in self.var_order}
        for file in files:
            for k,v in file.items():
                if k not in self.var_order:
                    continue
                vals[k].add(v)
        
        kwargs = {}
        for k,v in vals.items():
            v = list(v)
            if len(v) == 1 and v[0] != -1:
                kwargs[k] = v[0]
                
        return output_name(self.pattern,files,kwargs)

    # Get filenames matching values for specified variables
    def get_matching(self,**kwargs):
        """ Get all filenames matching specific values
    
        This function runs the get_matching function using the objects file
        dictionary. For more information, see :any:`get_matching`.

        Args:
            **kwargs: One of :any:`VARIABLES`, must be uppercase, can be single
                values or a list of values
                
        Returns:
            A list of all files matching the input values
        """
        # get matching files
        files = get_matching(self.files,self.var_order,out_var=None,**kwargs)
        return files

    def __iter__(self):
        
        group_by = self._group_by
        kwargs = self._kwargs
        self._group_by = None
        self._kwargs = None
        
        if kwargs == None:
            kwargs = {}
            
        if group_by == None:
            group_by = ''

        # If self.files is a list, no parsing took place so just loop through the files
        if isinstance(self.files,list):
            for f in self.files:
                yield [f]
            return

        # Generate the values to iterate through
        iter_vars = {}
        for v in self.var_order:
            
            # Proceed to the next variable if v is not a grouping variable
            if v in group_by:
                continue
            
            # Check to see if the current variable has a matching value
            elif v.upper() in kwargs.keys():
                # If the value is a list, then we copy the list since we modify
                # it later
                if isinstance(kwargs[v.upper()],list):
                    iter_vars[v] = copy.deepcopy(kwargs[v.upper()])
                    
                # If the value is not a list, turn it into a list for consistent
                # access when looping over values
                else:
                    iter_vars[v] = [kwargs[v.upper()]]
            
            # If the variable is neither in group_by or kwargs, just copy the
            # dictionary or list since it gets modified later
            else:
                iter_vars[v] = copy.deepcopy(self.uniques[v])
        
        # Find the shallowest variable in the dictionary structure
        # Shallowest means the variable containing the list of file dictionaries
        shallowest = None
        for v in iter_vars.keys():
            
            # -1 indicates the variable doesn't exist in the file names
            if -1 in iter_vars[v] and len(iter_vars[v]):
                continue
            
            else:
                shallowest = v
                break

        # If shallowest is undefined, return all file names since no variables
        # were found in any of the file names
        if shallowest == None:
            yield get_matching(self.files,self.var_order,**{key.upper():iter_vars[key][0] for key in iter_vars.keys()})
            return

        # Loop through every combination of files
        while len(iter_vars[shallowest])>0:
            # Get list of filenames and return as iterator
            iter_files = []
            iter_files = get_matching(self.files,self.var_order,**{key.upper():iter_vars[key][0] for key in iter_vars.keys()})
            if len(iter_files)>0:
                yield iter_files

            # Delete last iteration indices
            for v in reversed(self.var_order):
                if v in group_by:
                    continue
                del iter_vars[v][0]
                if len(iter_vars[v])>0:
                    break
                elif v == shallowest:
                    break
                iter_vars[v] = copy.deepcopy(self.uniques[v])

class FilePattern(PatternObject):
    """ Main class for handling filename patterns
    
    Most of the functions in filepattern.py return complicated variable
    structures that might be difficult to use in an abstract way. This class
    provides tools to use the above functions in a  simpler way. In particular,
    the iterate function is an iterable that permits simple iteration over
    filenames with specific values and grouped by any desired variable.

    """
    
    def parse_data(self,file_path: typing.Union[pathlib.Path,str]) -> dict:
        """Parse data in a directory
        
        In the future, this function will parse data from a directory, and add
        it to the existing dictionary if it exists. For more information on how
        this method works, see :any:`parse_directory`.
            
        Args:
            file_path: Path to target file directory to parse
            
        Returns:
            A nested dictionary of file dictionaries
        """
        return parse_directory(file_path,regex=self.regex,variables=self.variables,var_order=self.var_order)
                
class VectorPattern(PatternObject):
    """ Main class for handling stitching vectors
    
    This class works nearly identically to :any:`FilePattern`, except it works
    with lines inside of a stitching vector. As with FilePattern, the iterate
    method will iterate through values, which in the case of VectorPattern are
    parsed lines of a stitching vector.
    
    Note:
        One major difference between this class and :any:`FilePattern` is that
        the ``file`` values in the file dictionaries contain strings rather than
        ``pathlib.Path`` objects.
        
    """
    
    def parse_data(self,file_path: typing.Union[pathlib.Path,str]):
        """Parse data in a directory
        
        In the future, this function will parse data from a directory, and add
        it to the existing dictionary if it exists. For more information on how
        this method works, see :any:`parse_vector`.
            
        Args:
            file_path: Path to target stitching vector to parse
            
        Returns:
            A nested dictionary of file dictionaries
        """
        return parse_vector(file_path,regex=self.regex,variables=self.variables,var_order=self.var_order)