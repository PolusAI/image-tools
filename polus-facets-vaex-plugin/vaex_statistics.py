import vaex
import pandas as pd
import numpy as np

def MyProtoFromDataFrames(dict1):
    proto_dict = {}
    proto_dict['name'] = dict1['name']
    #: Count the # rows in the dataframe
    proto_dict['num_examples'] = dict1['table'].length_original()
    print(proto_dict)
    return proto_dict

def return_entries(df, i):
    """This function returns a list of dictoinaries  such as {'name': 'num'} for numeric values and {'name':'str', 'type': 'STRING'} for string values
    
    Args:
        df: vaex dataframe
        
    Returns:
        entry_dict: list of dictionaries [{'name': 'num'}, {'name':'str', 'type': 'STRING'}]
        """
    features = []
    for col in df.column_names:
        dtype = DtypeToType(df[col].dtype)
        if dtype == "STRING":
            entry_dict = {}
            entry_dict["name"] = col
            entry_dict["type"] = dtype
        else:
            entry_dict = {}
            entry_dict["name"] = col
        features.append(entry_dict)
    print(features[i]) #: TODO Remove hardcoding
    return features

def calc_mean_stdv_min_max_med(df, col):
    std_dev1 = df.std(col)
    mean_stats = df.mean(col)
    min_stats = df.min(col)
    max_stats = df.max(col)
    median_stats = df.median_approx(col)
    print("\nmean_stats: {} \nstd_dev: {}\nmin_stats: {}\nmax_stats: {}\nmedian_stats: {}".format(mean_stats, std_dev1, min_stats, max_stats, median_stats))
    return mean_stats, std_dev1, min_stats, max_stats, median_stats

def DtypeToType(dtype):
    fs_proto = None
    if str(dtype) == "string": 
        dtype = "str"
    dtype = np.dtype(str(dtype))
    if dtype.char in np.typecodes['AllFloat']:
        return "FLOAT"
    elif (dtype.char in np.typecodes['AllInteger'] or dtype == np.bool or np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64)):
        return "INT"
    else:
        return "STRING"

def gen_second_histogram(df, n_bins=10):
    """
    Args:
        df: vaex df
    Returns:
        list of dicts containing histogram edges and value counts
    """
    #: Get min/max for expressions, possibly on a grid defined by binby.
    x_min, x_max = df.minmax(df.num)
    n_bins = 10
    counts = df.count(binby=df.num, shape=n_bins, limits='minmax', edges=True)
    num_nan = counts[0]
    counts[-2] += counts[-1]
    #: Remove NaN and overflow counts
    counts = counts[2:-1]
    # linspace function produces a evenly spaced observations within a defined interval.
    #: Example np.linspace(start = 0, stop = 100, num = 5) --> [0, 25, 50, 75, 100]
    bin_edges=np.linspace(x_min, x_max, n_bins + 1)
    left, right = bin_edges[:-1], bin_edges[1:]
    #: Build buckets dict list
    buckets = []
    for i in range(0, len(counts)):
        bucket_dict = {}
        bucket_dict["low_value"] = left[i]
        bucket_dict["high_value"] = right[i]
        #: Omit sample_count if equal to 0, like facets does
        if counts[i] != 0:
            bucket_dict["sample_count"] = counts[i]
        buckets.append(bucket_dict) 
    print("\nSecond histogram: ")
    for bucket in buckets:
        print(bucket)   
    return buckets

def gen_third_histogram(df, nums=[1, 2, 3, 4, 3], n_bins=10): #: TODO Automatically remove NaNs from list and replace hardcoding. Ensure only doing for int values
    """Generate the quantiles histogram
    Args:
        df: vaex df
    Returns:
        list of dicts containing histogram edges and value counts
    """
    num_quantile_buckets = 10
    #quantiles_to_get = [x * 100 / num_quantile_buckets for x in range(num_quantile_buckets + 1)]
    sample_count = float(len(nums)) / num_quantile_buckets
    
    #: Get min/max for expressions, possibly on a grid defined by binby.
    x_min, x_max = df.minmax(df.num)
    n_bins = 10
    counts = df.count(binby=df.num, shape=n_bins, limits='minmax', edges=True)
    num_nan = counts[0]
    counts[-2] += counts[-1]
    #: Remove NaN and overflow counts
    counts = counts[2:-1]
    # linspace function produces a evenly spaced observations within a defined interval.
    #: Example np.linspace(start = 0, stop = 100, num = 5) --> [0, 25, 50, 75, 100]
    bin_edges=np.linspace(x_min, x_max, n_bins + 1)
    left, right = bin_edges[:-1], bin_edges[1:]
    #: Build buckets dict list
    buckets = []
    for i in range(0, len(counts)):
        bucket_dict = {}
        bucket_dict["low_value"] = left[i]
        bucket_dict["high_value"] = right[i]
        #: Omit sample_count if equal to 0, like facets does
        if counts[i] != 0:
            bucket_dict["sample_count"] = sample_count
        buckets.append(bucket_dict)   
    print("\nThird histogram: ") 
    for bucket in buckets:
        print(bucket)
    return buckets
 
def generate_unique(df):
    temp_stats = []
    for col in df.column_names:
        temp_entry_dict = {}
        if col == "str":
            unique = df[col].unique(dropmissing=True)
            temp_entry_dict["unique"] = len(unique)
        if temp_entry_dict != {}:
            temp_stats.append(temp_entry_dict)  
    print(temp_stats)
    return temp_stats 

def calc_missing(df):
    """TODO: This function is incomplete
    
    #Question: what is source of min_num_values: 1, max_num_values: 1, avg_num_values: 1.0?"""
    common_stats = []
    for col in df.column_names:
        temp_entry_dict = {}

        #: Return True where there are missing values (masked arrays), missing strings or None
        #: Uses df.x.isnan/isna/ismissing
        #: Source: https://github.com/vaexio/vaex/issues/146#issuecomment-526480738
        temp_df = df[col].ismissing()
        df.select(temp_df == True)
        num_missing = df.count(temp_df, selection=True)
        #print("num_missing", num_missing)
        #print(H)
        #num_non_missing = x[x == False].count()
        temp_df = df[col].ismissing()
        df.select(temp_df == False)
        num_non_missing = df.count(temp_df, selection=True)
        #print("num non missing: ", num_non_missing)
        temp_entry_dict["num_non_missing"] = num_non_missing

        if col == "str":
            temp_df = df[col].ismissing()
            df.select(temp_df == True)
            num_missing = df.count(temp_df, selection=True)
            #print("num_missing for str: ", num_missing)
            #temp_entry_dict["num_missing"] = num_missing

        #: TODO Remove hardcoding. Find vaex equivalent
        min_num_values = 1
        temp_entry_dict["min_num_values"] = min_num_values 

        max_num_values = 1 #: TODO remove hardcoding
        temp_entry_dict["max_num_values"] = max_num_values

        # Get avg num values
        lst = [min_num_values, max_num_values]
        avg_num_values = sum(lst) / len(lst)
        temp_entry_dict["avg_num_values"] = avg_num_values

        common_stats.append(temp_entry_dict)
    print(common_stats)
    return common_stats

#: Set up pandas df
print("\n###Converting Pandas Dataframe to Vaex...###\n")
df = pd.DataFrame({"num" : [1, 2, 3, 4], "str": ["a", "a", "b", None]})
df = vaex.from_pandas(df)

# datasets {name
# datasets {num_examples
proto_dict = MyProtoFromDataFrames({'name': 'test', 'table': df})

#: datasets {features {name
return_entries(df, 0)
#: datasets {features {num_stats {common_stats {num_non_missing **NUM STATS
# min_num_values
# max_num_values
# avg_num_values
# num_values_histogram

# mean, std_dev, min, max, median
temp_dict = {}   
for col in df.column_names:
    temp_dict["name"] = col
    dtype = DtypeToType(df[col].dtype)
    if dtype != "STRING":
        mean_stats, std_dev_stats, min_stats, max_stats, median_stats = calc_mean_stdv_min_max_med(df, col)

#: histograms1
#: histograms2
gen_second_histogram(df)
#: type
#: features {name
#: features {type
return_entries(df, 1)
#: string_stats {common_stats {
#: num_non_missing, num_missing, min_num_values, max_num_values, avg_num_values:
calc_missing(df)
#: num_values_histogram3
gen_third_histogram(df)
#: type
#: unique
generate_unique(df)
#: top values{value, frequency}
#: avg_length
#: rank_histogram