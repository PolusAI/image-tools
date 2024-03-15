"""Midrc Download."""

from pathlib import Path
import os
import logging
import os
import requests
from typing import Optional
from gen3.auth import Gen3Auth
import pandas as pd
from typing import Any
from polus.images.utils.midrc_download.utils import ENDPOINT, CustomValidation
import subprocess
import itertools
import preadator
from multiprocessing import cpu_count
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


cred = os.environ.get("MIDRC_API_KEY")

num_workers = max([cpu_count(), 2])

class MIDRIC_download(CustomValidation):
    """Advanced scripts for interacting with the Gen3 submission, query and index APIs

    Args:
        credentials: A Gen3Auth class instance.

    """

    credentials:str
    data_type:str
    project_id: Optional[Any] = None
    sex:Optional[Any] = None
    race:Optional[Any]= None
    ethnicity:Optional[Any] = None
    min_age:Optional[int] = 0
    max_age:Optional[int] = 89
    study_modality:Optional[Any] = None
    body_part_examined:Optional[Any]= None
    loinc_contrast:Optional[Any] = None
    loinc_method:Optional[Any]= None
    loinc_system:Optional[Any] = None
    covid19_positive:Optional[Any]=  None
    out_dir:Path
 
    def _authentication(self) -> object:
        auth = Gen3Auth(ENDPOINT, refresh_file=self.credentials)
        return auth
    
    def convert_float_to_str(x:pd.DataFrame, feature:str):
        feat_lst = list(x[feature].values)
        feature_values = []
        string = ''
        for s in feat_lst:
            if isinstance(s, float):
                string = str(s)
            if isinstance(s, list):
                string= ' '.join([str(item) for item in s])
            if isinstance(s, str):
                string = s
            feature_values.append(string)
        x[feature] = feature_values
        return x
    
    def get_query(self, x):
        my_dict = {k: v for k, v in x.items() if not k in ["credentials", "data_type"]}
        fn = []
        for k, v in  my_dict.items():
            if (k == "min_age") and isinstance(v, int):
                k = "age_at_index"
                fn_dict = {">=": {k:v}}
            if (k == "max_age") and isinstance(v, int):
                k = "age_at_index"
                fn_dict = {"<=": {k:v}}  
            if (k == "study_year") and isinstance(v, int):
                fn_dict = {">=": {k:v}}
    
            if not k in ["age_at_index", "study_year"]  and isinstance(v, str):
                fn_dict = {"=": {k:v}}
            if not k in ["age_at_index", "study_year"] and isinstance(v, list) and len(v) > 1:
                fn_dict = {"IN": {k:v}}  
            fn.append(fn_dict)
        my_dict = {"AND": fn}

        return my_dict
     
    def download_request(self,
                        data_type,
                        fields,
                        filter_object=None,
                        sort_fields=None,
                        accessibility=None,
                        first=None,
                        offset=None):
        """Get a list of project_ids you have access to in a data commons.
        """
        url = f"{ENDPOINT}/guppy/download"
        
        if not accessibility:
            accessibility = "accessible"
        if not fields:
            fields = None
        try:
            body = {"type": data_type, "fields": fields, "accessibility":  accessibility}
            if filter_object:
                body["filter"] = filter_object
            if sort_fields:
                body["sort"] = sort_fields
            response = requests.post(url,
                        json=body,
                        auth=self._authentication())
        
            if response.status_code != 200:
                return f"Error: Unexpected response {response}"
            data = response.json()
            
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"
        
        if offset:
            data = data[offset:]
        if first:
            data = data[:first]


        if len(data) > 0:
            logger.info(f"Successfully fetched records!!")
            return data
        else:
            raise ValueError(f"Unable to fetch records!! Please query again")
     
    
    def download_data(self, data):
        ## Simple loop to download all files and keep track of success and failures
        if len(data ) > 0:
            object_ids = [i['object_id'] for i in data]
            object_ids= list(itertools.chain.from_iterable(object_ids))

            with preadator.ProcessManager(
                name="Downloading Midrc data",
                num_processes=num_workers,
                threads_per_process=2,
            ) as pm:
                count= 0
                total = len(data)
                for object_id in tqdm(object_ids, desc="Progress", total=len(data)):
                    count+=1
                    cmd = f"gen3 --auth {self.credentials} --endpoint data.midrc.org drs-pull object {object_id} --output-dir {self.out_dir}"
                    pm.submit_process(subprocess.run, cmd, shell=True, capture_output=True)
                    logger.info("Progress ({}/{})".format(count,total))

                pm.join_processes()


                
        # else:
        #     print("Your query returned no data! Please, check that query parameters are valid.") 

        # success=[]
        # fail_queries=[]
        # other=[]
        # count,total = 0,len(data)
        # for object_id in object_ids:
        #     count+=1
        #     cmd = f"gen3 --auth {self.credentials} --endpoint data.midrc.org drs-pull object {object_id} --output-dir {self.out_dir}"
        #     os.system(cmd) 
        #     stout = subprocess.run(cmd, shell=True, capture_output=True)
        #     print("Progress ({}/{}): {}".format(count,total,stout.stdout))
        #     if "failed" in str(stout.stdout):
        #         fail_queries.append(object_id)
        #     elif "successfully" in str(stout.stdout):
        #         success.append(object_id)
        #     else:
        #         other.append(object_id)

        return     

    




