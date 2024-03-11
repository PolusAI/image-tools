"""Midrc Download."""

from pathlib import Path
from pydantic import BaseModel as V2BaseModel
import os
import logging
import json
from typing import Union
import pydantic
import os
import gen3
import requests
from pandas.errors import EmptyDataError

from gen3.auth import Gen3Auth
from gen3.query import Gen3Query
from gen3.submission import Gen3Submission
import pandas as pd
from polus.images.utils.midrc_download.utils import ENDPOINT

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


cred = os.environ.get("MIDRC_API_KEY")


class MIDRIC_download(V2BaseModel):
    """Advanced scripts for interacting with the Gen3 submission, query and index APIs

    Args:
        credentials: A Gen3Auth class instance.

    """
    credentials:str 

    @pydantic.field_validator("credentials")
    def validate_credentials(cls, value: Union[Path, str]) -> Union[Path, str]:
        if not Path(value).exists():
            msg = f"{value} do not exist! Please do check it again"
            raise ValueError(msg)
        with open(value, "r") as json_file:
            cred = json.load(json_file)
            if len(list(cred.values())) == 0 or list(cred.keys()) != ["api_key", "key_id"]:
                raise ValueError('Invalid API key')
        return value

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
    
    
    def download_request(self,
                        data_type,
                        fields,
                        accessibility=None):
        """Get a list of project_ids you have access to in a data commons.
        """
        url = f"{ENDPOINT}/guppy/download"
        
        if not accessibility:
            accessibility = "accessible"
        if not fields:
            fields = None
        try:
            body = {"type": data_type, "fields": fields, "accessibility":  "accessible"}
            response = requests.post(url,
                        json=body,
                        auth=self._authentication())
        
            if response.status_code != 200:
                return f"Error: Unexpected response {response}"
            data = response.json()
            
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"
        
        return data
    
    def quering_data(self, data_type, fields):
        """Get a list of project_ids you have access to in a data commons.
        """
        data = self.download_request(data_type, fields=None)
        data = pd.DataFrame(data)
        for col in list(data.columns):
            data = self.convert_float_to_str(data, col)  


        params = {"loinc_method":'CT',
          'loinc_system':'Chest',
          "body_part_examined":'CHEST', 
          "loinc_contrast":'W', 
          "study_modality":None}  
        
        params = {k:v for k, v in params.items() if v is not None}

        try:
            data.loc[data[list(params.keys())].isin(list(params.values())).all(axis=1), :]
            if data.empty:
                raise EmptyDataError
        except EmptyDataError:
            logger.error('dataframe query is empty!! Please search again')


        return data
    

    

        




        
    

        



model = MIDRIC_download(credentials=cred) 
data = model.download_request(data_type="imaging_study", fields=None)

print(data)
 
    




