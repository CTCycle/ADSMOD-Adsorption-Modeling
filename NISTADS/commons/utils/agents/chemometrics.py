# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables

import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger

###############################################################################
class AdsorbentMolecularProperties(BaseModel):
    smile: str
    molecular_formula: str
    molecular_weight: float

###############################################################################
class OpenAIMolecularProperties:

    def __init__(self, configuration):
        EV = EnvironmentVariables()
        self.openai_api_key = EV.get_openai_access_key()
        self.client = OpenAI(api_key=self.openai_api_key)        
        self.openai_model = configuration["collection"]["OPENAI_MODEL"]          
        self.configuration = configuration

    #--------------------------------------------------------------------------
    def ask_properties_to_LLM(self, adsorbent_name: str) -> AdsorbentMolecularProperties:
        # Prepare a prompt instructing the model to respond with only a JSON object
        # containing the SMILE code, the molecular formula and molecular weight 
        # of the adsorbent based on its name
        system_prompt = "You are a helpful chemistry assistant with knowledge about adsorbent materials"
        user_prompt = f"""
            Given the name or molecular formula of an adsorbent material (e.g. Zeolite 3A), 
            check if the material exists and then provide a JSON output with three keys (use 'NA' if does not exist): 
            - "smile": the canonical SMILE representation of the adsorbent material;
            - "molecular_formula": the molecular formula of the adsorbent material;
            - "molecular_weight": the molecular weight as a floating point number;                       

            Adsorbent material name or molecular formula: {adsorbent_name}
            """
        try:            
            completion = self.client.beta.chat.completions.parse(
                model=self.openai_model,
                messages=[
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": user_prompt}],
                temperature=0.2,
                timeout=5,
                response_format=AdsorbentMolecularProperties,
                max_completion_tokens=500,
                store=True)           
            
            # Attempt to parse the JSON from the response
            content = completion.choices[0].message.content
            properties = json.loads(content)            

            return properties

        except (json.JSONDecodeError, ValidationError) as e:          
            logger.error(f"Error parsing response: {e}")   
        except Exception as e:            
            logger.error(f"Error retrieving properties for {adsorbent_name}: {e}")
            