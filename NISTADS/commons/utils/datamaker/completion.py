# [SETTING ENVIRONMENT VARIABLES]
from NISTADS.commons.variables import EnvironmentVariables

import openai
import json
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
        openai.api_key = EV.get_openai_access_key()
        self.openai_model = configuration["collection"]["OPENAI_MODEL"]          
        self.configuration = configuration

    #--------------------------------------------------------------------------
    def ask_properties_to_LLM(self, adsorbent_name: str) -> AdsorbentMolecularProperties:
        # Prepare a prompt instructing the model to respond with only a JSON object
        # containing the SMILE code and molecular weight of the adsorbent based on its name
        prompt = f"""
            Given an adsorbent material name, provide a JSON output with three keys:
            - "smile": the SMILE representation of the adsorbent material.
            - "molecular_formula": the molecular formula of the adsorbent material.
            - "molecular_weight": the molecular weight as a floating point number.

            Do not include any additional text or formatting.

            Adsorbent material: {adsorbent_name}
            """
        try:            
            completion = openai.beta.chat.completions.parse(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful chemistry assistant."},
                    {"role": "user", "content": prompt}],
                temperature=0,
                timeout=5,
                response_format=AdsorbentMolecularProperties)
            
            content = completion.choices[0].message.content
            
            # Attempt to parse the JSON from the response
            data = json.loads(content)

            # Validate and parse the JSON with the Pydantic model
            properties = AdsorbentMolecularProperties(**data)

            return properties

        except (json.JSONDecodeError, ValidationError) as e:          
            logger.error(f"Error parsing response: {e}")   
        except Exception as e:            
            logger.error(f"Error retrieving properties for {adsorbent_name}: {e}")
            