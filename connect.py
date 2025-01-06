import requests
from huggingface_hub.inference_api import InferenceApi
import requests
from openai import OpenAI
import json
import os

def inference():

    ACCESS_TOKEN = "sk-10efaa4d38dc4cee919b30b229e843a0"
    client = OpenAI(
    # If the environment variable is not configured, replace the following line with: api_key="sk-xxx",
        api_key="sk-10efaa4d38dc4cee919b30b229e843a0", 
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    def query(system_message, human_message):
        completion = client.chat.completions.create(
            model="qwen-turbo", # Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': human_message}],
        )
        return completion

    return query
