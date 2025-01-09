import requests
from huggingface_hub.inference_api import InferenceApi
import requests
from openai import OpenAI
import json
import os

def inference():

    client = OpenAI(
    # If the environment variable is not configured, replace the following line with: api_key="sk-xxx",
        api_key="sk-71a04cbb628c4c96bdf7d8da50d9ea18", 
        base_url="https://api.deepseek.com/v1",
    )

    def query(system_message, human_message):
        completion = client.chat.completions.create(
            model="deepseek-chat", # Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': human_message}],
        )
        return completion

    return query
