import os
from typing import Self
from dataclasses import dataclass

from anthropic import Anthropic
from google import genai
from google.genai.types import GenerateContentConfig
from openai import OpenAI
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from utils import retry


@dataclass
class AnthropicClient:
    '''
    Auth requires ANTHROPIC_API_KEY to be set in environment variables

    Python client: https://github.com/anthropics/anthropic-sdk-python

    Models: https://docs.anthropic.com/en/docs/about-claude/models/all-models
    '''
    model_version: str = 'claude-3-7-sonnet-20250219'
    model_temperature: float = 0.3
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def generate_text(self: Self, user_message: str, system_instruction: str) -> str:
        '''
        Generate LLM completion from a prompt
        '''
        # print(system_instruction)
        # print(user_message)
        def _generate_text(system_instruction: str, user_message: str) -> str:

            response = self.client.messages.create(
                model=self.model_version,
                temperature=self.model_temperature,
                system=system_instruction,
                # stop_sequences=['</category>'], # if there is any trouble getting a one word response, try this with a role: assistant message with content: <category>
                max_tokens=1024,
                messages=[
                    {'role': 'user', 'content': user_message},
                ]
            )
            return response.content[0].text.strip() or ''

        return retry(_generate_text, (system_instruction, user_message))


@dataclass
class GoogleClient:
    '''
    For whatever reason, Google has two APIs for Gemini with two different setups:
        Developer API (https://ai.google.dev/gemini-api/docs)
        Vertex AI API (https://cloud.google.com/vertex-ai/generative-ai/docs/)

    Vertex AI requirements:
        * Google Cloud SDK set up: https://cloud.google.com/sdk/docs/how-to
        * log in through the CLI
        * environment variables:
            - export GOOGLE_GENAI_USE_VERTEXAI=true
            - export GOOGLE_CLOUD_PROJECT='your-project-id'
            - export GOOGLE_CLOUD_LOCATION='your-cloud-region'

    Gemini Developer API requirements:
        * environment variables:
            - export GOOGLE_API_KEY='your-api-key'

    Python client docs: https://googleapis.github.io/python-genai/

    Gemini Developer API Models: https://ai.google.dev/api/models
    Vertex AI Models: https://cloud.google.com/vertex-ai/generative-ai/docs/models
    '''
    model_version: str = 'gemini-2.5-flash-preview-04-17'
    model_temperature: float = 1.3
    client: genai.Client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

    def generate_text(self: Self, user_message: str, system_instruction: str) -> str:
        '''
        Generate LLM completion from a prompt
        '''
        # print(system_instruction)
        # print(user_message)
        def _generate_text(system_instruction: str, user_message: str) -> str:
            response = self.client.models.generate_content(
                model=self.model_version,
                contents=[user_message],
                config=GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.model_temperature,
                ),
            )
            return response.text or ''
        return retry(_generate_text, (system_instruction, user_message))


@dataclass
class OpenAIClient:
    '''
    Auth requires OPENAI_API_KEY to be set in environment variables
    https://platform.openai.com/docs/api-reference/authentication

    Python client: https://github.com/openai/openai-python

    Models: https://platform.openai.com/docs/models
    '''
    model_version: str = 'gpt-4.1-2025-04-14'
    model_temperature: float = 1.3
    client: OpenAI = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def generate_text(self: Self, user_message: str, dev_message: str) -> str:
        '''
        Generate LLM completion from a prompt using the Responses API
        '''
        # print(dev_message)
        # print(user_message)
        def _generate_text(user_message: str, dev_message: str) -> str:
            messages: ResponseInputParam = [
                    {'role': 'developer', 'content': dev_message},
                    {'role': 'user', 'content': user_message}
            ]
            response = self.client.responses.create(
                model=self.model_version,
                input=messages,
                temperature=self.model_temperature,
            )
            return response.output_text

        return retry(_generate_text, (user_message, dev_message))

    def generate_text_completions_api(self: Self, user_message: str, dev_message: str) -> str:
        '''
        Generate LLM completion from a prompt using the Completions API
        '''
        # print(dev_message)
        # print(user_message)
        def _generate_text_completions_api(user_message: str, dev_message: str) -> str:
            '''
            Generate LLM completion from a prompt using the Completions API
            '''
            messages: list[ChatCompletionMessageParam] = [
                {'role': 'system', 'content': dev_message},
                {'role': 'user', 'content': user_message},
            ]
            # print(dev_message)
            # print(user_message)
            response=self.client.chat.completions.create(
                model=self.model_version,
                messages=messages,
                temperature=self.model_temperature,
            )
            return response.choices[0].message.content or ''

        return retry(_generate_text_completions_api, (user_message, dev_message))
