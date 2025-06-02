import os
from typing import Self
from dataclasses import dataclass, field

from anthropic import Anthropic
from google import genai
from google.genai.types import GenerateContentConfig
from openai import OpenAI
from openai.types.responses.response_input_param import ResponseInputParam

from prompts import render_template
from utils import retry


@dataclass
class AnthropicClient:
    '''
    Auth requires ANTHROPIC_API_KEY to be set in environment variables

    Python client: https://github.com/anthropics/anthropic-sdk-python

    Models: https://docs.anthropic.com/en/docs/about-claude/models/all-models
    '''
    model_version: str = 'claude-3-7-sonnet-20250219'
    model_temperature: float = 0.0
    intents: list[str] = field(default_factory=list)
    examples: list[dict[str, str]] = field(default_factory=list)  # [{'text': str, 'intent': str}, ...]
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def generate_text(self: Self, user_message: str, template_name: str) -> str:
        '''
        Generate LLM completion from a prompt
        '''
        system_instruction = render_template(name=template_name, content={'examples': self.examples, 'intents': self.intents})
        user_message_formatted = render_template(name='anthropic.user_message', content={'user_message': user_message})
        # print(system_instruction)
        # print(user_message_formatted)
        def _generate_text(system_instruction: str, user_message_formatted: str) -> str:

            response = self.client.messages.create(
                model=self.model_version,
                temperature=self.model_temperature,
                system=system_instruction,
                # stop_sequences=['</category>'], # if there is any trouble getting a one word response, try this with a role: assistant message with content: <category>
                max_tokens=20,
                messages=[
                    {'role': 'user', 'content': user_message_formatted},
                ]
            )
            return response.content[0].text.strip() or ''

        return retry(_generate_text, (system_instruction, user_message_formatted))


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
    model_temperature: float = 1.0
    intents: list[str] = field(default_factory=list)
    examples: list[dict[str, str]] = field(default_factory=list)  # [{'text': str, 'intent': str}, ...]
    client: genai.Client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

    def generate_text(self: Self, user_message: str, template_name: str) -> str:
        '''
        Generate LLM completion from a prompt
        '''
        system_instruction = render_template(name=template_name, content={'examples': self.examples, 'intents': self.intents})
        user_message_formatted = render_template(name='google.user_message', content={'user_message': user_message})
        # print(system_instruction)
        # print(user_message_formatted)
        def _generate_text(system_instruction: str, user_message_formatted: str) -> str:
            response = self.client.models.generate_content(
                model=self.model_version,
                contents=[user_message_formatted],
                config=GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.model_temperature,
                ),
            )
            return response.text or ''
        return retry(_generate_text, (system_instruction, user_message_formatted))


@dataclass
class OpenAIClient:
    '''
    Auth requires OPENAI_API_KEY to be set in environment variables
    https://platform.openai.com/docs/api-reference/authentication

    Python client: https://github.com/openai/openai-python

    Models: https://platform.openai.com/docs/models
    '''
    model_version: str = 'gpt-4.1-2025-04-14'
    model_temperature: float = 1.0
    intents: list[str] = field(default_factory=list)
    examples: list[dict[str, str]] = field(default_factory=list)  # [{'text': str, 'intent': str}, ...]
    client: OpenAI = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def generate_text(self: Self, user_message: str, template_name: str) -> str:
        '''
        Generate LLM completion from a prompt
        '''
        dev_message = render_template(name=template_name, content={'examples': self.examples, 'intents': self.intents})
        user_message_formatted = render_template(name='openai.user_message', content={'user_message': user_message})
        messages: ResponseInputParam = [
                # developer messages are roughtly equivalent to the `instructions` param in the response.create call
                {'role': 'developer', 'content': dev_message},
                {'role': 'user', 'content': user_message_formatted}
        ]
        # print(dev_message)
        # print(user_message_formatted)
        def _generate_text(messages: ResponseInputParam) -> str:
            response = self.client.responses.create(
                model=self.model_version,
                input=messages,
                temperature=self.model_temperature,
            )
            return response.output_text

        return retry(_generate_text, (messages,))
