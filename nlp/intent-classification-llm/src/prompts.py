from jinja2 import Environment, FunctionLoader

from utils import strip_indent

def render_template(name: str, content: dict):
    def _load_template(name):
        match name:
            case 'openai.user_message':
                return strip_indent('''
                    <user_message>
                    {{ user_message }}
                    </user_message>
                ''')
            case 'openai.zero_shot_prompt':
                return strip_indent('''
                    # Identity

                    You are a helpful assistant that labels user messages as one of these intents:
                    <intents>
                    {% for intent in intents %}
                    {{ intent }}
                    {% endfor %}
                    </intents>

                    # Instructions

                    * Only output a single word in your response with no additional formatting or commentary.
                    * Your response should only be one of the words from the <intents> block above, depending on the intent the user is expressing in the message.
                ''')
            case 'openai.k_shot_prompt':
                return strip_indent('''
                    # Identity

                    You are a helpful assistant that labels user messages as one of these intents:
                    <intents>
                    {% for intent in intents %}
                    {{ intent }}
                    {% endfor %}
                    </intents>

                    # Instructions

                    * Only output a single word in your response with no additional formatting or commentary.
                    * Your response should only be one of the words from the <intents> block above, depending on the intent the user is expressing in the message.

                    # Examples
                    {% for e in examples +%}
                    <user_message id="example-{{ loop.index }}">
                    {{ e['text'] }}
                    </user_message>

                    <assistant_response id="example-{{ loop.index }}">
                    {{ e['intent'] }}
                    </assistant_response>
                    {% endfor %}
                ''')
            case 'openai.zero_shot_cot_prompt':
                return strip_indent('''
                    # Identity

                    You are a helpful assistant that labels user messages as one of these intents:
                    <intents>
                    {% for intent in intents %}
                    {{ intent }}
                    {% endfor %}
                    </intents>

                    # Instructions

                    * Only output a single word in your response with no additional formatting or commentary.
                    * Your response should only be one of the words from the <intents> block above, depending on the intent the user is expressing in the message.

                    First think carefully about this step-by-step:
                    1. What is the user trying to do?
                    2. What are the key words that indicate the intent?
                    3. Which category best matches this intent?
                ''')
            case 'openai.k_shot_cot_prompt':
                return strip_indent('''
                    # Identity

                    You are a helpful assistant that labels user messages as one of these intents:
                    <intents>
                    {% for intent in intents %}
                    {{ intent }}
                    {% endfor %}
                    </intents>

                    # Instructions

                    * Only output a single word in your response with no additional formatting or commentary.
                    * Your response should only be one of the words from the <intents> block above, depending on the intent the user is expressing in the message.

                    # Examples
                    {% for e in examples +%}
                    <user_message id="example-{{ loop.index }}">
                    {{ e['text'] }}
                    </user_message>

                    <assistant_response id="example-{{ loop.index }}">
                    {{ e['intent'] }}
                    </assistant_response>
                    {% endfor %}

                    First think carefully about this step-by-step:
                    1. What is the user trying to do?
                    2. What are the key words that indicate the intent?
                    3. Which category best matches this intent?
                ''')
            case 'google.user_message':
                return strip_indent('''
                    USER MESSAGE: {{ user_message }}
                    INTENT:
                ''')
            case 'google.zero_shot_prompt':
                return strip_indent('''
                    You are a helpful assistant that labels the intent of user messages.
                    Your task is to classify the user message as one of the following intents:

                    <INTENTS>
                    {% for intent in intents %}
                    {{ intent }}
                    {% endfor %}
                    </INTENTS>

                    Only output a single word in your response with no additional formatting or commentary.
                    Your response should only be one of the words from the <INTENTS> block above, depending on the intent the user is expressing in the message.
                ''')
            case 'google.k_shot_prompt':
                return strip_indent('''
                    You are a helpful assistant that labels the intent of user messages.
                    Your task is to classify the user message as one of the following intents:

                    <INTENTS>
                    {% for intent in intents %}
                    {{ intent }}
                    {% endfor %}
                    </INTENTS>

                    Only output a single word in your response with no additional formatting or commentary.
                    Your response should only be one of the words from the <INTENTS> block above, depending on the intent the user is expressing in the message.

                    {% for e in examples +%}
                    <EXAMPLE>
                    USER MESSAGE: {{ e['text'] }}
                    INTENT: {{ e['intent'] }}
                    </EXAMPLE>
                    {% endfor %}
                ''')
            case 'google.zero_shot_cot_prompt':
                return strip_indent('''
                    You are a helpful assistant that labels the intent of user messages.
                    Your task is to classify the user message as one of the following intents:

                    <INTENTS>
                    {% for intent in intents %}
                    {{ intent }}
                    {% endfor %}
                    </INTENTS>

                    Only output a single word in your response with no additional formatting or commentary.
                    Your response should only be one of the words from the <INTENTS> block above, depending on the intent the user is expressing in the message.

                    First think carefully about this step-by-step:
                    1. What is the user trying to do?
                    2. What are the key words that indicate the intent?
                    3. Which category best matches this intent?
                ''')
            case 'google.k_shot_cot_prompt':
                return strip_indent('''
                    You are a helpful assistant that labels the intent of user messages.
                    Your task is to classify the user message as one of the following intents:

                    <INTENTS>
                    {% for intent in intents %}
                    {{ intent }}
                    {% endfor %}
                    </INTENTS>

                    Only output a single word in your response with no additional formatting or commentary.
                    Your response should only be one of the words from the <INTENTS> block above, depending on the intent the user is expressing in the message.

                    {% for e in examples +%}
                    <EXAMPLE>
                    USER MESSAGE: {{ e['text'] }}
                    INTENT: {{ e['intent'] }}
                    </EXAMPLE>
                    {% endfor %}

                    First think carefully about this step-by-step:
                    1. What is the user trying to do?
                    2. What are the key words that indicate the intent?
                    3. Which category best matches this intent?
                ''')
            case 'anthropic.user_message':
                return strip_indent('''
                    User Message: {{ user_message }}
                    Intent:
                ''')
            case 'anthropic.zero_shot_prompt':
                return strip_indent('''
                    You are an AI assistant trained to categorize user messages based on the intent expressed by the user. Your goal is to analyze each user message and classify it as one of the following intents:

                    <intents>
                    {% for intent in intents %}
                    {{ intent }}
                    {% endfor %}
                    </intents>

                    Only output a single word in your response with no additional formatting or commentary.
                    Your response should only be one of the words from the <intents> block above, depending on the intent the user is expressing in the message.
                ''')
            case 'anthropic.k_shot_prompt':
                return strip_indent('''
                    You are an AI assistant trained to categorize user messages based on the intent expressed by the user. Your goal is to analyze each user message and classify it as one of the following intents:

                    <intents>
                    {% for intent in intents %}
                    {{ intent }}
                    {% endfor %}
                    </intents>

                    Only output a single word in your response with no additional formatting or commentary.
                    Your response should only be one of the words from the <intents> block above, depending on the intent the user is expressing in the message.

                    <examples>
                    {% for e in examples +%}
                    <example>
                    User Message: {{ e['text'] }}
                    Intent: {{ e['intent'] }}
                    </example>
                    {% endfor %}

                    </examples>
                ''')
            case 'anthropic.zero_shot_cot_prompt':
                return strip_indent('''
                    You are an AI assistant trained to categorize user messages based on the intent expressed by the user. Your goal is to analyze each user message and classify it as one of the following intents:

                    <intents>
                    {% for intent in intents %}
                    {{ intent }}
                    {% endfor %}
                    </intents>

                    Only output a single word in your response with no additional formatting or commentary.
                    Your response should only be one of the words from the <intents> block above, depending on the intent the user is expressing in the message.

                    First think carefully about this step-by-step:
                    1. What is the user trying to do?
                    2. What are the key words that indicate the intent?
                    3. Which category best matches this intent?
                ''')
            case 'anthropic.k_shot_cot_prompt':
                return strip_indent('''
                    You are an AI assistant trained to categorize user messages based on the intent expressed by the user. Your goal is to analyze each user message and classify it as one of the following intents:

                    <intents>
                    {% for intent in intents %}
                    {{ intent }}
                    {% endfor %}
                    </intents>

                    Only output a single word in your response with no additional formatting or commentary.
                    Your response should only be one of the words from the <intents> block above, depending on the intent the user is expressing in the message.

                    <examples>
                    {% for e in examples +%}
                    <example>
                    User Message: {{ e['text'] }}
                    Intent: {{ e['intent'] }}
                    </example>
                    {% endfor %}

                    </examples>

                    First think carefully about this step-by-step:
                    1. What is the user trying to do?
                    2. What are the key words that indicate the intent?
                    3. Which category best matches this intent?
                ''')

    env = Environment(
        loader=FunctionLoader(_load_template),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(name)
    output = template.render(**content)
    return output

