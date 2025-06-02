import sys
from typing import Callable

from llm import AnthropicClient, GoogleClient, OpenAIClient
from prompts import system_prompt, prompt_generic, prompt_templated


def chat_anthropic(user_message: str, system_instruction: str) -> str:
    client = AnthropicClient()
    return client.generate_text(user_message, system_instruction)


def chat_google(user_message: str, system_instruction: str) -> str:
    client = GoogleClient()
    return client.generate_text(user_message, system_instruction)


def chat_openai(user_message: str, system_instruction: str) -> str:
    client = OpenAIClient()
    return client.generate_text(user_message, system_instruction)


def chat_iterative(
    chat_fn: Callable,
    system_prompt: str,
    prompt: str,
    instructions: list[str],
    params: dict[str, str] | None = None,
) -> str:
    '''
    Send instructions one-by-one to an OpenAI model (LLM_MODEL)
    with a given system prompt prompt

    Params are injected into the prompt: genre, setting, characters

    Params in context:
        - genre: "...write part of a chapter in a {genre} novel."
        - setting: "The novel takes place {setting}."
        - characters: "The characters in the novel are:\n{characters}"

    Concatenate the completions as current_chapter and add them
    to the next prompt along with the next instruction

    Return the response as a string
    '''
    paragraphs: list[str] = []
    for i, instruction in enumerate(instructions):
        print(f'- processing instruction {i + 1}: {instruction}')
        current_chapter: str = '\n'.join(paragraphs)
        current_params = {'current_chapter': current_chapter, 'instruction': instruction}
        if params:
            current_params |= params
        current_prompt: str = prompt.format(**current_params)
        completion: str = chat_fn(current_prompt, system_prompt)
        completion = completion.strip()
        paragraphs.append(completion)
    print()
    chapter: str = '\n\n'.join(paragraphs)
    return chapter

# test instructions for the generic prompt
TEST_INSTRUCTIONS_1 = [
    "open with Mina and Elias trekking through a dense, misty forest at dawn, building a sense of mystery and wonder",
    "describe the ancient trees and the strange, hooting sounds that echo through the woods",
    "show Mina discovering a hidden, moss-covered stone with cryptic carvings, and her curiosity about its origin",
    "convey Elias's skepticism and protectiveness as he urges Mina to keep moving",
    "end with the sudden flash of bright light through the sky, creating suspense, and making them think there is trouble back at the base"
]

TEST_INSTRUCTIONS_2 = [
    "begin with Captain Lyra standing on the deck of her airship as it hovers above a sprawling, steampunk city",
    "use vivid language to capture the sights, sounds, and smells of the bustling city below",
    "show Lyra receiving a mysterious letter delivered by a mechanical bird, hinting at a secret mission",
    "describe the crew's reaction to the news and the tension it creates among them",
    "include a flashback to Lyra's first flight, revealing her motivation for becoming a captain",
    "have the airship encounter a rival vessel in the sky, leading to a tense standoff",
    "as thunder rumbles in the distance, Lyra makes a bold decision that surprises her crew"
]

TEST_INSTRUCTIONS_3 = [
    "open with Jamie nervously arriving at a quirky, bustling coffee shop for a blind date",
    "describe the awkward first moments as Jamie and Taylor realize they have met before under embarrassing circumstances",
    "show the barista making a playful comment that breaks the ice between them",
    "have Jamie accidentally spill coffee, leading to laughter and a shared moment of vulnerability",
    "include a flashback to their previous encounter at a disastrous wedding reception",
    "Taylor suggests a spontaneous game to lighten the mood, surprising Jamie",
    "describe the lively atmosphere of the coffee shop and the colorful regulars who eavesdrop on their conversation",
    "as the date progresses, Jamie and Taylor discover an unexpected shared passion",
    "the chapter closes with a mix-up involving their coffee orders, setting up a playful cliffhanger for their next meeting"
]

if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise Exception('please include desired provider as the single script arg')

    match sys.argv[1].lower():
        case 'anthropic':
            chat_fn = chat_anthropic
        case 'google':
            chat_fn = chat_google
        case 'openai':
            chat_fn = chat_openai
        case _:
            raise Exception(f'unknown provider: {sys.argv[1]}')

    print(f'using provider {sys.argv[1]}...')

    ############################### generic prompt ###############################
    # print(chat_iterative(chat_fn, system_prompt, prompt_generic, TEST_INSTRUCTIONS_1))
    # print(chat_iterative(chat_fn, system_prompt, prompt_generic, TEST_INSTRUCTIONS_2))
    # print(chat_iterative(chat_fn, system_prompt, prompt_generic, TEST_INSTRUCTIONS_3))

    ############################### templated prompt ###############################

    params_scifi = {
        'genre': 'science fiction',
        'setting': 'on a lush, green planet far from Earth in the year 2870',
        'characters': 'Mina and Elias, both twenty-something, completed high school and recently college, selected to go on a space mission to repopulate a faraway planet, but their ship crashed',
    }
    print(chat_iterative(chat_fn, system_prompt, prompt_templated, TEST_INSTRUCTIONS_1, params_scifi))

    print()
    print('*'*88)
    print()

    params_romcom = {
        'genre': 'romantic comedy',
        'setting': 'in a coffeeshop in New York City in the early 2000s',
        'characters': 'Jamie and Taylor, both twenty-something gay men, completed high school and in college at NYU. Jamie studies English and like to go out and dance. Taylor studies law and studies a lot, but also likes to party.',
    }
    print(chat_iterative(chat_fn, system_prompt, prompt_templated, TEST_INSTRUCTIONS_3, params_romcom))


