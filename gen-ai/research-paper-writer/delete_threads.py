'''
https://github.com/OlajideOgun/delete_openai_resources

You can't list threads through the OpenAI API. I took
this hacky script from the repo linked above and modified it.
It worked to list threads but not to delete them; now it
does both. I don't think it's absolutely necessary to delete
threads, but this script will do it if you want.
'''

import os

import requests
from openai import OpenAI

OPENAI_API_KEY = os.environ['OPENAI_API_KEY'] # https://platform.openai.com/api-keys
# OPENAI_ORG_ID = os.environ['OPENAI_ORG_ID']
OPENAI_ORG_ID = "org-wWDwi3RMz4AmKc2ghueVzaNA" # https://platform.openai.com/settings/organization/general
# OPENAI_PROJECT_ID = os.environ['OPENAI_PROJECT_ID']
OPENAI_PROJECT_ID = "proj_y9xUzdXEexElT5L09mEhVE6P" # https://platform.openai.com/settings/organization/projects
# OPENAI_SESSION_KEY = os.environ['OPENAI_SESSION_KEY']
OPENAI_SESSION_KEY = "sess-YS5XltVnf8s7Y2PbFPMTmaOid7NCZhHKJGzZ2p1W"
# for session key:
#  * Log in to OpenAI's platform and open the browser developer tools
#  * Go to the 'Network' tab and filter for fetch or XHR requests
#  * Refresh the page and look for a request related to threads
#  * Find the Authorization or session-related key in the request headers

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta": "assistants=v2"
}

SESSION_HEADERS = {
    "Authorization": f"Bearer {OPENAI_SESSION_KEY}",
    "OpenAI-Beta": "assistants=v2",
    "OpenAI-Organization": OPENAI_ORG_ID,
    "OpenAI-Project": OPENAI_PROJECT_ID,
    "Origin": "https://platform.openai.com",
    "Referer": "https://platform.openai.com/"
}

TIMEOUT = 30

def list_threads(limit=100, after=None):
    """Get a list of threads from OpenAI API with pagination."""
    all_threads = []
    has_more = True

    while has_more:
        url = "https://api.openai.com/v1/threads"
        params = {"limit": limit}
        if after:
            params["after"] = after

        # Use SESSION_HEADERS instead of HEADERS for thread operations
        response = requests.get(url, headers=SESSION_HEADERS, params=params, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            threads = data.get("data", [])
            all_threads.extend(threads)

            # Check if there are more results
            has_more = data.get("has_more", False)
            if has_more and threads:
                after = threads[-1]["id"]
            else:
                break
        else:
            print(f"Failed to list threads: {response.status_code}, {response.text}")
            break

    return all_threads


if __name__ == "__main__":

    threads = list_threads()

    client: OpenAI = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    for thread in threads:
        client.beta.threads.delete(thread['id'])
