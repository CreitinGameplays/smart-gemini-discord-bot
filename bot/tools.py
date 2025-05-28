import asyncio
import aiohttp
import logging
import traceback
import textwrap
import io
import sys
import random
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from logging.handlers import RotatingFileHandler
from google.oauth2 import service_account
import os
import base64
import json
from bs4 import BeautifulSoup
from PIL import Image
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Constants
SEARCH_SNIPPET_SIZE = 6000
image_model_id = "imagen-3.0-fast-generate-001" # will be premium cuz damn this costs 0.03 bucks per image generated 😭

# ENV VARS
THE_CREDENTIALS = os.environ.get("GCP_CREDS") # required for Imagen 3
if not THE_CREDENTIALS:
    raise ValueError("Environment variable GCP_CREDS is not set")
decoded_credentials = base64.b64decode(THE_CREDENTIALS).decode("utf-8")
credentials_info = json.loads(decoded_credentials)
credentials = service_account.Credentials.from_service_account_info(credentials_info)

brave_token = os.getenv('BRAVE_TOKEN')
gcp_project = os.getenv('GCP_PROJECT')

uri = os.getenv('MONGO_URI')
if not uri:
    raise ValueError("MONGO_URI environment variable is not set.")
mongo_client = MongoClient(uri, server_api=ServerApi('1'))
db = mongo_client["gemini-bot-db"]

# Logging
handler = RotatingFileHandler(
    filename='bot_errors.log', mode='a', maxBytes=80 * 1024, backupCount=1, encoding='utf-8',
)
console_handler = logging.StreamHandler()  # Console output

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.ERROR)  # Set to ERROR level to log only errors
logger.addHandler(handler)
logger.addHandler(console_handler)

# Brave search
async def search_brave(search_query, session):
    url = f'https://api.search.brave.com/res/v1/web/search?q={search_query}'
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json',
        'X-Subscription-Token': brave_token
    }
    async with session.get(url, headers=headers, timeout=15) as response:
        if response.status != 200:
            return f'Error: Unable to fetch results (status code {response.status})'
        data = await response.json()
        results = data.get('web', {}).get('results', [])
        if not results:
            return 'Error: No search results found.'
        search_results = []
        for result in results:
            title = result.get('title', '')
            link = result.get('url', '')
            search_results.append({'title': title, 'link': link})
        return search_results

async def fetch_snippet(url, session, max_length=SEARCH_SNIPPET_SIZE):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        async with session.get(url, headers=headers, timeout=15) as response:
            if response.status != 200:
                return f'Error: Unable to fetch content from {url} (status code {response.status})'
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join([para.get_text() for para in paragraphs])
            if len(content) > max_length:
                return content[:max_length] + '‎ '
            else:
                return content
    except Exception as e:
        logger.error("An error occurred:\n" + traceback.format_exc())
        return f'Error: Unable to fetch content from {url} ({str(e)})'

async def browser(search_query: str, search_rn: int):
    search_rn = max(10, int(search_rn))
    try:
        async with aiohttp.ClientSession() as session:
            results = await search_brave(search_query, session)
            if not isinstance(results, list):
                return results
            limited_results = results[:search_rn]
            snippet_tasks = [fetch_snippet(result['link'], session) for result in limited_results]
            snippets = await asyncio.gather(*snippet_tasks)
            results_output = []
            for i, (result, snippet) in enumerate(zip(limited_results, snippets)):
                result_str = f'{i+1}. Title: {result["title"]}\nLink: {result["link"]}\nSnippet: {snippet}\n'
                results_output.append(result_str)
            return '\n'.join(results_output)
    except Exception as e:
        logger.error("An error occurred:\n" + traceback.format_exc())
        return f'Error in `search` function: {e}'

# Imagen 3
async def imagine(img_prompt: str, ar: str, author_id: int):
    vertexai.init(project=gcp_project, location="us-central1", credentials=credentials)
    img_info_var = {"is_error": 0, "img_error_msg": "null"}
    generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")
    user_settings = db.bot_settings.find_one({"user_id": author_id})
    is_donator = None
    try:
        if user_settings:
            is_donator = user_settings.get("is_donator", False)
            is_donator = bool(is_donator)
            if is_donator == False:
                error = "Image generation failed because the user is not a donator."
                img_info_var = {"is_error": 1, "img_error_msg": error}
                return img_info_var

        image_response = generation_model.generate_images(
            prompt=img_prompt,
            number_of_images=1,
            aspect_ratio=ar,
            safety_filter_level="block_some",
        )
        generated_image = image_response[0]
        image_bytes = generated_image._image_bytes
        image = Image.open(io.BytesIO(image_bytes))
        image_filename = f"output_{random.randint(1000, 9999)}.png"
        image.save(image_filename)
        img_info_var.update({"filename": image_filename})
        return img_info_var
    except Exception as e:
        img_info_var.update({"is_error": 1, "img_error_msg": f"{e}"})
        return img_info_var

# code execution
def exec_python(code):
    code = textwrap.dedent(code)
    print(code)
    buffer = io.StringIO()
    sys.stdout = buffer
    try:
        exec(code)
        output = buffer.getvalue()
        print('output:' + output)
        return output
    except Exception as e:
        return f"An error occurred: {e}"
    finally:
        sys.stdout = sys.__stdout__

# export
__all__ = [
    'search_brave',
    'fetch_snippet',
    'browser',
    'imagine',
    'exec_python'
]

# am i missing something?
# no, i think this is all we need for now
# alr