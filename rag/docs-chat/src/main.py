import logging
from contextlib import asynccontextmanager
from io import BytesIO
from os import getenv
from pathlib import Path
from shutil import rmtree
from typing import Annotated, Optional

from fastapi import FastAPI, Cookie, Header, Request, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from httpx import AsyncClient
from openai import APIConnectionError, AuthenticationError
from sqlalchemy.exc import NoResultFound
from sqlmodel import Session, select

from db import ApiKey, Chat, Message, Document, Sender, DB_ENGINE, DATA_DIR
from utils import (check_api_key, get_bot_response, get_document_context, save_to_file,
                   process_pdf_bytes, process_text_bytes, set_api_key, slugify, VECTOR_DB_CLIENT)

logging.basicConfig()
logging.getLogger().setLevel(getenv('LOGLEVEL', 'INFO'))
logger = logging.getLogger(__name__)

DOCS_PATH: Path = DATA_DIR / 'docs'
MAX_FILESIZE_MB: int = 100


@asynccontextmanager
async def lifespan(app: FastAPI):
    client: AsyncClient = AsyncClient()
    with Session(DB_ENGINE) as session:
        statement = select(ApiKey)
        results = session.exec(statement)
        api_keys = list(results)
        if api_keys:
            api_key = api_keys[0]
            set_api_key(api_key.cred)
        else:
            logger.info('No API key found in db')
    yield
    await client.aclose()

app: FastAPI = FastAPI(lifespan=lifespan)
app.mount('/static', StaticFiles(directory='static'), name='static')

def format_datetime(value):
    if not value:
        return ''
    return value.strftime('%m/%d/%Y %H:%M:%S')

templates: Jinja2Templates = Jinja2Templates(directory='templates')
templates.env.filters['format_datetime'] = format_datetime

@app.get('/')
async def index(request: Request, theme: Annotated[str | None, Cookie()] = None):
    context = {'request': request, 'theme': theme}
    return templates.TemplateResponse('index.html', context)


@app.post('/api-key')
async def api_key_set(request: Request, hx_request: Optional[str] = Header(None)):
    form = await request.form()
    api_key = str(form.get('api-key-input', ''))
    if hx_request and api_key:
        with Session(DB_ENGINE) as session:
            statement = select(ApiKey)
            results = session.exec(statement)
            try:
                existing_api_key = results.one()
                session.delete(existing_api_key)
            except NoResultFound:
                pass
            api_key_obj = ApiKey(cred=api_key)
            session.add(api_key_obj)
            session.commit()
            set_api_key(api_key)
            api_key_masked = 16 * '*' + api_key[-4:]
            return PlainTextResponse(api_key_masked)
    return PlainTextResponse('None')


@app.delete('/api-key')
async def api_key_delete(request: Request, hx_request: Optional[str] = Header(None)):
    if hx_request:
        with Session(DB_ENGINE) as session:
            statement = select(ApiKey)
            results = session.exec(statement)
            try:
                existing_api_key = results.one()
                session.delete(existing_api_key)
                session.commit()
            except NoResultFound:
                pass
    return PlainTextResponse('None')


@app.get('/api-key')
async def api_key_check(request: Request, hx_request: Optional[str] = Header(None)):
    if hx_request:
        with Session(DB_ENGINE) as session:
            statement = select(ApiKey)
            results = session.exec(statement)
            try:
                api_key_obj = results.one()
                api_key = api_key_obj.cred
                api_key_masked = 16 * '*' + api_key[-4:]
            except NoResultFound:
                api_key = api_key_masked = 'None'
        set_api_key(api_key)
        try:
            check_api_key()
        except (AuthenticationError, APIConnectionError) as e:
            return PlainTextResponse(api_key_masked + '<span class="api-key-check">❌</span>')
        return PlainTextResponse(api_key_masked + '<span class="api-key-check">✅</span>')


@app.get('/settings')
async def settings(request: Request, theme: Annotated[str | None, Cookie()] = None):
    with Session(DB_ENGINE) as session:
        statement = select(ApiKey)
        results = session.exec(statement)
        api_keys = list(results)
        try:
            api_key = api_keys[0]
        except IndexError:
            api_key = None
    api_key_masked = ''
    if api_key is not None:
        api_key_masked = 16 * '*' + api_key.cred[-4:]
    context = {'request': request, 'api_key_masked': api_key_masked, 'js_file': 'settings.js', 'css_file': 'settings.css', 'theme': theme}
    return templates.TemplateResponse('settings.html', context)


@app.get('/history')
async def history(request: Request, theme: Annotated[str | None, Cookie()] = None):
    with Session(DB_ENGINE) as session:
        statement = select(Chat).order_by(Chat.created_at.desc())
        results = session.exec(statement)
        chats = list(results)
        [chat.messages for chat in chats] # load messages for all chats
    context = {'request': request, 'chats': chats, 'js_file': 'history.js', 'css_file': 'history.css', 'theme': theme}
    return templates.TemplateResponse('history.html', context)


@app.get('/chat/new')
async def chat_new(request: Request, hx_request: Optional[str] = Header(None)):
    with Session(DB_ENGINE) as session:
        chat = Chat()
        session.add(chat)
        session.commit()
    return RedirectResponse('/chat', status_code=302)


@app.get('/chat/{chat_id}')
async def chat_data(request: Request, chat_id: Annotated[int, Path()], hx_request: Optional[str] = Header(None)):
    with Session(DB_ENGINE) as session:
        chat = session.get(Chat, chat_id)
        chat_data = ''
        if chat:
            for message in chat.messages:
                message_formatted = f'''
                <div class="chat-message { message.sender.lower() }">
                  <div class="message-bubble">
                    <p class="message-text">{ message.text }</p>
                    <!-- <span class="message-timestamp">9:30 AM</span> -->
                  </div>
                </div>
                '''
                chat_data += message_formatted
    return HTMLResponse(chat_data)


@app.get('/chat')
@app.post('/chat')
async def chat(request: Request, hx_request: Optional[str] = Header(None), theme: Annotated[str | None, Cookie()] = None):
    context = {'request': request, 'js_file': 'chat.js', 'css_file': 'chat.css', 'theme': theme}
    # load last conversation if one is present
    with Session(DB_ENGINE) as session:
        statement = select(Chat).order_by(Chat.created_at.desc()).limit(1)
        results = session.exec(statement)
        chats = list(results)
        if chats:
            chat = chats[0]
            context['chat'] = chat
            context['messages'] = list(chat.messages)
        else:
            return RedirectResponse('/chat/new', status_code=302)

        if hx_request:
            form = await request.form()
            user_message = str(form.get('user_message', '')).strip()
            if not user_message:
                return HTMLResponse('')

            document_context = get_document_context(user_message)
            bot_message = get_bot_response(user_message, document_context, chat)

            new_messages = [Message(text=user_message, sender=Sender.USER.name, chat_id=chat.id),
                            Message(text=bot_message, sender=Sender.BOT.name, chat_id=chat.id)]
            for m in new_messages:
                session.add(m)
            session.commit()
            session.refresh(chat)
            context['messages'] = list(chat.messages)
            return templates.TemplateResponse('chat_table.html', context)

    return templates.TemplateResponse('chat.html', context)


@app.get('/documents')
async def documents(request: Request, theme: Annotated[str | None, Cookie()] = None):
    with Session(DB_ENGINE) as session:
        statement = select(Document)
        results = session.exec(statement)
        documents = list(results)
    context = {'request': request, 'documents': documents, 'js_file': 'documents.js', 'css_file': 'documents.css', 'theme': theme}
    return templates.TemplateResponse('documents.html', context)


@app.post('/upload')
async def upload(request: Request, documents: list[UploadFile], background_tasks: BackgroundTasks, hx_request: Optional[str] = Header(None)):
    if not DOCS_PATH.exists():
        DOCS_PATH.mkdir(parents=True)
    total_bytes: float = 0
    with Session(DB_ENGINE) as session:
        for document in documents:
            file_bytes = BytesIO(await document.read())
            num_file_bytes = len(file_bytes.read())
            num_file_megabytes = round(float(num_file_bytes) / 1000000, 2)
            file_bytes.seek(0)

            if num_file_megabytes > MAX_FILESIZE_MB:
                continue
            total_bytes += num_file_megabytes

            if document.filename is None:
                continue

            if document.content_type == 'text/plain':
                doctype = 'txt'
            elif document.content_type == 'application/pdf':
                doctype = 'pdf'
            else:
                doctype = 'unknown'
                logger.error(f'file not uploaded: unknown document type for file: {document.filename}')
                continue

            # assume filename is normal (something.pdf), but slugify the name part in case of spaces (something else.pdf)
            filename_parts = document.filename.split('.')
            filename = slugify(filename_parts[:-1]) + '.' + filename_parts[-1]

            doc = Document(filename=filename, type=doctype)
            session.add(doc)
            session.commit()

            background_tasks.add_task(save_to_file, file_bytes, doc.filename, DOCS_PATH)
            if doc.type == 'txt':
                background_tasks.add_task(process_text_bytes, file_bytes)
            elif doc.type == 'pdf':
                background_tasks.add_task(process_pdf_bytes, file_bytes)

    return RedirectResponse('/documents', status_code=302)


@app.post('/clear')
async def clear(request: Request, hx_request: Optional[str] = Header(None)):
    with Session(DB_ENGINE) as session:
        statement = select(Document)
        results = session.exec(statement)
        for doc in list(results):
            session.delete(doc)
        session.commit()
    rmtree(str(DOCS_PATH), ignore_errors=True)
    DOCS_PATH.mkdir(parents=True)
    global VECTOR_DB_CLIENT
    VECTOR_DB_CLIENT.reset()
    return RedirectResponse('/documents', status_code=302)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True, log_level=getenv('LOGLEVEL', 'INFO').lower())
