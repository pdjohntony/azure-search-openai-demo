import os
import mimetypes
import time
import logging
import logging.handlers
import openai
import re
from urllib.parse import quote
from flask import Flask, request, jsonify
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from approaches.retrievethenread import RetrieveThenReadApproach
from approaches.readretrieveread import ReadRetrieveReadApproach
from approaches.readdecomposeask import ReadDecomposeAsk
from approaches.chatreadretrieveread import ChatReadRetrieveReadApproach
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from db import cosmosdb_client

# Replace these with your own values, either in environment variables or directly here
BACKEND_URI = os.environ.get("BACKEND_URI") or "http://localhost:5000"
AZURE_STORAGE_ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT") or "mystorageaccount"
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER") or "content"
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE") or "gptkb"
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX") or "gptkbindex"
AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE") or "myopenai"
AZURE_OPENAI_GPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT") or "davinci"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT") or "chat"

AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AZURE_STORAGE_KEY = os.environ.get("AZURE_STORAGE_KEY")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")

KB_FIELDS_CONTENT = os.environ.get("KB_FIELDS_CONTENT") or "content"
KB_FIELDS_CATEGORY = os.environ.get("KB_FIELDS_CATEGORY") or "category"
KB_FIELDS_SOURCEPAGE = os.environ.get("KB_FIELDS_SOURCEPAGE") or "sourcepage"

AZURE_DB_URL = os.environ.get("AZURE_DB_URL")
AZURE_DB_KEY = os.environ.get("AZURE_DB_KEY")
AZURE_DB_NAME = os.environ.get("AZURE_DB_NAME")
AZURE_DB_CONTAINER = os.environ.get("AZURE_DB_CONTAINER")

CHAT_HISTORY_DB_MIN = os.environ.get("CHAT_HISTORY_DB_MIN") or "5"

# LOGGING
LOG_FILE = "app.log"
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# Console handler with INFO level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
# File handler with DEBUG level
file_handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=4, encoding='utf8') # 10 Mb, 0 backup files
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('[{asctime}] [{module}] [{levelname:<7}]: {message}', style='{')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
# http.client.HTTPConnection.debuglevel = 0
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
# set openai util to warning

logger.debug("""

 #       ####   ####      ####  #####   ##   #####  ##### 
 #      #    # #    #    #        #    #  #  #    #   #   
 #      #    # #          ####    #   #    # #    #   #   
 #      #    # #  ###         #   #   ###### #####    #   
 #      #    # #    #    #    #   #   #    # #   #    #   
 ######  ####   ####      ####    #   #    # #    #   #   
""")

# Use the current user identity to authenticate with Azure OpenAI, Cognitive Search and Blob Storage (no secrets needed, 
# just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the 
# keys for each service
# If you encounter a blocking error during a DefaultAzureCredntial resolution, you can exclude the problematic credential by using a parameter (ex. exclude_shared_token_cache_credential=True)
azure_credential = DefaultAzureCredential()

# Use the Azure keys if provided, otherwise use DefaultAzureCredential
if AZURE_SEARCH_KEY:
    AZURE_SEARCH_CREDENTIAL = AzureKeyCredential(AZURE_SEARCH_KEY)
else:
    AZURE_SEARCH_CREDENTIAL = azure_credential
if AZURE_STORAGE_KEY:
    AZURE_STORAGE_CREDENTIAL = AZURE_STORAGE_KEY
else:
    AZURE_STORAGE_CREDENTIAL = azure_credential

# Used by the OpenAI SDK
openai.api_type = "azure"
openai.api_base = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
openai.api_version = "2022-12-01"

if AZURE_OPENAI_KEY:
    openai.api_key = AZURE_OPENAI_KEY
else:
    openai.api_type = "azure_ad"
    openai_token = azure_credential.get_token("https://cognitiveservices.azure.com/.default")
    openai.api_key = openai_token.token

# Set up clients for Cognitive Search and Storage
search_client = SearchClient(
    endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
    index_name=AZURE_SEARCH_INDEX,
    credential=AZURE_SEARCH_CREDENTIAL)
blob_client = BlobServiceClient(
    account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net", 
    credential=AZURE_STORAGE_CREDENTIAL)
blob_container = blob_client.get_container_client(AZURE_STORAGE_CONTAINER)

# Various approaches to integrate GPT and external knowledge, most applications will use a single one of these patterns
# or some derivative, here we include several for exploration purposes
ask_approaches = {
    "rtr": RetrieveThenReadApproach(search_client, AZURE_OPENAI_GPT_DEPLOYMENT, KB_FIELDS_SOURCEPAGE, KB_FIELDS_CONTENT),
    "rrr": ReadRetrieveReadApproach(search_client, AZURE_OPENAI_GPT_DEPLOYMENT, KB_FIELDS_SOURCEPAGE, KB_FIELDS_CONTENT),
    "rda": ReadDecomposeAsk(search_client, AZURE_OPENAI_GPT_DEPLOYMENT, KB_FIELDS_SOURCEPAGE, KB_FIELDS_CONTENT)
}

chat_approaches = {
    "rrr": ChatReadRetrieveReadApproach(search_client, AZURE_OPENAI_CHATGPT_DEPLOYMENT, AZURE_OPENAI_GPT_DEPLOYMENT, KB_FIELDS_SOURCEPAGE, KB_FIELDS_CONTENT)
}

db = cosmosdb_client(AZURE_DB_URL, AZURE_DB_KEY, AZURE_DB_NAME, AZURE_DB_CONTAINER)

# Add MIME types for JavaScript and CSS files to prevent Flask from serving them as text/plain on Windows
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')

def convert_to_md_link(text):
    # Convert [filename] inside a string to [filename](link) for markdown rendering
    try:
        pattern = re.compile(r'\[(.*?)\]')
        matches = pattern.findall(text)
        for match in matches:
            link = quote(match)
            text = text.replace(f'[{match}]', f'[{match}]({BACKEND_URI}/content/{link})')
        return text
    except Exception as e:
        logger.error(f"Error converting text to markdown: {e}")
        return text

app = Flask(__name__)

@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def static_file(path):
    return app.send_static_file(path)

# Serve content files from blob storage from within the app to keep the example self-contained. 
# *** NOTE *** this assumes that the content files are public, or at least that all users of the app
# can access all the files. This is also slow and memory hungry.
@app.route("/content/<path>")
def content_file(path):
    blob = blob_container.get_blob_client(path).download_blob()
    mime_type = blob.properties["content_settings"]["content_type"]
    if mime_type == "application/octet-stream":
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return blob.readall(), 200, {"Content-Type": mime_type, "Content-Disposition": f"inline; filename={path}"}
    
@app.route("/ask", methods=["POST"])
def ask():
    ensure_openai_token()
    approach = request.json["approach"]
    try:
        impl = ask_approaches.get(approach)
        if not impl:
            return jsonify({"error": "unknown approach"}), 400
        r = impl.run(request.json["question"], request.json.get("overrides") or {})
        return jsonify(r)
    except Exception as e:
        logging.exception("Exception in /ask")
        return jsonify({"error": str(e)}), 500
    
@app.route("/chat", methods=["POST"])
def chat():
    ensure_openai_token()
    try:
        logger.info(f"INCOMING /chat Request body: {request.get_data()}")
        user_email = request.json.get("user_email")
        approach = request.json["approach"]
        impl = chat_approaches.get(approach)
        if not impl:
            return jsonify({"error": "unknown approach"}), 400
        # if user_email is provided, query the DB for recent history and append it to the request history
        if user_email != None and len(request.json["history"]) <= 1:
            logger.debug(f"Previous chat history not present in request, querying recent history from DB")
            db_chat_history = db.select_recent(user_email=user_email, last_minutes=int(CHAT_HISTORY_DB_MIN))
            history = db_chat_history + request.json["history"]
        else:
            history = request.json["history"]
        r = impl.run(history, request.json.get("overrides") or {})
        resp = r
        resp["history"] = history # return the original history
        history[-1]["bot"] = r["answer"] # add the bot's answer to the history
        if user_email: db.insert(user_email=user_email, user_query=request.json["history"][-1]["user"], bot_response=r["answer"])
        resp["answer"] = convert_to_md_link(resp["answer"])
        logger.info(f"OUTGOING /chat Response body: {resp}")
        return jsonify(resp)
    except Exception as e:
        logging.exception("Exception in /chat")
        return jsonify({"error": str(e)}), 500

def ensure_openai_token():
    global openai_token
    if not AZURE_OPENAI_KEY:
        if openai_token.expires_on < int(time.time()) - 60:
            openai_token = azure_credential.get_token("https://cognitiveservices.azure.com/.default")
            openai.api_key = openai_token.token
    
if __name__ == "__main__":
    app.run()
