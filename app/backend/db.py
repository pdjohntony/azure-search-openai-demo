import logging
from azure.cosmos import CosmosClient
import uuid
import datetime

class cosmosdb_client():
	def __init__(self, AZURE_DB_URL: str, AZURE_DB_KEY: str, AZURE_DB_NAME: str, AZURE_DB_CONTAINER: str):
		try:
			self.AZURE_DB_URL = AZURE_DB_URL
			self.AZURE_DB_KEY = AZURE_DB_KEY
			self.AZURE_DB_NAME = AZURE_DB_NAME
			self.AZURE_DB_CONTAINER = AZURE_DB_CONTAINER

			# Initialize Cosmos Client
			self.client = CosmosClient(AZURE_DB_URL, credential=AZURE_DB_KEY)

			# Select Cosmos database
			self.database = self.client.get_database_client(AZURE_DB_NAME)

			# Select Cosmos container
			self.container = self.database.get_container_client(AZURE_DB_CONTAINER)

			logging.debug(f"CosmosDB client initialized, container '{AZURE_DB_CONTAINER}' selected")
		except Exception as e:
			logging.exception(f"Error initializing CosmosDB client: {e}")
	
	def insert(self, user_email: str, user_query: str, bot_response: str):
		"""Inserts a chat history into the DB"""
		try:
			rid = str(uuid.uuid4())
			self.container.create_item(body={
				"id": rid,
				"user_email": user_email,
				"user_query": user_query,
				"bot_response": bot_response,
			})
			logging.debug(f"Inserted chat history for '{user_email}' into DB")
		except Exception as e:
			logging.exception(f"Error inserting DB Item: {e}")
	
	def select_recent(self, user_email: str, last_minutes: int) -> list:
		"""Selects the most recent chats for a user from the last x minutes

		Args:
				user_email (str): user@xyz.com
				last_minutes (int): 30

		Returns:
				recent_chat (list): a list of recent chats in ascending order
		"""
		try:
			recent_chat = []
			db_items = self.container.query_items(
				query=f"SELECT * FROM c WHERE c.user_email = '{user_email}' ORDER BY c._ts DESC OFFSET 0 LIMIT 4",
				enable_cross_partition_query=True
			)

			# Iterate items in reverse and append if within last_minutes
			for item in reversed(list(db_items)):
				# print(item)
				if datetime.datetime.fromtimestamp(item["_ts"]) > datetime.datetime.now() - datetime.timedelta(minutes=last_minutes):
					recent_chat.append({
						"user": item["user_query"],
						"bot": item["bot_response"],
					})
			
			logging.debug(f"Returning {len(recent_chat)} recent chats for '{user_email}' from the last {last_minutes} minutes")
			return recent_chat
		except Exception as e:
			logging.exception(f"Error while selecting recent chats for '{user_email}' from the last {last_minutes} minutes")
			return []