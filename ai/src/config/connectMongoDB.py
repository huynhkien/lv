from pymongo import MongoClient
from src.config.config import settings

# Kết nối MongoDB Atlas
client = MongoClient(settings.MONGO_URI)

# Truy cập database "rynan"
db = client["sea_food"]

# Truy cập 2 collection
products_collection = db["products"]
warehouses_collection = db["warehouses"]
user_collection = db['users']
order_collection = db['orders']
receipt_collection = db['receipts']
