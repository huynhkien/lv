import random
import json
import torch
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.config.config import settings
from sklearn.preprocessing import LabelEncoder
import tempfile
from PIL import Image
import io
import os
from tensorflow.keras.models import load_model
import pandas as pd
from src.utils import price
from datetime import datetime, timedelta
from src.utils.helper import parse_date_from_query, parse_month_from_query, parse_year_from_query, format_message
import json

# Nhập các mô-đun từ thư mục ai
from src.config.config import settings
from src.config.connectMongoDB import warehouses_collection, products_collection, user_collection, order_collection
from pydantic import BaseModel
import re
from collections import Counter
from bson import ObjectId
from datetime import datetime, timedelta

# Nhập các mô-đun từ thư mục ai
from chatbot_lstm.train import ChatbotLSTM, ChatbotPredictor
from chatbot_lstm.utils import TextPreprocessor

app = FastAPI()


# Setting CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.CLIENT_URL,
    ],  # Allow URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"], 
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Định nghĩa model cho request
class ChatRequest(BaseModel):
    message: str
@app.get("/")
async def run_server():
    try:
        return JSONResponse(content={"status": "run server"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "server is failed", "error": str(e)}, status_code=500)
# Load model 1  
with open('chatbot_lstm/intents.json', 'r', encoding='utf-8') as json_data:
    intents_data = json.load(json_data)
    
def load_model_predictor(path="chatbot_lstm/chatbot_model.pth"):
    data = torch.load(path, map_location=torch.device('cpu'))

    # Khởi tạo lại mô hình
    model = ChatbotLSTM(
        vocab_size=data["vocab_size"],
        embedding_dim=data["embedding_dim"],
        hidden_size=data["hidden_size"],
        num_layers=data["num_layers"],
        num_classes=data["num_classes"],
        dropout=0.5
    )
    model.load_state_dict(data["model_state"])
    model.eval()

    # Tạo preprocessor từ dữ liệu đã lưu
    preprocessor = TextPreprocessor(
        max_vocab_size=data["vocab_size"],
        max_seq_len=data["max_seq_len"]
    )
    preprocessor.word2idx = data["word2idx"]
    preprocessor.tag2idx = data["tag2idx"]
    preprocessor.idx2tag = data["idx2tag"]

    predictor = ChatbotPredictor(model, preprocessor, device='cpu')
    return predictor
# Load model 2    
with open('chatbot_lstm/intents_chatbot.json', 'r', encoding='utf-8') as json_data:
    intents_chatbot_data = json.load(json_data)
    
def load_model_2_predictor(path="chatbot_lstm/chatbot_public_model.pth"):
    data = torch.load(path, map_location=torch.device('cpu'))

    # Khởi tạo lại mô hình
    model = ChatbotLSTM(
        vocab_size=data["vocab_size"],
        embedding_dim=data["embedding_dim"],
        hidden_size=data["hidden_size"],
        num_layers=data["num_layers"],
        num_classes=data["num_classes"],
        dropout=0.5
    )
    model.load_state_dict(data["model_state"])
    model.eval()

    # Tạo preprocessor từ dữ liệu đã lưu
    preprocessor = TextPreprocessor(
        max_vocab_size=data["vocab_size"],
        max_seq_len=data["max_seq_len"]
    )
    preprocessor.word2idx = data["word2idx"]
    preprocessor.tag2idx = data["tag2idx"]
    preprocessor.idx2tag = data["idx2tag"]

    predictor = ChatbotPredictor(model, preprocessor, device='cpu')
    return predictor

predictor = load_model_predictor()
predictor_1 = load_model_2_predictor()
@app.post("/chat")
async def chat(request: ChatRequest):
    message = format_message(request.message)
    tag, confidence, _ = predictor.predict(message)
    if round(confidence, 4) > 0.5:
        for intent in intents_data['intents']:
            if tag == intent["tag"]:
                return JSONResponse(content={"response": random.choice(intent['responses'])})

    return JSONResponse(content={"response": "Tôi không hiểu bạn nói gì. Vui lòng nhập các thông tin rõ ràng hơn để tôi có thể hiểu, cảm ơn rất nhiều."})
# Api chatbot
@app.post("/chat-lstm")
async def chatlstm(request: ChatRequest):
    message = format_message(request.message)
    tag, confidence, _ = predictor_1.predict(message)
    response = "Xin lỗi, tôi không hiểu ý của bạn."
    if round(confidence, 4) > 0.5:
        # truy vấn tồn kho sản phẩm
        if tag == 'ton_kho_thap':
            # Tính tồn kho từ warehouses
            inventory = {}
            warehouses = warehouses_collection.find()
            
            for wh in warehouses:
                for item in wh.get('products', []):
                    product_id = str(item.get('product'))
                    quantity = item.get('quantity', 0)
                    
                    if product_id not in inventory:
                        inventory[product_id] = 0
                    
                    if wh.get('type') == 'import':
                        inventory[product_id] += quantity
                    elif wh.get('type') == 'export':
                        inventory[product_id] -= quantity
            
            # Lọc sản phẩm có tồn kho thấp
            low_stock_products = []
            for product_id, stock in inventory.items():
                if stock < 160:
                    product = products_collection.find_one({"_id": product_id})
                    if product:
                        low_stock_products.append({
                            'name': product.get('name', 'Không rõ tên'),
                            'stock': stock
                        })
            
            if low_stock_products:
                list_items = []
                for p in low_stock_products:
                    list_items.append(f"<li>{p['name']} ({p['stock']} sản phẩm)</li>")
                
                headers = [
                    "<p><strong>Các sản phẩm có tồn kho thấp hiện nay:</strong></p>",
                    "<p><strong>Danh sách những sản phẩm sắp hết hàng:</strong></p>",
                    "<p><strong>Những mặt hàng có số lượng thấp trong kho:</strong></p>",
                    "<p><strong>Kho đang thiếu các sản phẩm sau:</strong></p>",
                    "<p><strong>Lưu ý: Các sản phẩm dưới đây sắp hết kho!</strong></p>"
                ]
                footers = [
                    "<p><strong>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</strong></p>",
                    "<p><strong> Nếu có câu hỏi, bạn có thể hỏi?</strong></p>",
                    "<p><strong>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</strong></p>",
                    "<p><strong>Ngoài ra, bạn còn câu hỏi nào khác không?</strong></p>",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                
                response_html = header_html + "<ul>" + "".join(list_items) + "</ul>" + footer_html
                
                return JSONResponse(content={"response": response_html})
            else:
                no_data_responses = [
                    "<p>Hiện tại không có sản phẩm nào gần hết hàng. Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Tất cả sản phẩm đều còn tồn kho ổn định. Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Không có mặt hàng nào dưới mức tồn kho quy định. Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                    "<p>Kho hàng hiện đang đầy đủ, chưa có sản phẩm nào cần bổ sung. Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>"
                ]
                return JSONResponse(content={"response": random.choice(no_data_responses)})

        # Truy vấn sản phẩm tồn kho cao
        if tag == 'ton_kho_cao':
            # Tính tồn kho từ warehouses
            inventory = {}
            warehouses = warehouses_collection.find()
            
            for wh in warehouses:
                for item in wh.get('products', []):
                    product_id = str(item.get('product'))
                    quantity = item.get('quantity', 0)
                    
                    if product_id not in inventory:
                        inventory[product_id] = 0
                    
                    if wh.get('type') == 'import':
                        inventory[product_id] += quantity
                    elif wh.get('type') == 'export':
                        inventory[product_id] -= quantity
            
            # Lọc sản phẩm có tồn kho cao
            high_stock_products = []
            for product_id, stock in inventory.items():
                if stock > 160:
                    product = products_collection.find_one({"_id": product_id})
                    if product:
                        high_stock_products.append({
                            'name': product.get('name', 'Không rõ tên'),
                            'stock': stock
                        })
            
            if high_stock_products:
                list_items = []
                for p in high_stock_products:
                    list_items.append(f"<li>{p['name']} ({p['stock']} sản phẩm)</li>")
                
                headers = [
                    "<p><strong>Các sản phẩm có tồn kho cao hiện nay:</strong></p>",
                    "<p><strong>Những sản phẩm còn số lượng nhiều trong kho:</strong></p>",
                    "<p><strong>Những mặt hàng có số lượng lớn trong kho:</strong></p>",
                    "<p><strong>Các sản phẩm còn khá nhiều, bạn không cần nhập, cụ thể:</strong></p>",
                    "<p><strong>Bạn cần nên nhanh chóng đẩy các sản phẩm này tránh hết hạn sử dụng!</strong></p>"
                ]
                footers = [
                    "<p><strong>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</strong></p>",
                    "<p><strong> Nếu có câu hỏi, bạn có thể hỏi?</strong></p>",
                    "<p><strong>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</strong></p>",
                    "<p><strong>Ngoài ra, bạn còn câu hỏi nào khác không?</strong></p>",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                response_html = header_html + "<ul>" + "".join(list_items) + "</ul>" + footer_html
                return JSONResponse(content={"response": response_html})
            else:
                no_data_responses = [
                    "<p>Hiện tại không có sản phẩm nào có số lượng lớn trong kho. Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Tất cả sản phẩm đều còn tồn kho ổn định. Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Không có mặt hàng nào có số lượng lớn trong kho. Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                    "<p>Các sản phẩm trong kho đang trong trạng thái hết hàng. Vui lòng nhập kho để duy trì hoạt động kinh doanh. Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>"
                ]
                return JSONResponse(content={"response": random.choice(no_data_responses)})

        # Truy vấn tồn kho tất cả các sản phẩm
        if tag == 'truy_van_ton_kho_tat_ca':
            # Tính tồn kho từ warehouses
            inventory = {}
            warehouses = warehouses_collection.find()
            
            for wh in warehouses:
                for item in wh.get('products', []):
                    product_id = str(item.get('product'))
                    quantity = item.get('quantity', 0)
                    
                    if product_id not in inventory:
                        inventory[product_id] = 0
                    
                    if wh.get('type') == 'import':
                        inventory[product_id] += quantity
                    elif wh.get('type') == 'export':
                        inventory[product_id] -= quantity
            
            if inventory:
                list_items = []
                for product_id, stock in inventory.items():
                    product = products_collection.find_one({"_id": product_id})
                    product_name = product.get('name', 'Không rõ tên') if product else "Không rõ tên"
                    list_items.append(f"<li>{product_name}: ({stock} sản phẩm)</li>")
                
                headers = [
                    "<p><strong>Số lượng sản phẩm trong kho hiện nay:</strong></p>",
                    "<p><strong>Dưới đây là danh sách số lượng của các sản phẩm trong kho:</strong></p>",
                    "<p><strong>Tôi sẽ thống kê cho bạn về danh sách các sản phẩm trong kho</strong></p>",
                    "<p><strong>Về số lượng sản phẩm trong kho, bạn có thể tham khảo danh sách dưới đây.</strong></p>",
                    "<p><strong>Số lượng sản phẩm trong kho mà bạn cần đây:</strong></p>"
                ]
                footers = [
                    "<p><strong>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</strong></p>",
                    "<p><strong> Nếu có câu hỏi, bạn có thể hỏi?</strong></p>",
                    "<p><strong>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</strong></p>",
                    "<p><strong>Ngoài ra, bạn còn câu hỏi nào khác không?</strong></p>",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                response_html = header_html + "<ul>" + "".join(list_items) + "</ul>" + footer_html
                return JSONResponse(content={"response": response_html})
            else:
                no_data_responses = [
                    "<p>Hiện tại trong kho không chứa sản phẩm nào. Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Không có sản phẩm nào tồn tại trong kho. Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Kho hiện không chứa sản phẩm, bạn nên nhập sản phẩm vào kho để duy trì kinh doanh. Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                    "<p>Kho hiện nay đang trống, bạn có thể nhập thêm sản phẩm vào kho. Vui lòng nhập kho để duy trì hoạt động kinh doanh. Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>"
                ]
                return JSONResponse(content={"response": random.choice(no_data_responses)})

        # Truy vấn tồn kho của từng sản phẩm
        if tag == 'truy_van_ton_kho':
            name_patterns = [
                r"sản phẩm (.+) còn bao nhiêu",
                r"sản phẩm (.+) trong kho",
                r"sản phẩm (.+) trong kho là bao nhiêu",
                r"tồn kho sản phẩm (.+)",
                r"sản phẩm tên (.+)",
                r"sản phẩm (.+)"
            ]
            name_query = extract_value(name_patterns, message)
            
            if not name_query:
                return JSONResponse(content={"response": "Bạn vui lòng cung cấp thông tin chính xác của tên sản phẩm cần tìm."})

            # Tìm sản phẩm theo regex (chuẩn hóa keyword)
            keywords = name_query.strip().lower().split()
            pattern = "(?=.*" + ")(?=.*".join(map(re.escape, keywords)) + ")"

            matched_products = list(products_collection.find(
                {"name": {"$regex": pattern, "$options": "i"}}
            ))

            if matched_products:
                # Tính tồn kho từ warehouses
                inventory = {}
                warehouses = warehouses_collection.find()
                
                for wh in warehouses:
                    for item in wh.get('products', []):
                        product_id = str(item.get('product'))
                        quantity = item.get('quantity', 0)
                        
                        if product_id not in inventory:
                            inventory[product_id] = 0
                        
                        if wh.get('type') == 'import':
                            inventory[product_id] += quantity
                        elif wh.get('type') == 'export':
                            inventory[product_id] -= quantity

                results = []
                for product in matched_products:
                    product_id = str(product['_id'])
                    product_name = product.get('name', 'Không rõ tên')
                    stock = inventory.get(product_id, 0)
                    
                    status = "⚠️ Cần nhập thêm" if stock < 160 else "✅ Đủ hàng"
                    results.append({
                        'name': product_name,
                        'stock': stock,
                        'status': status
                    })

                # 1 sản phẩm => hiển thị chi tiết
                if len(results) == 1:
                    result = results[0]
                    headers = [
                        "<p>Tìm thấy sản phẩm</p>",
                        "<p>Thông tin sản phân</p>",
                        "<p>Tôi tìm ra được 1</p>",
                        "<p>Với từ khóa, tôi tìm được 1</p>",
                        "<p>Đây là thông tin sản phẩm</p>"
                    ]
                    footers = [
                        "<p>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                        "<p>Nếu có câu hỏi, bạn có thể hỏi?</p>",
                        "<p>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>",
                        "<p>Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                    ]
                    header_html = random.choice(headers)
                    footer_html = random.choice(footers)

                    if result['stock'] < 160 and result['stock'] > 0:
                        response = (
                            header_html +
                            f"<p><strong>{result['name']}</strong> hiện có {result['stock']} sản phẩm trong kho. "
                            f"<strong>Vui lòng nhập thêm sản phẩm vào kho hàng.</strong></p>" +
                            footer_html
                        )
                    elif result['stock'] >= 160:
                        response = (
                            header_html +
                            f"<p><strong>{result['name']}</strong> hiện có {result['stock']} sản phẩm trong kho.</p>" +
                            footer_html
                        )
                    else:
                        response = (
                            f"<p>Không tìm thấy thông tin tồn kho cho sản phẩm <strong>{result['name']}</strong>.</p>"
                        )
                    return JSONResponse(content={"response": response})
                
                # Nhiều sản phẩm => hiển thị danh sách
                else:
                    headers = [
                        f"<p>Tìm thấy <strong>{len(results)}</strong> sản phẩm khớp với '<em>{name_query}</em>':</p>",
                        f"<p>Thông tin các sản phẩm liên quan đến '<em>{name_query}</em>':</p>",
                        f"<p>Tôi tìm thấy {len(results)} sản phẩm liên quan đến từ khóa '<em>{name_query}</em>'.</p>",
                        f"<p>Với từ khóa bạn cung cấp, tôi tìm được {len(results)} sản phẩm sau:</p>",
                        "<p>Đây là thông tin tồn kho các sản phẩm khớp:</p>"
                    ]
                    footers = [
                        "<p>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                        "<p>Nếu có câu hỏi, bạn có thể hỏi?</p>",
                        "<p>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>",
                        "<p>Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                    ]
                    lines = [random.choice(headers), "<ul>"]
                    for result in results:
                        if result['stock'] > 0:
                            lines.append(f"<li><strong>{result['name']}</strong>: {result['stock']} sản phẩm - {result['status']}</li>")
                        else:
                            lines.append(f"<li><strong>{result['name']}</strong>: {result['status']}</li>")
                    lines.append("</ul>")
                    lines.append(random.choice(footers))
                    return JSONResponse(content={"response": "".join(lines)})

            else:
                response = (
                    f"<p>Không tìm thấy sản phẩm nào có tên chứa '<em>{name_query}</em>'. "
                    "Hãy nhập tên sản phẩm cụ thể hơn.</p>"
                )
                return JSONResponse(content={"response": response})
        # Truy vấn lượt mua sản phẩm
        if tag == 'truy_van_luot_mua':
            name_patterns = [
                r"lượt mua sản phẩm (.+)",
                r"lượt mua sản phẩm (.+) là bao nhiêu",
                r"lượt mua của sản phẩm (.+)", 
                r"lượt mua của sản phẩm (.+) là bao nhiêu", 
                r"sản phẩm tên (.+)",
                r"sản phẩm (.+)",
                r"sản phẩm (.+) có lượt mua",
                r"(.+) là bao nhiêu",
            ]
            name_query = extract_value(name_patterns, message)
            if not name_query:
                return JSONResponse(content={"response": "Bạn vui lòng cung cấp tên sản phẩm cần tìm."})
            keywords = name_query.strip().lower().split()
            pattern = "(?=.*" + ")(?=.*".join(map(re.escape, keywords)) + ")"

            products = list(products_collection.find(
                {"name": {"$regex": pattern, "$options": "i"}}
            ))
            if not products:
                return JSONResponse(content={
                    "response": f"Không tìm thấy sản phẩm nào khớp với '{name_query}'. Vui lòng kiểm tra lại tên sản phẩm."
                })
            results = []
            
            for product in products:
                product_name = product['name']
                product_id = product['_id']
                
                # Kiểm tra trường 'sold' trong sản phẩm
                if 'sold' in product and product['sold'] is not None:
                    sold_count = product['sold']
                else:
                    sold_count = 0
                
                results.append({
                    'name': product_name,
                    'sold': sold_count,
                    'id': product_id
                })
            # Hiển thị kết quả
            if len(results) == 1:
                headers = [
                    "<p>Tìm thấy sản phẩm</p>",
                    "<p>Thông tin sản phân</p>",
                    "<p>Tôi tìm ra được 1</p>",
                    "<p>Với từ khóa, tôi tìm được 1</p>",
                    "<p>Đây là thông tin sản phẩm</p>"
                ]
                footers = [
                    "<p>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>",
                    "<p>Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                result = results[0]
                if result['sold'] >= 100:
                    response = (
                            header_html +
                            f"<p><strong>{(result['name'])}</strong> hiện có lượt mua là {result['sold']}. " +
                            footer_html
                        )
                else:
                    response = (
                            header_html +
                            f"<p><strong>{(result['name'])}</strong> hiện có lượt mua là {result['sold']}." +
                            footer_html
                        )
                return JSONResponse(content={"response": response})
            else:
                # Nếu có nhiều sản phẩm, hiển thị danh sách
                headers = [
                        f"<p>Tìm thấy <strong>{len(results)}</strong> sản phẩm khớp với '<em>{(name_query)}</em>':</p>",
                        f"<p>Thông tin các sản phẩm liên quan đến '<em>{(name_query)}</em>':</p>",
                        f"<p>Tôi tìm thấy {len(results)} sản phẩm liên quan đến từ khóa '<em>{(name_query)}</em>'.</p>",
                        f"<p>Với từ khóa bạn cung cấp, tôi tìm được {len(results)} sản phẩm sau:</p>",
                        "<p>Đây là thông tin tồn kho các sản phẩm khớp:</p>"
                    ]
                footers = [
                    "<p>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>",
                    "<p>Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                ]
                lines = [random.choice(headers), "<ul>"]
                for result in results:
                    if result['sold'] > 0:
                        lines.append(f"<li><strong>{(result['name'])}</strong>: {result['sold']} lượt mua")
                    else:
                        lines.append(f"<li><strong>{(result['name'])}</strong>: Chưa có lượt mua")
                lines.append("</ul>")
                lines.append(random.choice(footers))
                return JSONResponse(content={"response": "".join(lines)})
        # Hiển thị lượt mua của tất cả sản phẩm 
        if tag == 'truy_van_luot_mua_tat_ca':
            products = list(products_collection.find({}))
            if products:
                response_list = []
                for product in products:
                    response_list.append(f"{product['name']} có lượt mua ({product['sold']} lượt mua sản phẩm)")
                headers = [
                    "<p>Các sản phẩm có lượt mua lần lượt là:</p>",
                    "<p>Dưới đây là thông tin lượt mua của các sản phẩm:</p>",
                    "<p>Doanh số các sản phẩm bán ra, bạn có thể tham khảo:</p>",
                ]
                response = "Các sản phẩm có lượt mua lần lượt: " + ", ".join(response_list)
                return JSONResponse(content={"response": response})
            else:
                response = "Hiện tại không có sản phẩm nào được mua."
                return JSONResponse(content={"response": response})
        # Lượt mua thấp
        if tag == 'luot_mua_thap':
            products = list(products_collection.find({"sold": {"$lt": 100}}))
            if products:
                response_list = []
                for product in products:
                    response_list.append(f"<li>Sản phẩm {product['name']} có lượt mua là {product['sold']}</li>")
                headers = [
                    "<p><strong>Các sản phẩm có lượt mua thấp hiện nay:</strong></p>",
                    "<p><strong>Danh sách những sản phẩm có lượt mua thấp:</strong></p>",
                    "<p><strong>Dưới đây là những sản phẩm có lượt mua thấp:</strong></p>",
                    "<p><strong>Tôi đã liệt kê các sản phẩm có lượt mua thấp, dưới đây:</strong></p>",
                    "<p><strong>Về lượt mua thấp, dưới đây là các sản phẩm:</strong></p>"
                ]
                footers = [
                    "<p><strong>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</strong></p>",
                    "<p><strong>Nếu có câu hỏi, bạn có thể hỏi?</strong></p>",
                    "<p><strong>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</strong></p>",
                    "<p><strong>Ngoài ra, bạn còn câu hỏi nào khác không?</strong></p>",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                response_html = header_html + "<ul>" + "".join(response_list) + "</ul>" + footer_html
                return JSONResponse(content={"response": response_html})
            else:
                no_data_responses = [
                    "<p>Hiện tại không có sản phẩm nào có lượt mua thấp. Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Tất cả sản phẩm đều có lượt bán ổn định theo thời gian. Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Không có sản phẩm nào có lượt mua thấp. Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                    "<p>Lượt mua hiện tại của các sản phẩm khá cao, chưa phát hiện sản phẩm nào có lượt mua thấp. Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>"
                ]
                return JSONResponse(content={"response": random.choice(no_data_responses)})
        # Lượt mua cao
        if tag == 'luot_mua_cao':
            products = list(products_collection.find({"sold": {"$gt": 100}}))
            if products:
                response_list = []
                for product in products:
                    response_list.append(f"<li>Sản phẩm {product['name']} có lượt mua là {product['sold']}</li>")
                headers = [
                    "<p><strong>Các sản phẩm có lượt mua cao hiện nay:</strong></p>",
                    "<p><strong>Danh sách những sản phẩm có lượt mua cao:</strong></p>",
                    "<p><strong>Dưới đây là những sản phẩm có lượt mua cao:</strong></p>",
                    "<p><strong>Tôi đã liệt kê các sản phẩm có lượt mua cao, dưới đây:</strong></p>",
                    "<p><strong>Dưới đây là các sản phẩm có lượt mua cao:</strong></p>"
                ]
                footers = [
                    "<p><strong>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</strong></p>",
                    "<p><strong>Nếu có câu hỏi, bạn có thể hỏi?</strong></p>",
                    "<p><strong>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</strong></p>",
                    "<p><strong>Ngoài ra, bạn còn câu hỏi nào khác không?</strong></p>",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                response_html = header_html + "<ul>" + "".join(response_list) + "</ul>" + footer_html
                return JSONResponse(content={"response": response})
            else:
                no_data_responses = [
                    "<p>Hiện tại không có sản phẩm nào có lượt mua cao. Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Các sản phẩm trong cửa hàng có lượt mua khá thấp. Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Không tìm thấy sản phẩm nào có lượt mua cao. Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                    "<p>Lượt mua hiện tại của các sản phẩm khá thấp, chưa phát hiện sản phẩm nào có lượt mua cao. Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>"
                ]
                return JSONResponse(content={"response": random.choice(no_data_responses)})
        # Truy vấn đánh giá
        if tag == 'truy_van_danh_gia':
            name_patterns = [
                r"lượt đánh giá của sản phẩm (.+)",
                r"lượt đánh giá sản phẩm (.+) là bao nhiêu",
                r"lượt đánh giá của sản phẩm (.+) là bao nhiêu",
                r"lượt đánh giá của sản phẩm (.+)", 
                r"sản phẩm tên (.+) có lượt đánh giá là bao nhiêu",
                r"sản phẩm (.+)",
                r"(.+) là bao nhiêu",
            ]
            name_query = extract_value(name_patterns, message)
            if not name_query:
                return JSONResponse(content={"response": "Bạn vui lòng cung cấp tên sản phẩm cần tìm."})
            keywords = name_query.strip().lower().split()
            pattern = "(?=.*" + ")(?=.*".join(map(re.escape, keywords)) + ")"

            products = list(products_collection.find(
                {"name": {"$regex": pattern, "$options": "i"}}
            ))
            if not products:
                return JSONResponse(content={
                    "response": f"Không tìm thấy sản phẩm nào khớp với '{name_query}'. Vui lòng kiểm tra lại tên sản phẩm."
                })
            results = []
            
            for product in products:
                product_name = product['name']
                product_id = product['_id']
                
                # Kiểm tra trường 'totalRatings' trong sản phẩm
                if 'totalRatings' in product and product['totalRatings'] is not None:
                    totalRatings_count = product['totalRatings']
                else:
                    totalRatings_count = 0
                
                results.append({
                    'name': product_name,
                    'totalRatings': totalRatings_count,
                    'id': product_id
                })
            # Hiển thị kết quả
            if len(results) == 1:
                headers = [
                    "<p>Tìm thấy sản phẩm</p>",
                    "<p>Thông tin sản phân</p>",
                    "<p>Tôi tìm ra được 1</p>",
                    "<p>Với từ khóa, tôi tìm được 1</p>",
                    "<p>Đây là thông tin sản phẩm</p>"
                ]
                footers = [
                    "<p>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>",
                    "<p>Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                result = results[0]
                if result['totalRatings'] >= 100:
                    response = (
                            header_html +
                            f"<p><strong>{(result['name'])}</strong> có lượt đánh giá là {result['totalRatings']}. " +
                            footer_html
                        )
                else:
                    response = (
                            header_html +
                            f"<p><strong>{(result['name'])}</strong> có lượt đánh giá là {result['totalRatings']}." +
                            footer_html
                        )
                return JSONResponse(content={"response": response})
            else:
                # Nếu có nhiều sản phẩm, hiển thị danh sách
                headers = [
                        f"<p>Tìm thấy <strong>{len(results)}</strong> sản phẩm khớp với '<em>{(name_query)}</em>':</p>",
                        f"<p>Thông tin các sản phẩm liên quan đến '<em>{(name_query)}</em>':</p>",
                        f"<p>Tôi tìm thấy {len(results)} sản phẩm liên quan đến từ khóa '<em>{(name_query)}</em>'.</p>",
                        f"<p>Với từ khóa bạn cung cấp, tôi tìm được {len(results)} sản phẩm sau:</p>",
                        "<p>Đây là thông tin tồn kho các sản phẩm khớp:</p>"
                    ]
                footers = [
                    "<p>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>",
                    "<p>Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                ]
                lines = [random.choice(headers), "<ul>"]
                for result in results:
                    if result['totalRatings'] > 0:
                        lines.append(f"<li><strong>{(result['name'])}</strong>: {result['totalRatings']} đánh giá")
                    else:
                        lines.append(f"<li><strong>{(result['name'])}</strong>: Chưa có đánh giá")
                lines.append("</ul>")
                lines.append(random.choice(footers))
                return JSONResponse(content={"response": "".join(lines)})
        # Lượt đánh giá của tất cả sản phẩm 
        if tag == 'truy_van_danh_gia_tat_ca':
            products = list(products_collection.find({}))
            if products:
                response_list = []
                for product in products:
                    response_list.append(f"<li>{product['name']} có đánh giá là ({product['totalRatings']} sao)</li>")
                headers = [
                    "<p><strong>Lượt đánh giá của tất cả các sản phẩm là:</strong></p>",
                    "<p><strong>Dưới đây là danh sách lượt đánh giá của tất cả các sản phẩm:</strong></p>",
                    "<p><strong>Các lượt đánh giá của sản phẩm:</strong></p>",
                    "<p><strong>Bạn có thể tham khảo về lượt đánh giá của các sản phẩm:</strong></p>",
                    "<p><strong>Tôi cung cấp cho bạn thông tin về lượt đánh giá của tất cả sản phẩm:</strong></p>"
                ]
                footers = [
                    "<p><strong>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</strong></p>",
                    "<p><strong>Nếu có câu hỏi, bạn có thể hỏi?</strong></p>",
                    "<p><strong>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</strong></p>",
                    "<p><strong>Ngoài ra, bạn còn câu hỏi nào khác không?</strong></p>",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                response_html = header_html + "<ul>" + "".join(response_list) + "</ul>" + footer_html
                return JSONResponse(content={"response": response_html})
            else:
                no_data_responses = [
                    "<p>Hiện tại không có sản phẩm nào được đánh giá. Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Các sản phẩm trong cửa hàng chưa được khách hàng đánh giá. Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Không tìm thấy đánh giá về sản phẩm. Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                    "<p>Chưa có sản phẩm nào được đánh giá. Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>"
                ]
                return JSONResponse(content={"response": random.choice(no_data_responses)})
        # Lượt đánh giá thấp
        if tag == 'danh_gia_thap':
            products = list(products_collection.find({"totalRatings": {"$lt": 3}}))
            if products:
                response_list = []
                for product in products:
                    response_list.append(f"<li>{product['name']} có đánh giá là ({product['totalRatings']} sao)</li>")
                headers = [
                    "<p><strong>Các sản phẩm có lượt đánh giá thấp lần lượt là:</strong></p>",
                    "<p><strong>Dưới đây là danh sách sản phẩm có lượt đánh giá thấp:</strong></p>",
                    "<p><strong>Thông tin các sản phẩm có lượt đánh giá thấp:</strong></p>",
                    "<p><strong>Các sản phẩm có lượt đánh giá thấp, bạn có thể xem chi tiết dưới đây:</strong></p>",
                    "<p><strong>Tôi cung cấp cho bạn thông tin về lượt đánh giá thấp của các sản phẩm:</strong></p>"
                ]
                footers = [
                    "<p><strong>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</strong></p>",
                    "<p><strong>Nếu có câu hỏi, bạn có thể hỏi?</strong></p>",
                    "<p><strong>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</strong></p>",
                    "<p><strong>Ngoài ra, bạn còn câu hỏi nào khác không?</strong></p>",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                response_html = header_html + "<ul>" + "".join(response_list) + "</ul>" + footer_html
                return JSONResponse(content={"response": response_html})
            else:
                no_data_responses = [
                    "<p>Hiện tại không có sản phẩm nào được đánh giá thấp. Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Các sản phẩm đánh giá thấp không tồn tại. Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Không tìm thấy sản phẩm nào được đánh giá thấp. Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                    "<p>Chưa có sản phẩm nào được đánh giá thấp. Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>"
                ]
                return JSONResponse(content={"response": random.choice(no_data_responses)})
        # Lượt đánh giá cao
        if tag == 'danh_gia_cao':
            products = list(products_collection.find({"totalRatings": {"$gt": 3}}))
            if products:
                response_list = []
                for product in products:
                    response_list.append(f"<li>{product['name']} có đánh giá là ({product['totalRatings']} sao)</li>")
                headers = [
                    "<p><strong>Các sản phẩm có lượt đánh giá cao lần lượt là:</strong></p>",
                    "<p><strong>Dưới đây là danh sách sản phẩm có lượt đánh giá cao:</strong></p>",
                    "<p><strong>Thông tin các sản phẩm có lượt đánh giá cao:</strong></p>",
                    "<p><strong>Các sản phẩm có lượt đánh giá cao, bạn có thể xem chi tiết dưới đây:</strong></p>",
                    "<p><strong>Tôi cung cấp cho bạn thông tin về lượt đánh giá cao của các sản phẩm:</strong></p>"
                ]
                footers = [
                    "<p><strong>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</strong></p>",
                    "<p><strong>Nếu có câu hỏi, bạn có thể hỏi?</strong></p>",
                    "<p><strong>Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</strong></p>",
                    "<p><strong>Ngoài ra, bạn còn câu hỏi nào khác không?</strong></p>",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                response_html = header_html + "<ul>" + "".join(response_list) + "</ul>" + footer_html
                return JSONResponse(content={"response": response_html})
            else:
                no_data_responses = [
                    "<p>Hiện tại không có sản phẩm nào được đánh giá. Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Các sản phẩm trong cửa hàng chưa được khách hàng đánh giá. Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Không tìm thấy đánh giá về sản phẩm. Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                    "<p>Chưa có sản phẩm nào được đánh giá. Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>"
                ]
                return JSONResponse(content={"response": random.choice(no_data_responses)})
        # Hiển thị số lượng khác khách hàng
        if tag == 'so_luong_nguoi_dung':
            users = list(user_collection.find({"role": "2004"})) 
            users_count = len(users)
            if users_count:
                headers = [
                    "Về số lượng người dùng trong hệ thống, hiện nay có ",
                    "Tôi cung cấp cho bạn về số lượng người dùng trong hệ thống: ",
                    "Hệ thống hiện nay có khoảng ",
                ]
                footers = [
                    " Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.",
                    " Nếu có câu hỏi, bạn có thể hỏi?",
                    " Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.",
                    " Ngoài ra, bạn còn câu hỏi nào khác không?",
                ]
                header_text = random.choice(headers)
                footer_text = random.choice(footers)
                response_html = f"<p>{header_text}{users_count} tài khoản đăng ký mới.{footer_text}</p>"
                return JSONResponse(content={"response": response_html})
            else:
                no_data_responses = [
                    "<p>Hiện tại không có người dùng nào trong hệ thống. Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Tôi chưa tìm thấy người dùng trong hệ thống. Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Không tìm thấy tài khoản trong hệ thống. Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                    "<p>Hệ thống chưa có khách hàng đăng ký. Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>"
                ]
                return JSONResponse(content={"response": random.choice(no_data_responses)})
        # Số lượng nhân viên quản lý
        if tag == 'so_luong_nhan_vien_quan_ly':
            users = list(user_collection.find({"role": {"$in": ["2002",  "2006"]}})) 
            users_count = len(users)
            if users_count:
                headers = [
                    "Về số lượng nhân viên trong hệ thống, hiện nay có",
                    "Tôi cung cấp cho bạn về số lượng nhân viên trong hệ thống",
                    "Hệ thống hiện nay có khoảng",
                ]
                footers = [
                    "Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.",
                    "Nếu có câu hỏi, bạn có thể hỏi?",
                    "Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.",
                    "Ngoài ra, bạn còn câu hỏi nào khác không?",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                response_html = f"<p>{header_html}{users_count} tài khoản đăng ký.{footer_html}</p>"
                return JSONResponse(content={"response": response_html})
            else:
                no_data_responses = [
                    "<p>Hiện tại không có tài khoản nào trong hệ thống. Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</p>",
                    "<p>Tôi chưa tìm thấy thông tin nhân viên trong hệ thống. Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Không tìm thấy tài khoản trong hệ thống. Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                    "<p>Hệ thống chưa có nhân viên đăng ký. Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.</p>"
                ]
                return JSONResponse(content={"response": random.choice(no_data_responses)})
        # khách hàng chưa mua hàng
        if tag == 'khach_chua_mua_hang':
            users = list(user_collection.find({"role": "2004"}, {"_id": 1, "name": 1}))
            user_ids = [user["_id"] for user in users]
            orders = list(order_collection.find({"orderBy": {"$in": user_ids}}, {"orderBy": 1}))
            user_ids_with_orders = set(order["orderBy"] for order in orders)
            users_without_orders = [user for user in users if user["_id"] not in user_ids_with_orders]
            customer_names = [user.get("name", "Không rõ tên") for user in users_without_orders]
            users_count = len(customer_names)
            if users_count:
                headers = [
                    "<p><strong>Danh sách khách hàng chưa từng đặt hàng trong hệ thống:</strong></p>",
                    "<p><strong>Thông tin khách hàng chưa từng đăng hàng hiện có:</strong></p>",
                    "<p><strong>Dưới đây là các khách hàng chưa từng mua hàng:</strong></p>",
                ]
                footers = [
                    "<p><strong>Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.</strong></p>",
                    "<p><strong>Nếu có câu hỏi, bạn có thể hỏi?</strong></p>",
                    "<p><strong>Ngoài ra, bạn còn câu hỏi nào khác không?</strong></p>",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)

                name_list_html = "<ul>" + "".join(f"<li>{name}</li>" for name in customer_names) + "</ul>"
                body_html = f"<p><strong>Có tổng cộng {users_count} khách hàng tiềm năng:</strong></p>{name_list_html}"
                response_html = header_html + body_html + footer_html
            else:
                no_data_responses = [
                    "<p>Hiện không có khách hàng nào chưa từng đặt đơn hàng. Bạn còn câu hỏi nào không?</p>",
                    "<p>Tôi chưa tìm thấy tài khoản nào chưa từng đặt hàng trong hệ thống. Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Không tìm thấy khách hàng chưa đặt hàng. Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                ]
                response_html = random.choice(no_data_responses)

            return JSONResponse(content={"response": response_html})
        # khách hàng mua hàng nhiều
        if tag == 'khach_hang_mua_nhieu':
            users = list(user_collection.find({"role": "2004"}, {"_id": 1, "name": 1}))
            user_ids = [user["_id"] for user in users]
            orders = list(order_collection.find({"orderBy": {"$in": user_ids}}, {"orderBy": 1}))

            order_counts = Counter(order["orderBy"] for order in orders)
            customer_ids_many_orders = [user_id for user_id, count in order_counts.items() if count > 2]
            customer_info = [
                {
                    "name": user["name"],
                    "count": order_counts[user["_id"]]
                }
                for user in users if user["_id"] in customer_ids_many_orders
            ]
            users_count = len(customer_info)
            if users_count:
                headers = [
                    "Dưới đây là danh sách khách hàng đã mua hàng nhiều lần:",
                    "Tôi đã tìm thấy các khách hàng trung thành:",
                    "Thông tin khách hàng mua nhiều như sau:",
                ]
                footers = [
                    "Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.",
                    "Nếu có câu hỏi, bạn có thể hỏi?",
                    "Ngoài ra, bạn còn câu hỏi nào khác không?",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)

                name_list_html = "<ul>" + "".join(
                    f"<li>{info['name']}: <strong>{info['count']}</strong> đơn hàng</li>" for info in customer_info
                ) + "</ul>"

                body_html = f"<p><strong>Có tổng cộng {users_count} khách hàng đã mua hàng nhiều tại hệ thông:</strong></p>{name_list_html}"
                response_html = f"<p><strong>{header_html}</strong></p>{body_html}<p><strong>{footer_html}</strong></p>"
            else:
                no_data_responses = [
                    "<p>Không có khách hàng nào đã đặt hàng nhiều. Bạn còn câu hỏi nào không?</p>",
                    "<p>Tôi chưa tìm thấy khách hàng mua nhiều lần. Nếu có câu hỏi, bạn có thể hỏi?</p>",
                    "<p>Hiện tại chưa có khách hàng nào đã mua hàng nhiều. Ngoài ra, bạn còn câu hỏi nào khác không?</p>",
                ]
                response_html = random.choice(no_data_responses)
            return JSONResponse(content={"response": response_html})
        # tìm kiếm khách hàng theo tên
        if tag == 'tim_kiem_khach_theo_ten':
            name_patterns = [
                r"tên là ([\w\s]+)",
                r"khách hàng tên ([\w\s]+)",
                r"người dùng tên ([\w\s]+)",
                r"ai là người tên ([\w\s]+)",
                r"tài khoản tên ([\w\s]+)",
                r"tên ([\w\s]+)"
            ]
            name_query = extract_value(name_patterns, message)
            if not name_query:
                return JSONResponse(content={"response": "Bạn vui lòng cung cấp tên khách hàng cần tìm."})
            # Xử lý tìm kiếm với regex như đã trình bày
            keywords = name_query.strip().lower().split()
            pattern = "(?=.*" + ")(?=.*".join(map(re.escape, keywords)) + ")"

            matched_users = list(user_collection.find(
                {"name": {"$regex": pattern, "$options": "i"}, "role": "2000"},
                {"_id": 1, "name": 1, "email": 1, "phone": 1, "address": 1}
            ))

            headers = [
                "<p><strong>Dưới đây là thông tin khách hàng theo tên bạn đã cung cấp:</strong></p>",
                "<p><strong>Kết quả tìm kiếm khách hàng như sau:</strong></p>",
                "<p><strong>Thông tin chi tiết khách hàng được tìm thấy:</strong></p>",
            ]
            footers = [
                "<p><strong>Bạn cần hỗ trợ thêm gì không? Tôi luôn sẵn sàng giúp.</strong></p>",
                "<p><strong>Nếu cần tìm thông tin khác, bạn cứ hỏi nhé!</strong></p>",
                "<p><strong>Bạn còn câu hỏi nào khác không?</strong></p>",
            ]

            if matched_users:
                user_list_html = "<ul>" + "".join(
                    f"<li>Tên: {user.get('name', '-')}</li>"
                    f"<li>Email: {user.get('email', '-')}</li>"
                    f"<li>SĐT: {user.get('phone', '-')}</li>"
                    f"<li>Địa chỉ: {user.get("address", {}).get("detail", "-")}</li>"
                    for user in matched_users
                ) + "</ul>"
                header_html = random.choice(headers)
                footer_html = random.choice(footers)

                full_response = (
                    f"{header_html}"
                    f"<p><strong>Tìm thấy {len(matched_users)} khách hàng với tên {name_query}:</strong></p>"
                    f"{user_list_html}"
                    f"{footer_html}"
                )
            else:
                full_response = (
                    f"<p><strong>Không tìm thấy khách hàng nào với tên '{name_query}'.</strong></p>"
                    f"{random.choice(footers)}"
                )
            return JSONResponse(content={"response": full_response})
        # tìm kiếm khách hàng theo email
        if tag == 'tim_kiem_khach_theo_email':
            email_patterns = [
                r"email là ([\w\.\-]+@[\w\.\-]+)",
                r"email ([\w\.\-]+@[\w\.\-]+)"
            ]
            email = extract_value(email_patterns, message)
            if not email:
                return JSONResponse(content={"response": "Bạn vui lòng cung cấp email khách hàng cần tìm."})

            matched_users = list(user_collection.find(
                {"email": email,  "role": "2000"},
                {"_id": 1, "name": 1, "email": 1, "phone": 1, "address": 1}
            ))
            headers = [
                "<p><strong>Dưới đây là thông tin khách hàng theo email bạn đã cung cấp:</strong></p>",
                "<p><strong>Kết quả tìm kiếm khách hàng như sau:</strong></p>",
                "<p><strong>Thông tin chi tiết khách hàng được tìm thấy:</strong></p>",
            ]
            footers = [
                "<p><strong>Bạn cần hỗ trợ thêm gì không? Tôi luôn sẵn sàng giúp.</strong></p>",
                "<p><strong>Nếu cần tìm thông tin khác, bạn cứ hỏi nhé!</strong></p>",
                "<p><strong>Bạn còn câu hỏi nào khác không?</strong></p>",
            ]

            if matched_users:
                user_list_html = "<ul>" + "".join(
                    f"<li>Têm: {user.get('name', '-')}</li>"
                    f"<li>Email: {user.get('email', '-')}</li>"
                    f"<li>SĐT: {user.get('phone', '-')}</li>"
                    f"<li>Địa chỉ: {user.get("address", {}).get("detail", "-")}</li>"
                    for user in matched_users
                ) + "</ul>"
                header_html = random.choice(headers)
                footer_html = random.choice(footers)

                full_response = (
                    f"{header_html}"
                    f"<p><strong>Tìm thấy {len(matched_users)} khách hàng với email {email}:</strong></p>"
                    f"{user_list_html}"
                    f"{footer_html}"
                )
            else:
                full_response = (
                    f"<p><strong>Không tìm thấy khách hàng nào với email '{email}'.</strong></p>"
                    f"{random.choice(footers)}"
                )
            return JSONResponse(content={"response": full_response})
        # tìm kiếm khách hàng theo số điện thoại
        if tag == 'tim_kiem_khach_theo_sdt':
            sdt_patterns = [
                r"số điện thoại là (\d+)",
                r"số điện thoại (\d+)",
                r"sđt (\d+)"
            ]
            phone = extract_value(sdt_patterns, message)
            if not phone:
                return JSONResponse(content={"response": "Bạn vui lòng cung cấp số điện thoại khách hàng cần tìm."})

            matched_users = list(user_collection.find(
                {"phone": phone,  "role": "2000"},
                {"_id": 1, "name": 1, "email": 1, "phone": 1, "address": 1}
            ))
            headers = [
                "<p><strong>Dưới đây là thông tin khách hàng theo số điện thoại bạn đã cung cấp:</strong></p>",
                "<p><strong>Kết quả tìm kiếm khách hàng như sau:</strong></p>",
                "<p><strong>Thông tin chi tiết khách hàng được tìm thấy:</strong></p>",
            ]
            footers = [
                "<p><strong>Bạn cần hỗ trợ thêm gì không? Tôi luôn sẵn sàng giúp.</strong></p>",
                "<p><strong>Nếu cần tìm thông tin khác, bạn cứ hỏi nhé!</strong></p>",
                "<p><strong>Bạn còn câu hỏi nào khác không?</strong></p>",
            ]

            if matched_users:
                user_list_html = "<ul>" + "".join(
                    f"<li>Têm: {user.get('name', '-')}</li>"
                    f"<li>Email: {user.get('email', '-')}</li>"
                    f"<li>SĐT: {user.get('phone', '-')}</li>"
                    f"<li>Địa chỉ: {user.get("address", {}).get("detail", "-")}</li>"
                    for user in matched_users
                ) + "</ul>"
                header_html = random.choice(headers)
                footer_html = random.choice(footers)

                full_response = (
                    f"{header_html}"
                    f"<p><strong>Tìm thấy {len(matched_users)} khách hàng với số điện thoại {phone}:</strong></p>"
                    f"{user_list_html}"
                    f"{footer_html}"
                )
            else:
                full_response = (
                    f"<p><strong>Không tìm thấy khách hàng nào với số điện thoại '{phone}'.</strong></p>"
                    f"{random.choice(footers)}"
                )
            return JSONResponse(content={"response": full_response})
        # Tìm kiếm khách hàng theo địa chỉ  
        if tag == 'tim_kiem_khach_theo_dia_chi':
            address_patterns = [
                r"ở (.+)",
                r"địa chỉ (.+)",
                r"sống tại (.+)",
                r"tại (.+)",
                r"cư trú ở (.+)",
            ]
            address_query = extract_value(address_patterns, message)
            if not address_query:
                return JSONResponse(content={"response": "Vui lòng cung cấp địa chỉ cần tìm."})
            matched_users = list(user_collection.find({
                "$or": [
                    {"address.province.name": {"$regex": re.escape(address_query), "$options": "i"},  "role": "2000"},
                    {"address.district.name": {"$regex": re.escape(address_query), "$options": "i"},  "role": "2000"},
                    {"address.ward.name": {"$regex": re.escape(address_query), "$options": "i"},  "role": "2000"},
                    {"address.detail": {"$regex": re.escape(address_query), "$options": "i"},  "role": "2000"},
                    {"address.addressAdd": {"$regex": re.escape(address_query), "$options": "i"},  "role": "2000"},
                ]
            }, {"_id": 1, "name": 1, "email": 1, "phone": 1, "address": 1}))
            headers = [
                "<p><strong>Dưới đây là thông tin khách hàng theo địa chỉ bạn đã cung cấp:</strong></p>",
                "<p><strong>Kết quả tìm kiếm khách hàng như sau:</strong></p>",
                "<p><strong>Thông tin chi tiết khách hàng được tìm thấy:</strong></p>",
            ]
            footers = [
                "<p><strong>Bạn cần hỗ trợ thêm gì không? Tôi luôn sẵn sàng giúp.</strong></p>",
                "<p><strong>Nếu cần tìm thông tin khác, bạn cứ hỏi nhé!</strong></p>",
                "<p><strong>Bạn còn câu hỏi nào khác không?</strong></p>",
            ]

            if matched_users:
                user_list_html = "<ul>" + "".join(
                    f"<li>Têm: {user.get('name', '-')}</li>"
                    f"<li>Email: {user.get('email', '-')}</li>"
                    f"<li>SĐT: {user.get('phone', '-')}</li>"
                    f"<li>Địa chỉ: {user.get("address", {}).get("detail", "-")}</li>"
                    for user in matched_users
                ) + "</ul>"
                header_html = random.choice(headers)
                footer_html = random.choice(footers)

                full_response = (
                    f"{header_html}"
                    f"<p><strong>Tìm thấy {len(matched_users)} khách hàng với địa chỉ {address_query}:</strong></p>"
                    f"{user_list_html}"
                    f"{footer_html}"
                )
            else:
                full_response = (
                    f"<p><strong>Không tìm thấy khách hàng nào với số điện thoại '{address_query}'.</strong></p>"
                    f"{random.choice(footers)}"
                )
            return JSONResponse(content={"response": full_response})
        # tìm kiếm nhân viên theo tên
        if tag == 'tim_kiem_nhan_vien_theo_ten':
            name_patterns = [
                r"tên nhân viên là ([\w\s]+)",
                r"nhân viên tên ([\w\s]+)",
                r"ai là nhân viên tên ([\w\s]+)",
                r"tên ([\w\s]+)"
            ]
            name_query = extract_value(name_patterns, message)
            if not name_query:
                return JSONResponse(content={"response": "Bạn vui lòng cung cấp tên nhân viên cần tìm."})
            # Xử lý tìm kiếm với regex như đã trình bày
            keywords = name_query.strip().lower().split()
            pattern = "(?=.*" + ")(?=.*".join(map(re.escape, keywords)) + ")"

            matched_users = list(user_collection.find(
                {"name": {"$regex": pattern, "$options": "i"}, "role": {"$in": ['2002', '2006']}},
                {"_id": 1, "name": 1, "email": 1, "phone": 1, "address": 1}
            ))

            headers = [
                "<p><strong>Dưới đây là thông tin nhân viên theo tên bạn đã cung cấp:</strong></p>",
                "<p><strong>Kết quả tìm kiếm nhân viên như sau:</strong></p>",
                "<p><strong>Thông tin chi tiết nhân viên được tìm thấy:</strong></p>",
            ]
            footers = [
                "<p><strong>Bạn cần hỗ trợ thêm gì không? Tôi luôn sẵn sàng giúp.</strong></p>",
                "<p><strong>Nếu cần tìm thông tin khác, bạn cứ hỏi nhé!</strong></p>",
                "<p><strong>Bạn còn câu hỏi nào khác không?</strong></p>",
            ]

            if matched_users:
                user_list_html = "<ul>" + "".join(
                    f"<li>Têm: {user.get('name', '-')}</li>"
                    f"<li>Email: {user.get('email', '-')}</li>"
                    f"<li>SĐT: {user.get('phone', '-')}</li>"
                    f"<li>Địa chỉ: {user.get("address", {}).get("detail", "-")}</li>"
                    for user in matched_users
                ) + "</ul>"
                header_html = random.choice(headers)
                footer_html = random.choice(footers)

                full_response = (
                    f"{header_html}"
                    f"<p><strong>Tìm thấy {len(matched_users)} nhân viên với tên {name_query}:</strong></p>"
                    f"{user_list_html}"
                    f"{footer_html}"
                )
            else:
                full_response = (
                    f"<p><strong>Không tìm thấy nhân viên nào với tên '{name_query}'.</strong></p>"
                    f"{random.choice(footers)}"
                )
            return JSONResponse(content={"response": full_response})
        # tìm kiếm nhân viên theo email
        if tag == 'tim_kiem_nhan_vien_theo_email':
            email_patterns = [
                r"email là ([\w\.\-]+@[\w\.\-]+)",
                r"email ([\w\.\-]+@[\w\.\-]+)"
            ]
            email = extract_value(email_patterns, message)
            if not email:
                return JSONResponse(content={"response": "Bạn vui lòng cung cấp email nhân viên cần tìm."})

            matched_users = list(user_collection.find(
                {"email": email, "role": {"$in": ['2002', '2006']}},
                {"_id": 1, "name": 1, "email": 1, "phone": 1, "address": 1}
            ))
            headers = [
                "<p><strong>Dưới đây là thông tin nhân viên theo email bạn đã cung cấp:</strong></p>",
                "<p><strong>Kết quả tìm kiếm nhân viên như sau:</strong></p>",
                "<p><strong>Thông tin chi tiết nhân viên được tìm thấy:</strong></p>",
            ]
            footers = [
                "<p><strong>Bạn cần hỗ trợ thêm gì không? Tôi luôn sẵn sàng giúp.</strong></p>",
                "<p><strong>Nếu cần tìm thông tin khác, bạn cứ hỏi nhé!</strong></p>",
                "<p><strong>Bạn còn câu hỏi nào khác không?</strong></p>",
            ]

            if matched_users:
                user_list_html = "<ul>" + "".join(
                    f"<li>Têm: {user.get('name', '-')}</li>"
                    f"<li>Email: {user.get('email', '-')}</li>"
                    f"<li>SĐT: {user.get('phone', '-')}</li>"
                    f"<li>Địa chỉ: {user.get("address", {}).get("detail", "-")}</li>"
                    for user in matched_users
                ) + "</ul>"
                header_html = random.choice(headers)
                footer_html = random.choice(footers)

                full_response = (
                    f"{header_html}"
                    f"<p><strong>Tìm thấy {len(matched_users)} nhân viên với email {email}:</strong></p>"
                    f"{user_list_html}"
                    f"{footer_html}"
                )
            else:
                full_response = (
                    f"<p><strong>Không tìm thấy nhân viên nào với email '{email}'.</strong></p>"
                    f"{random.choice(footers)}"
                )
            return JSONResponse(content={"response": full_response})
        # tìm kiếm nhân viên theo số điện thoại
        if tag == 'tim_kiem_nhan_vien_theo_sdt':
            sdt_patterns = [
                r"số điện thoại là (\d+)",
                r"số điện thoại (\d+)",
                r"sđt (\d+)"
            ]
            phone = extract_value(sdt_patterns, message)
            if not phone:
                return JSONResponse(content={"response": "Bạn vui lòng cung cấp số điện thoại nhân viên cần tìm."})

            matched_users = list(user_collection.find(
                {"phone": phone, "role": {"$in": ['2002', '2006']}},
                {"_id": 1, "name": 1, "email": 1, "phone": 1, "address": 1}
            ))
            headers = [
                "<p><strong>Dưới đây là thông tin nhân viên theo số điện thoại bạn đã cung cấp:</strong></p>",
                "<p><strong>Kết quả tìm kiếm nhân viên như sau:</strong></p>",
                "<p><strong>Thông tin chi tiết nhân viên được tìm thấy:</strong></p>",
            ]
            footers = [
                "<p><strong>Bạn cần hỗ trợ thêm gì không? Tôi luôn sẵn sàng giúp.</strong></p>",
                "<p><strong>Nếu cần tìm thông tin khác, bạn cứ hỏi nhé!</strong></p>",
                "<p><strong>Bạn còn câu hỏi nào khác không?</strong></p>",
            ]

            if matched_users:
                user_list_html = "<ul>" + "".join(
                    f"<li>Têm: {user.get('name', '-')}</li>"
                    f"<li>Email: {user.get('email', '-')}</li>"
                    f"<li>SĐT: {user.get('phone', '-')}</li>"
                    f"<li>Địa chỉ: {user.get("address", {}).get("detail", "-")}</li>"
                    for user in matched_users
                ) + "</ul>"
                header_html = random.choice(headers)
                footer_html = random.choice(footers)

                full_response = (
                    f"{header_html}"
                    f"<p><strong>Tìm thấy {len(matched_users)} nhân viên với số điện thoại {phone}:</strong></p>"
                    f"{user_list_html}"
                    f"{footer_html}"
                )
            else:
                full_response = (
                    f"<p><strong>Không tìm thấy nhân viên nào với số điện thoại '{phone}'.</strong></p>"
                    f"{random.choice(footers)}"
                )
            return JSONResponse(content={"response": full_response})
        # Tìm kiếm nhân viên theo địa chỉ  
        if tag == 'tim_kiem_nhan_vien_theo_dia_chi':
            address_patterns = [
                r"ở (.+)",
                r"địa chỉ (.+)",
                r"sống tại (.+)",
                r"tại (.+)",
                r"cư trú ở (.+)",
            ]
            address_query = extract_value(address_patterns, message)
            if not address_query:
                return JSONResponse(content={"response": "Vui lòng cung cấp địa chỉ cần tìm."})
            matched_users = list(user_collection.find({
                "$or": [
                    {"address.province.name": {"$regex": re.escape(address_query), "$options": "i"}, "role": {"$in": ['2002', '2006']}},
                    {"address.district.name": {"$regex": re.escape(address_query), "$options": "i"}, "role": {"$in": ['2002', '2006']}},
                    {"address.ward.name": {"$regex": re.escape(address_query), "$options": "i"}, "role": {"$in": ['2002', '2006']}},
                    {"address.detail": {"$regex": re.escape(address_query), "$options": "i"}, "role": {"$in": ['2002', '2006']}},
                    {"address.addressAdd": {"$regex": re.escape(address_query), "$options": "i"}, "role": {"$in": ['2002', '2006']}},
                ]
            }, {"_id": 1, "name": 1, "email": 1, "phone": 1, "address": 1}))
            headers = [
                "<p><strong>Dưới đây là thông tin nhân viên theo địa chỉ bạn đã cung cấp:</strong></p>",
                "<p><strong>Kết quả tìm kiếm nhân viên như sau:</strong></p>",
                "<p><strong>Thông tin chi tiết nhân viên được tìm thấy:</strong></p>",
            ]
            footers = [
                "<p><strong>Bạn cần hỗ trợ thêm gì không? Tôi luôn sẵn sàng giúp.</strong></p>",
                "<p><strong>Nếu cần tìm thông tin khác, bạn cứ hỏi nhé!</strong></p>",
                "<p><strong>Bạn còn câu hỏi nào khác không?</strong></p>",
            ]

            if matched_users:
                user_list_html = "<ul>" + "".join(
                    f"<li>Têm: {user.get('name', '-')}</li>"
                    f"<li>Email: {user.get('email', '-')}</li>"
                    f"<li>SĐT: {user.get('phone', '-')}</li>"
                    f"<li>Địa chỉ: {user.get("address", {}).get("detail", "-")}</li>"
                    for user in matched_users
                ) + "</ul>"
                header_html = random.choice(headers)
                footer_html = random.choice(footers)

                full_response = (
                    f"{header_html}"
                    f"<p><strong>Tìm thấy {len(matched_users)} nhân viên địa chỉ {address_query}:</strong></p>"
                    f"{user_list_html}"
                    f"{footer_html}"
                )
            else:
                full_response = (
                    f"<p><strong>Không tìm thấy nhân viên nào địa chỉ '{address_query}'.</strong></p>"
                    f"{random.choice(footers)}"
                )
            return JSONResponse(content={"response": full_response})
        # tìm kiếm đơn hàng đang xử lý
        if tag == 'don_hang_dang_xu_ly':
            order_pending = list(order_collection.find({"status": "Processing"}))
            order_count = len(order_pending)
            if order_count:
                headers = [
                    "Về số lượng đơn hàng trong trạng thái đạng xử lý, hiện nay có",
                    "Tôi sẽ thống kế được số lượng đơn hàng đang xử lý là",
                    "Đơn hàng đang xử lý hiện nay có khoảng",
                ]
                footers = [
                    "Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.",
                    "Nếu có câu hỏi, bạn có thể hỏi?",
                    "Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.",
                    "Ngoài ra, bạn còn câu hỏi nào khác không?",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                response_html = f"<p>{header_html} {order_count}.{footer_html}</p>"
            else:
                response_html = f"Không tim thấy đơn hàng nào với trạng thái đang xử lý. Bạn có câu hỏi nào khác không?"
            return JSONResponse(content={"response": response_html})    

        # Chi tiết đơn hàng đang xử lý
        if tag == 'chi_tiet_don_hang_dang_xu_ly':
            order_pending = list(order_collection.find({"status": "Processing"}))
            order_count = len(order_pending)

            if not order_count:
                return JSONResponse(content={"response": "<p><strong>Không tìm thấy đơn hàng đang xử lý.</strong></p>"})

            response_html = f"<p><strong>Tìm thấy {order_count} đơn hàng trong trạng thái đang xử lý:</strong></p><hr/>"

            for i, order in enumerate(order_pending, 1):
                response_html += f"<div style='margin-bottom:20px;'>"
                response_html += f"<h4>📦 Đơn hàng {i}</h4>"
                response_html += f"<ul>"
                response_html += f"<li><strong>Mã đơn:</strong> {order.get('_id', 'N/A')}</li>"

                # Khách hàng
                order_by_id = order.get('orderBy')
                if order_by_id:
                    try:
                        # Chuyển đổi sang ObjectId nếu cần
                        if isinstance(order_by_id, str):
                            order_by_id = ObjectId(order_by_id)
                        
                        customer = user_collection.find_one({"_id": order_by_id})
                        if customer:
                            customer_name = customer.get('name', 'N/A')
                            customer_phone = customer.get('phone', '')
                            customer_email = customer.get('email', '')
                            
                            customer_info = f"{customer_name}"
                            if customer_phone:
                                customer_info += f" - {customer_phone}"
                            if customer_email:
                                customer_info += f" ({customer_email})"
                            
                            response_html += f"<li><strong>Khách hàng:</strong> {customer_info}</li>"
                            
                            # Hiển thị địa chỉ nếu có
                            customer_address = customer.get('address', '')
                            if customer_address:
                                response_html += f"<li><strong>Địa chỉ khách hàng:</strong> {customer_address}</li>"
                        else:
                            response_html += f"<li><strong>Khách hàng:</strong> Không tìm thấy thông tin</li>"
                    except Exception as e:
                        response_html += f"<li><strong>Khách hàng:</strong> Lỗi truy vấn - {str(e)}</li>"
                else:
                    response_html += f"<li><strong>Khách hàng:</strong> N/A</li>"

                # Tổng tiền
                total = order.get('total', 0)
                response_html += f"<li><strong>Tổng tiền:</strong> {total:,} VNĐ</li>"

                # Trạng thái
                status = order.get('status', 'N/A')
                status_display = {
                    'Processing': 'Đang xử lý',
                    'Cancelled': 'Đã hủy',
                    'Delivering': 'Đang giao',
                    'Succeed': 'Thành công',
                    'Confirm': 'Đã xác nhận'
                }
                response_html += f"<li><strong>Trạng thái:</strong> {status_display.get(status, status)}</li>"

                # Vị trí giao hàng (nếu có)
                location = order.get('location')
                if location and location.get('lat') and location.get('lng'):
                    response_html += f"<li><strong>Vị trí giao hàng:</strong> Lat: {location['lat']}, Lng: {location['lng']}</li>"

                # Voucher áp dụng (nếu có)
                voucher_id = order.get('applyVoucher')
                if voucher_id:
                    response_html += f"<li><strong>Đã áp dụng voucher:</strong> {voucher_id}</li>"

                # Ngày tạo
                created_at = order.get('createdAt', '')
                if created_at:
                    try:
                        if isinstance(created_at, str):
                            created_date = datetime.fromisoformat(str(created_at).replace('Z', '+00:00'))
                        else:
                            created_date = created_at
                        response_html += f"<li><strong>Ngày tạo:</strong> {created_date.strftime('%d/%m/%Y %H:%M')}</li>"
                    except:
                        response_html += f"<li><strong>Ngày tạo:</strong> {created_at}</li>"

                # Danh sách sản phẩm
                products = order.get('products', [])
                if products:
                    response_html += f"<li><strong>Sản phẩm ({len(products)} mặt hàng):</strong><ul>"
                    for j, product in enumerate(products, 1):
                        product_name = product.get('name', 'Tên sản phẩm không có')
                        quantity = product.get('quantity', 0)
                        price = product.get('price', 0)
                        variant = product.get('variant', '')
                        total_product = quantity * price
                        
                        product_info = f"{j}. {product_name}"
                        if variant:
                            product_info += f" ({variant})"
                        product_info += f" - SL: {quantity} | Giá: {price:,} VNĐ | Thành tiền: {total_product:,} VNĐ"
                        
                        response_html += f"<li>{product_info}</li>"
                    response_html += "</ul></li>"
                else:
                    response_html += f"<li><strong>Sản phẩm:</strong> Không có sản phẩm</li>"

                response_html += "</ul></div><hr/>"

            response_html += "<p><strong>Bạn cần thêm thông tin gì nữa không? Tôi sẵn sàng hỗ trợ bạn!</strong></p>"

            return JSONResponse(content={"response": response_html})
        # tìm kiếm đơn hàng đang giao
        if tag == 'don_hang_dang_giao':
            order_pending = list(order_collection.find({"status": "Delivering"}))
            order_count = len(order_pending)
            if order_count:
                headers = [
                    "Về số lượng đơn hàng trong trạng thái đang giao, hiện nay có",
                    "Tôi sẽ thống kế được số lượng đơn hàng đang giao là",
                    "Đơn hàng đang giao hiện nay có khoảng",
                ]
                footers = [
                    "Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.",
                    "Nếu có câu hỏi, bạn có thể hỏi?",
                    "Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.",
                    "Ngoài ra, bạn còn câu hỏi nào khác không?",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                response_html = f"<p>{header_html} {order_count}.{footer_html}</p>"
            else:
                response_html = f"Không tim thấy đơn hàng nào với trạng thái đang giao. Bạn có câu hỏi nào khác không?"
            return JSONResponse(content={"response": response_html})  
        # chi tiết đơn hàng đơn hàng đang giao
        if tag == 'chi_tiet_don_hang_dang_giao':
            order_pending = list(order_collection.find({"status": "Delivering"}))
            order_count = len(order_pending)

            if not order_count:
                return JSONResponse(content={"response": "<p><strong>Không tìm thấy đơn hàng đang giao.</strong></p>"})

            response_html = f"<p><strong>Tìm thấy {order_count} đơn hàng trong trạng thái đang giao:</strong></p><hr/>"

            for i, order in enumerate(order_pending, 1):
                response_html += f"<div style='margin-bottom:20px;'>"
                response_html += f"<h4>📦 Đơn hàng {i}</h4>"
                response_html += f"<ul>"
                response_html += f"<li><strong>Mã đơn:</strong> {order.get('_id', 'N/A')}</li>"

                # Khách hàng
                order_by_id = order.get('orderBy')
                if order_by_id:
                    try:
                        # Chuyển đổi sang ObjectId nếu cần
                        if isinstance(order_by_id, str):
                            order_by_id = ObjectId(order_by_id)
                        
                        customer = user_collection.find_one({"_id": order_by_id})
                        if customer:
                            customer_name = customer.get('name', 'N/A')
                            customer_phone = customer.get('phone', '')
                            customer_email = customer.get('email', '')
                            
                            customer_info = f"{customer_name}"
                            if customer_phone:
                                customer_info += f" - {customer_phone}"
                            if customer_email:
                                customer_info += f" ({customer_email})"
                            
                            response_html += f"<li><strong>Khách hàng:</strong> {customer_info}</li>"
                            
                            # Hiển thị địa chỉ nếu có
                            customer_address = customer.get('address', '')
                            if customer_address:
                                response_html += f"<li><strong>Địa chỉ khách hàng:</strong> {customer_address}</li>"
                        else:
                            response_html += f"<li><strong>Khách hàng:</strong> Không tìm thấy thông tin</li>"
                    except Exception as e:
                        response_html += f"<li><strong>Khách hàng:</strong> Lỗi truy vấn - {str(e)}</li>"
                else:
                    response_html += f"<li><strong>Khách hàng:</strong> N/A</li>"

                # Tổng tiền
                total = order.get('total', 0)
                response_html += f"<li><strong>Tổng tiền:</strong> {total:,} VNĐ</li>"

                # Trạng thái
                status = order.get('status', 'N/A')
                status_display = {
                    'Processing': 'Đang xử lý',
                    'Cancelled': 'Đã hủy',
                    'Delivering': 'Đang giao',
                    'Succeed': 'Thành công',
                    'Confirm': 'Đã xác nhận'
                }
                response_html += f"<li><strong>Trạng thái:</strong> {status_display.get(status, status)}</li>"

                # Vị trí giao hàng (nếu có)
                location = order.get('location')
                if location and location.get('lat') and location.get('lng'):
                    response_html += f"<li><strong>Vị trí giao hàng:</strong> Lat: {location['lat']}, Lng: {location['lng']}</li>"

                # Voucher áp dụng (nếu có)
                voucher_id = order.get('applyVoucher')
                if voucher_id:
                    response_html += f"<li><strong>Đã áp dụng voucher:</strong> {voucher_id}</li>"

                # Ngày tạo
                created_at = order.get('createdAt', '')
                if created_at:
                    try:
                        if isinstance(created_at, str):
                            created_date = datetime.fromisoformat(str(created_at).replace('Z', '+00:00'))
                        else:
                            created_date = created_at
                        response_html += f"<li><strong>Ngày tạo:</strong> {created_date.strftime('%d/%m/%Y %H:%M')}</li>"
                    except:
                        response_html += f"<li><strong>Ngày tạo:</strong> {created_at}</li>"

                # Danh sách sản phẩm
                products = order.get('products', [])
                if products:
                    response_html += f"<li><strong>Sản phẩm ({len(products)} mặt hàng):</strong><ul>"
                    for j, product in enumerate(products, 1):
                        product_name = product.get('name', 'Tên sản phẩm không có')
                        quantity = product.get('quantity', 0)
                        price = product.get('price', 0)
                        variant = product.get('variant', '')
                        total_product = quantity * price
                        
                        product_info = f"{j}. {product_name}"
                        if variant:
                            product_info += f" ({variant})"
                        product_info += f" - SL: {quantity} | Giá: {price:,} VNĐ | Thành tiền: {total_product:,} VNĐ"
                        
                        response_html += f"<li>{product_info}</li>"
                    response_html += "</ul></li>"
                else:
                    response_html += f"<li><strong>Sản phẩm:</strong> Không có sản phẩm</li>"

                response_html += "</ul></div><hr/>"

            response_html += "<p><strong>Bạn cần thêm thông tin gì nữa không? Tôi sẵn sàng hỗ trợ bạn!</strong></p>"

            return JSONResponse(content={"response": response_html})
        # tìm kiếm đơn hàng đã nhận
        if tag == 'don_hang_da_nhan':
            order_pending = list(order_collection.find({"status": "Received"}))
            order_count = len(order_pending)
            if order_count:
                headers = [
                    "Về số lượng đơn hàng trong trạng thái đã nhận, hiện nay có",
                    "Tôi sẽ thống kế được số lượng đơn hàng đã nhận là",
                    "Đơn hàng đã nhận hiện nay có khoảng",
                ]
                footers = [
                    "Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.",
                    "Nếu có câu hỏi, bạn có thể hỏi?",
                    "Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.",
                    "Ngoài ra, bạn còn câu hỏi nào khác không?",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                response_html = f"<p>{header_html} {order_count}.{footer_html}</p>"
            else:
                response_html = f"Không tim thấy đơn hàng nào với trạng thái đã nhận. Bạn có câu hỏi nào khác không?"
            return JSONResponse(content={"response": response_html})  
        # Chi tiết đơn hàng đã nhận
        if tag == 'chi_tiet_don_hang_da_nhan':
            order_pending = list(order_collection.find({"status": "Received"}))
            order_count = len(order_pending)

            if not order_count:
                return JSONResponse(content={"response": "<p><strong>Không tìm thấy đơn hàng đã nhận.</strong></p>"})

            response_html = f"<p><strong>Tìm thấy {order_count} đơn hàng trong trạng thái đã nhận:</strong></p><hr/>"

            for i, order in enumerate(order_pending, 1):
                response_html += f"<div style='margin-bottom:20px;'>"
                response_html += f"<h4>📦 Đơn hàng {i}</h4>"
                response_html += f"<ul>"
                response_html += f"<li><strong>Mã đơn:</strong> {order.get('_id', 'N/A')}</li>"

                # Khách hàng
                order_by_id = order.get('orderBy')
                if order_by_id:
                    try:
                        # Chuyển đổi sang ObjectId nếu cần
                        if isinstance(order_by_id, str):
                            order_by_id = ObjectId(order_by_id)
                        
                        customer = user_collection.find_one({"_id": order_by_id})
                        if customer:
                            customer_name = customer.get('name', 'N/A')
                            customer_phone = customer.get('phone', '')
                            customer_email = customer.get('email', '')
                            
                            customer_info = f"{customer_name}"
                            if customer_phone:
                                customer_info += f" - {customer_phone}"
                            if customer_email:
                                customer_info += f" ({customer_email})"
                            
                            response_html += f"<li><strong>Khách hàng:</strong> {customer_info}</li>"
                            
                            # Hiển thị địa chỉ nếu có
                            customer_address = customer.get('address', '')
                            if customer_address:
                                response_html += f"<li><strong>Địa chỉ khách hàng:</strong> {customer_address}</li>"
                        else:
                            response_html += f"<li><strong>Khách hàng:</strong> Không tìm thấy thông tin</li>"
                    except Exception as e:
                        response_html += f"<li><strong>Khách hàng:</strong> Lỗi truy vấn - {str(e)}</li>"
                else:
                    response_html += f"<li><strong>Khách hàng:</strong> N/A</li>"

                # Tổng tiền
                total = order.get('total', 0)
                response_html += f"<li><strong>Tổng tiền:</strong> {total:,} VNĐ</li>"

                # Trạng thái
                status = order.get('status', 'N/A')
                status_display = {
                    'Processing': 'Đang xử lý',
                    'Cancelled': 'Đã hủy',
                    'Delivering': 'Đang giao',
                    'Succeed': 'Thành công',
                    'Confirm': 'Đã xác nhận'
                }
                response_html += f"<li><strong>Trạng thái:</strong> {status_display.get(status, status)}</li>"

                # Vị trí giao hàng (nếu có)
                location = order.get('location')
                if location and location.get('lat') and location.get('lng'):
                    response_html += f"<li><strong>Vị trí giao hàng:</strong> Lat: {location['lat']}, Lng: {location['lng']}</li>"

                # Voucher áp dụng (nếu có)
                voucher_id = order.get('applyVoucher')
                if voucher_id:
                    response_html += f"<li><strong>Đã áp dụng voucher:</strong> {voucher_id}</li>"

                # Ngày tạo
                created_at = order.get('createdAt', '')
                if created_at:
                    try:
                        if isinstance(created_at, str):
                            created_date = datetime.fromisoformat(str(created_at).replace('Z', '+00:00'))
                        else:
                            created_date = created_at
                        response_html += f"<li><strong>Ngày tạo:</strong> {created_date.strftime('%d/%m/%Y %H:%M')}</li>"
                    except:
                        response_html += f"<li><strong>Ngày tạo:</strong> {created_at}</li>"

                # Danh sách sản phẩm
                products = order.get('products', [])
                if products:
                    response_html += f"<li><strong>Sản phẩm ({len(products)} mặt hàng):</strong><ul>"
                    for j, product in enumerate(products, 1):
                        product_name = product.get('name', 'Tên sản phẩm không có')
                        quantity = product.get('quantity', 0)
                        price = product.get('price', 0)
                        variant = product.get('variant', '')
                        total_product = quantity * price
                        
                        product_info = f"{j}. {product_name}"
                        if variant:
                            product_info += f" ({variant})"
                        product_info += f" - SL: {quantity} | Giá: {price:,} VNĐ | Thành tiền: {total_product:,} VNĐ"
                        
                        response_html += f"<li>{product_info}</li>"
                    response_html += "</ul></li>"
                else:
                    response_html += f"<li><strong>Sản phẩm:</strong> Không có sản phẩm</li>"

                response_html += "</ul></div><hr/>"

            response_html += "<p><strong>Bạn cần thêm thông tin gì nữa không? Tôi sẵn sàng hỗ trợ bạn!</strong></p>"

            return JSONResponse(content={"response": response_html})
        # tìm kiếm đơn hàng đã hủy
        if tag == 'don_hang_da_huy':
            order_pending = list(order_collection.find({"status": "Cancelled"}))
            order_count = len(order_pending)
            if order_count:
                headers = [
                    "Về số lượng đơn hàng trong trạng thái đã hủy, hiện nay có",
                    "Tôi sẽ thống kế được số lượng đơn hàng đã hủy là",
                    "Đơn hàng đã hủy hiện nay có khoảng",
                ]
                footers = [
                    "Bạn còn câu hỏi nào không? Tôi luôn sẵn sàng hỗ trợ.",
                    "Nếu có câu hỏi, bạn có thể hỏi?",
                    "Nếu có câu hỏi nào khác, bạn có thể hỏi tôi.",
                    "Ngoài ra, bạn còn câu hỏi nào khác không?",
                ]
                header_html = random.choice(headers)
                footer_html = random.choice(footers)
                response_html = f"<p>{header_html} {order_count}.{footer_html}</p>"
            else:
                response_html = f"Không tim thấy đơn hàng nào với trạng thái đã hủy. Bạn có câu hỏi nào khác không?"
            return JSONResponse(content={"response": response_html})  
        # Thông tin chi tiết đơn hàng đã hủy
        if tag == 'chi_tiet_don_hang_da_huy':
            order_pending = list(order_collection.find({"status": "Cancelled"}))
            order_count = len(order_pending)

            if not order_count:
                return JSONResponse(content={"response": "<p><strong>Không tìm thấy đơn hàng đã nhận.</strong></p>"})

            response_html = f"<p><strong>Tìm thấy {order_count} đơn hàng trong trạng thái đã nhận:</strong></p><hr/>"

            for i, order in enumerate(order_pending, 1):
                response_html += f"<div style='margin-bottom:20px;'>"
                response_html += f"<h4>📦 Đơn hàng {i}</h4>"
                response_html += f"<ul>"
                response_html += f"<li><strong>Mã đơn:</strong> {order.get('_id', 'N/A')}</li>"

                # Khách hàng
                order_by_id = order.get('orderBy')
                if order_by_id:
                    try:
                        # Chuyển đổi sang ObjectId nếu cần
                        if isinstance(order_by_id, str):
                            order_by_id = ObjectId(order_by_id)
                        
                        customer = user_collection.find_one({"_id": order_by_id})
                        if customer:
                            customer_name = customer.get('name', 'N/A')
                            customer_phone = customer.get('phone', '')
                            customer_email = customer.get('email', '')
                            
                            customer_info = f"{customer_name}"
                            if customer_phone:
                                customer_info += f" - {customer_phone}"
                            if customer_email:
                                customer_info += f" ({customer_email})"
                            
                            response_html += f"<li><strong>Khách hàng:</strong> {customer_info}</li>"
                            
                            # Hiển thị địa chỉ nếu có
                            customer_address = customer.get('address', '')
                            if customer_address:
                                response_html += f"<li><strong>Địa chỉ khách hàng:</strong> {customer_address}</li>"
                        else:
                            response_html += f"<li><strong>Khách hàng:</strong> Không tìm thấy thông tin</li>"
                    except Exception as e:
                        response_html += f"<li><strong>Khách hàng:</strong> Lỗi truy vấn - {str(e)}</li>"
                else:
                    response_html += f"<li><strong>Khách hàng:</strong> N/A</li>"

                # Tổng tiền
                total = order.get('total', 0)
                response_html += f"<li><strong>Tổng tiền:</strong> {total:,} VNĐ</li>"

                # Trạng thái
                status = order.get('status', 'N/A')
                status_display = {
                    'Processing': 'Đang xử lý',
                    'Cancelled': 'Đã hủy',
                    'Delivering': 'Đang giao',
                    'Succeed': 'Thành công',
                    'Confirm': 'Đã xác nhận'
                }
                response_html += f"<li><strong>Trạng thái:</strong> {status_display.get(status, status)}</li>"

                # Vị trí giao hàng (nếu có)
                location = order.get('location')
                if location and location.get('lat') and location.get('lng'):
                    response_html += f"<li><strong>Vị trí giao hàng:</strong> Lat: {location['lat']}, Lng: {location['lng']}</li>"

                # Voucher áp dụng (nếu có)
                voucher_id = order.get('applyVoucher')
                if voucher_id:
                    response_html += f"<li><strong>Đã áp dụng voucher:</strong> {voucher_id}</li>"

                # Ngày tạo
                created_at = order.get('createdAt', '')
                if created_at:
                    try:
                        if isinstance(created_at, str):
                            created_date = datetime.fromisoformat(str(created_at).replace('Z', '+00:00'))
                        else:
                            created_date = created_at
                        response_html += f"<li><strong>Ngày tạo:</strong> {created_date.strftime('%d/%m/%Y %H:%M')}</li>"
                    except:
                        response_html += f"<li><strong>Ngày tạo:</strong> {created_at}</li>"

                # Danh sách sản phẩm
                products = order.get('products', [])
                if products:
                    response_html += f"<li><strong>Sản phẩm ({len(products)} mặt hàng):</strong><ul>"
                    for j, product in enumerate(products, 1):
                        product_name = product.get('name', 'Tên sản phẩm không có')
                        quantity = product.get('quantity', 0)
                        price = product.get('price', 0)
                        variant = product.get('variant', '')
                        total_product = quantity * price
                        
                        product_info = f"{j}. {product_name}"
                        if variant:
                            product_info += f" ({variant})"
                        product_info += f" - SL: {quantity} | Giá: {price:,} VNĐ | Thành tiền: {total_product:,} VNĐ"
                        
                        response_html += f"<li>{product_info}</li>"
                    response_html += "</ul></li>"
                else:
                    response_html += f"<li><strong>Sản phẩm:</strong> Không có sản phẩm</li>"

                response_html += "</ul></div><hr/>"

            response_html += "<p><strong>Bạn cần thêm thông tin gì nữa không? Tôi sẵn sàng hỗ trợ bạn!</strong></p>"

            return JSONResponse(content={"response": response_html})
        # Doanh thu theo ngày
        if tag == 'doanh_thu_theo_ngay':
            try:
                # Extract ngày từ message
                day_patterns = [
                    r"doanh thu theo ngày (.+)",
                    r"doanh thu theo ngày (.+) là bao nhiêu",
                    r"ngày (.+)",
                    r"ngày (.+) là bao nhiêu",
                    r"doanh thu ngày (.+) là bao nhiêu", 
                ]
                day_query = extract_value(day_patterns, message)                
                target_date = parse_date_from_query(day_query)
                
                if not target_date:
                    error_responses = [
                        "Không thể hiểu định dạng ngày. Vui lòng nhập theo format: DD/MM/YYYY, DD/MM hoặc 'hôm nay'",
                        "Format ngày không đúng. Bạn có thể nhập như: 15/7, 15/7/2025, hôm nay, hôm qua",
                        "Tôi không hiểu ngày bạn muốn xem. Vui lòng thử lại với format: DD/MM hoặc DD/MM/YYYY"
                    ]
                    return JSONResponse(content={"response": random.choice(error_responses)})
                
                # Tạo start và end time cho ngày cụ thể
                start_of_day = datetime.combine(target_date, datetime.min.time())
                end_of_day = start_of_day + timedelta(days=1)
                
                # Query đơn hàng trong ngày cụ thể (chỉ tính đơn thành công)
                orders = list(order_collection.find({
                    "status": {"$in": ["Succeed", "Delivering"]},  # Chỉ tính đơn thành công và đang giao
                    "createdAt": { 
                        "$gte": start_of_day,  
                        "$lt": end_of_day     
                    }
                }))
                
                # Query phiếu xuất trong ngày cụ thể (warehouse với type = 'export')
                warehouses = list(warehouses_collection.find({
                    "type": "export",
                    "createdAt": { 
                        "$gte": start_of_day,  
                        "$lt": end_of_day     
                    }
                }))

                # Format ngày để hiển thị
                formatted_date = target_date.strftime("%d/%m/%Y")

                if not orders and not warehouses:
                    response = [
                        f"Không tìm thấy doanh thu trong ngày {formatted_date}. Bạn còn câu hỏi nào khác không, tôi luôn sẵn sàng trả lời?",
                        f"Chưa có thông tin về doanh thu trong ngày {formatted_date}. Đồng nghĩa với việc không có đơn hàng và phiếu xuất nào được tạo. Nếu có câu hỏi khác, tôi sẵn sàng hỗ trợ bạn.",
                        f"Doanh thu ngày {formatted_date} là 0 VNĐ. Bạn có thể tìm kiếm thêm thông tin về doanh thu của những ngày trước đó. Chào bạn nha!!!"
                    ]
                    return JSONResponse(content={"response": random.choice(response)})

                # Tính tổng doanh thu ngày
                total_order_revenue = sum(order.get("total", 0) for order in orders)
                total_warehouse_revenue = sum(wh.get("total", 0) for wh in warehouses)
                total_revenue = total_order_revenue + total_warehouse_revenue
                
                # Hiển thị trạng thái
                status_mapping = {
                    "Processing": "Đang xử lý",
                    "Delivering": "Đang giao hàng",
                    "Succeed": "Thành công",
                    "Cancelled": "Đã hủy",
                    "Confirm": "Đã xác nhận"
                }
                
                def create_detailed_response():
                    html_content = f"""
                    <div style="background: #f8f9fa; border-radius: 8px; padding: 20px;">
                        <h2 style="color: #2c3e50; margin-bottom: 20px;">📊 Báo cáo doanh thu ngày {formatted_date}</h2>
                        
                        <div style="background: white; padding: 15px; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #27ae60; margin-top: 0;">💰 Tổng quan doanh thu</h3>
                            <p><strong>Tổng doanh thu: </strong><span style="color: #e74c3c; font-size: 18px; font-weight: bold;">{total_revenue:,} VNĐ</span></p>
                            <p>• Doanh thu từ đơn hàng: {total_order_revenue:,} VNĐ ({len(orders)} đơn)</p>
                            <p>• Doanh thu từ phiếu xuất: {total_warehouse_revenue:,} VNĐ ({len(warehouses)} phiếu)</p>
                        </div>
                    """
                    
                    # Chi tiết đơn hàng
                    if orders:
                        html_content += """
                        <div style="background: white; padding: 15px; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #3498db; margin-top: 0;">🛒 Chi tiết đơn hàng</h3>
                        """
                        for idx, order in enumerate(orders, 1):
                            order_time = order.get('createdAt', '').strftime('%H:%M') if isinstance(order.get('createdAt'), datetime) else 'N/A'
                            order_status = order.get('status', 'N/A')
                            status_color = {
                                'Succeed': '#27ae60',
                                'Processing': '#f39c12', 
                                'Delivering': '#3498db',
                                'Cancelled': '#e74c3c',
                                'Confirm': '#9b59b6'
                            }.get(order_status, '#95a5a6')
                            
                            html_content += f"""
                            <div style="border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: #2c3e50;">Đơn hàng #{idx}</h4>
                                    <span style="background: {status_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                        {status_mapping.get(order_status, order_status)}
                                    </span>
                                </div>
                                <p style="margin: 5px 0; color: #7f8c8d;">⏰ Thời gian: {order_time}</p>
                                <p style="margin: 5px 0;"><strong>Tổng tiền: {order.get('total', 0):,} VNĐ</strong></p>
                            """
                            
                            # Thông tin khách hàng
                            order_by_id = order.get('orderBy')
                            if order_by_id:
                                try:
                                    if isinstance(order_by_id, str):
                                        order_by_id = ObjectId(order_by_id)
                                    customer = user_collection.find_one({"_id": order_by_id})
                                    if customer:
                                        customer_name = customer.get('name', 'N/A')
                                        customer_phone = customer.get('phone', '')
                                        html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>👤 Khách hàng: {customer_name}"
                                        if customer_phone:
                                            html_content += f" - {customer_phone}"
                                        html_content += "</p>"
                                except:
                                    pass
                            
                            # Voucher (nếu có)
                            voucher_id = order.get('applyVoucher')
                            if voucher_id:
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>🎫 Đã áp dụng voucher</p>"
                            
                            # Thông tin sản phẩm trong đơn hàng
                            products = order.get('products', [])
                            if products:
                                html_content += "<h5 style='margin: 10px 0 5px 0; color: #34495e;'>📦 Sản phẩm:</h5>"
                                html_content += "<div style='background: #f8f9fa; padding: 8px; border-radius: 4px;'>"
                                
                                for product in products:
                                    product_name = product.get('name', 'Sản phẩm không rõ tên')
                                    quantity = product.get('quantity', 0)
                                    price = product.get('price', 0)
                                    variant = product.get('variant', '')
                                    subtotal = quantity * price
                                    
                                    html_content += f"""
                                    <div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px;">
                                        <div style="font-weight: bold; color: #2c3e50; margin-bottom: 3px;">
                                            {product_name}
                                            {f" ({variant})" if variant else ""}
                                        </div>
                                        <div style="font-size: 14px; color: #7f8c8d;">
                                            Số lượng: {quantity} × {price:,} VNĐ = <strong style="color: #e74c3c;">{subtotal:,} VNĐ</strong>
                                        </div>
                                    </div>
                                    """
                                
                                html_content += "</div>"
                            
                            html_content += "</div>"
                        
                        html_content += "</div>"
                    
                    # Chi tiết phiếu xuất
                    if warehouses:
                        html_content += """
                        <div style="background: white; padding: 15px; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #9b59b6; margin-top: 0;">📋 Chi tiết phiếu xuất</h3>
                        """
                        for idx, warehouse in enumerate(warehouses, 1):
                            wh_time = warehouse.get('createdAt', '').strftime('%H:%M') if isinstance(warehouse.get('createdAt'), datetime) else 'N/A'
                            
                            html_content += f"""
                            <div style="border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: #2c3e50;">Phiếu xuất #{idx}</h4>
                                    <span style="background: #9b59b6; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                        Xuất kho
                                    </span>
                                </div>
                                <p style="margin: 5px 0; color: #7f8c8d;">⏰ Thời gian: {wh_time}</p>
                            """
                            
                            # Người xử lý
                            handled_by = warehouse.get('handledBy', 'N/A')
                            html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>👤 Người xử lý: {handled_by}</p>"
                            
                            # Nhà cung cấp
                            supplier = warehouse.get('supplier', '')
                            if supplier:
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>🏢 Nhà cung cấp: {supplier}</p>"
                            
                            # Thông tin xuất đến
                            exported_to = warehouse.get('exportedTo', {})
                            if exported_to and exported_to.get('name'):
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>📍 Xuất đến: {exported_to.get('name', '')}"
                                if exported_to.get('phone'):
                                    html_content += f" - {exported_to.get('phone')}"
                                html_content += "</p>"
                                if exported_to.get('address'):
                                    html_content += f"<p style='margin: 5px 0; color: #7f8c8d; font-size: 13px;'>   Địa chỉ: {exported_to.get('address')}</p>"
                            
                            html_content += f"<p style='margin: 5px 0;'><strong>Tổng tiền: {warehouse.get('total', 0):,} VNĐ</strong></p>"
                            
                            # Thông tin sản phẩm trong phiếu xuất
                            products = warehouse.get('products', [])
                            if products:
                                html_content += "<h5 style='margin: 10px 0 5px 0; color: #34495e;'>📦 Sản phẩm:</h5>"
                                html_content += "<div style='background: #f8f9fa; padding: 8px; border-radius: 4px;'>"
                                
                                for product in products:
                                    product_name = product.get('name', 'Sản phẩm không rõ tên')
                                    quantity = product.get('quantity', 0)
                                    price = product.get('price', 0)
                                    variant = product.get('variant', '')
                                    total_price = product.get('totalPrice', quantity * price)
                                    
                                    html_content += f"""
                                    <div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px;">
                                        <div style="font-weight: bold; color: #2c3e50; margin-bottom: 3px;">
                                            {product_name}
                                            {f" ({variant})" if variant else ""}
                                        </div>
                                        <div style="font-size: 14px; color: #7f8c8d;">
                                            Số lượng: {quantity} × {price:,} VNĐ = <strong style="color: #e74c3c;">{total_price:,} VNĐ</strong>
                                        </div>
                                    </div>
                                    """
                                
                                html_content += "</div>"
                            
                            html_content += "</div>"
                        
                        html_content += "</div>"
                    
                    html_content += """
                        <div style="text-align: center; margin-top: 20px; padding: 15px; background: #ecf0f1; border-radius: 6px;">
                            <p style="margin: 0; color: #7f8c8d;">Hãy nhập thêm câu hỏi nếu có, tôi luôn sẵn sàng giải đáp các thắc mắc của bạn</p>
                        </div>
                    </div>
                    """
                    
                    return html_content
                
                # Kiểm tra xem message có chứa từ "chi tiết" không
                show_detail = any(keyword in message.lower() for keyword in ["chi tiết", "thông tin cụ thể", "doanh thu cụ thể"])
                
                if show_detail:
                    # Hiển thị chi tiết đầy đủ
                    response = create_detailed_response()
                else:
                    # Hiển thị thông tin tổng quan
                    responses = [
                        f"Doanh thu ngày {formatted_date} là: {total_revenue:,} VNĐ. Trong đó doanh thu từ đơn hàng là {total_order_revenue:,} VNĐ, doanh thu từ phiếu xuất là {total_warehouse_revenue:,} VNĐ. Ngoài ra, bạn còn câu hỏi nào khác không?",
                        f"Tổng doanh thu ngày {formatted_date}: {total_revenue:,} VNĐ. Doanh thu từ đơn hàng là {total_order_revenue:,} VNĐ. Doanh thu từ phiếu xuất là {total_warehouse_revenue:,} VNĐ. Bạn cần tôi giải đáp gì nữa không?",
                        (
                            f"<p>Doanh thu ngày {formatted_date}:</p>"
                            f"<ul>"
                            f"<li>Doanh thu đơn hàng: {total_order_revenue:,} VNĐ ({len(orders)} đơn)</li>"
                            f"<li>Doanh thu phiếu xuất sản phẩm: {total_warehouse_revenue:,} VNĐ ({len(warehouses)} phiếu)</li>"
                            f"<li><strong>Tổng cộng: {total_revenue:,} VNĐ</strong></li>"
                            f"</ul>"
                            f"<p>Bạn có muốn xem chi tiết đơn hàng không? Chỉ cần nói 'doanh thu chi tiết ngày {formatted_date}'</p>"
                        )
                    ]
                    response = random.choice(responses)
                
                return JSONResponse(content={"response": response})

            except Exception as e:
                return JSONResponse(content={"response": f"Đã xảy ra lỗi khi truy vấn doanh thu theo ngày: {str(e)}"})
        # Doanh thu tháng
        if tag == 'doanh_thu_theo_thang':
            try:
                month_patterns = [
                    r"doanh thu theo tháng (.+)",
                    r"doanh thu tháng (.+)",
                    r"tháng (.+)",
                    r"tháng (.+) là bao nhiêu",
                    r"doanh thu tháng (.+) là bao nhiêu",
                ]
                
                month_query = extract_value(month_patterns, message)
                # Parse tháng từ query
                parsed_month = parse_month_from_query(month_query)
                
                if not parsed_month:
                    error_responses = [
                        "Không thể hiểu định dạng tháng. Vui lòng nhập theo format: MM/YYYY, MM hoặc 'tháng này'",
                        "Format tháng không đúng. Bạn có thể nhập như: 7/2025, 7, tháng này, tháng trước",
                    ]
                    return JSONResponse(content={"response": random.choice(error_responses)})
                
                year, month = parsed_month
                
                # Tạo start và end time cho tháng cụ thể
                start_of_month = datetime(year, month, 1)
                if month == 12:
                    end_of_month = datetime(year + 1, 1, 1)
                else:
                    end_of_month = datetime(year, month + 1, 1)
                
                # Query đơn hàng trong tháng cụ thể (chỉ tính đơn thành công)
                orders = list(order_collection.find({
                    "status": {"$in": ["Succeed", "Delivering"]},
                    "createdAt": { 
                        "$gte": start_of_month,
                        "$lt": end_of_month
                    }
                }))
                
                # Query phiếu xuất trong tháng cụ thể (warehouse với type = 'export')
                warehouses = list(warehouses_collection.find({
                    "type": "export",
                    "createdAt": { 
                        "$gte": start_of_month,
                        "$lt": end_of_month
                    }
                }))

                # Format tháng để hiển thị
                formatted_month = f"{month:02d}/{year}"

                if not orders and not warehouses:
                    response = [
                        f"Không tìm thấy bất kỳ khoản doanh thu nào trong tháng {formatted_month}. Bạn còn câu hỏi nào khác không?",
                        f"Chưa có doanh thu nào được tạo trong tháng {formatted_month}. Nếu có câu hỏi khác, tôi sẵn sàng hỗ trợ bạn.",
                        f"Doanh thu tháng {formatted_month} là 0 VNĐ. Nếu có câu hỏi khác, tôi sẽ giải đáp giúp bạn"
                    ]
                    return JSONResponse(content={"response": random.choice(response)})

                # Tính tổng doanh thu tháng
                total_order_revenue = sum(order.get("total", 0) for order in orders)
                total_warehouse_revenue = sum(wh.get("total", 0) for wh in warehouses)
                total_revenue = total_order_revenue + total_warehouse_revenue
                
                # Hiển thị trạng thái
                status_mapping = {
                    "Processing": "Đang xử lý",
                    "Delivering": "Đang giao hàng",
                    "Succeed": "Thành công",
                    "Cancelled": "Đã hủy",
                    "Confirm": "Đã xác nhận"
                }
                
                def create_detailed_response():
                    html_content = f"""
                    <div style="background: #f8f9fa; border-radius: 8px; padding: 20px;">
                        <h2 style="color: #2c3e50; margin-bottom: 20px;">📊 Báo cáo doanh thu tháng {formatted_month}</h2>
                        
                        <div style="background: white; padding: 15px; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #27ae60; margin-top: 0;">💰 Tổng quan doanh thu</h3>
                            <p><strong>Tổng doanh thu: </strong><span style="color: #e74c3c; font-size: 18px; font-weight: bold;">{total_revenue:,} VNĐ</span></p>
                            <p>• Doanh thu từ đơn hàng: {total_order_revenue:,} VNĐ ({len(orders)} đơn)</p>
                            <p>• Doanh thu từ phiếu xuất: {total_warehouse_revenue:,} VNĐ ({len(warehouses)} phiếu)</p>
                        </div>
                    """
                    
                    # Chi tiết đơn hàng
                    if orders:
                        html_content += """
                        <div style="background: white; padding: 15px; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #3498db; margin-top: 0;">🛒 Chi tiết đơn hàng</h3>
                        """
                        for idx, order in enumerate(orders, 1):
                            order_date = order.get('createdAt')
                            if isinstance(order_date, datetime):
                                order_time = order_date.strftime('%d/%m/%Y %H:%M')
                            else:
                                order_time = 'N/A'
                            
                            order_status = order.get('status', 'N/A')
                            status_color = {
                                'Succeed': '#27ae60',
                                'Processing': '#f39c12', 
                                'Delivering': '#3498db',
                                'Cancelled': '#e74c3c',
                                'Confirm': '#9b59b6'
                            }.get(order_status, '#95a5a6')
                            
                            html_content += f"""
                            <div style="border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: #2c3e50;">Đơn hàng #{idx}</h4>
                                    <span style="background: {status_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                        {status_mapping.get(order_status, order_status)}
                                    </span>
                                </div>
                                <p style="margin: 5px 0; color: #7f8c8d;">⏰ Thời gian: {order_time}</p>
                                <p style="margin: 5px 0;"><strong>Tổng tiền: {order.get('total', 0):,} VNĐ</strong></p>
                            """
                            
                            # Thông tin khách hàng
                            order_by_id = order.get('orderBy')
                            if order_by_id:
                                try:
                                    if isinstance(order_by_id, str):
                                        order_by_id = ObjectId(order_by_id)
                                    customer = user_collection.find_one({"_id": order_by_id})
                                    if customer:
                                        customer_name = customer.get('name', 'N/A')
                                        customer_phone = customer.get('phone', '')
                                        html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>👤 Khách hàng: {customer_name}"
                                        if customer_phone:
                                            html_content += f" - {customer_phone}"
                                        html_content += "</p>"
                                except:
                                    pass
                            
                            # Voucher (nếu có)
                            voucher_id = order.get('applyVoucher')
                            if voucher_id:
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>🎫 Đã áp dụng voucher</p>"
                            
                            # Vị trí giao hàng
                            location = order.get('location')
                            if location and location.get('lat') and location.get('lng'):
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d; font-size: 13px;'>📍 Vị trí: ({location['lat']}, {location['lng']})</p>"
                            
                            # Thông tin sản phẩm trong đơn hàng
                            products = order.get('products', [])
                            if products:
                                html_content += "<h5 style='margin: 10px 0 5px 0; color: #34495e;'>📦 Sản phẩm:</h5>"
                                html_content += "<div style='background: #f8f9fa; padding: 8px; border-radius: 4px;'>"
                                
                                for product in products:
                                    product_name = product.get('name', 'Sản phẩm không rõ tên')
                                    quantity = product.get('quantity', 0)
                                    price = product.get('price', 0)
                                    variant = product.get('variant', '')
                                    subtotal = quantity * price
                                    
                                    html_content += f"""
                                    <div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px;">
                                        <div style="font-weight: bold; color: #2c3e50; margin-bottom: 3px;">
                                            {product_name}
                                            {f" ({variant})" if variant else ""}
                                        </div>
                                        <div style="font-size: 14px; color: #7f8c8d;">
                                            Số lượng: {quantity} × {price:,} VNĐ = <strong style="color: #e74c3c;">{subtotal:,} VNĐ</strong>
                                        </div>
                                    </div>
                                    """
                                
                                html_content += "</div>"
                            
                            html_content += "</div>"
                        
                        html_content += "</div>"
                    
                    # Chi tiết phiếu xuất
                    if warehouses:
                        html_content += """
                        <div style="background: white; padding: 15px; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #9b59b6; margin-top: 0;">📋 Chi tiết phiếu xuất</h3>
                        """
                        for idx, warehouse in enumerate(warehouses, 1):
                            wh_date = warehouse.get('createdAt')
                            if isinstance(wh_date, datetime):
                                wh_time = wh_date.strftime('%d/%m/%Y %H:%M')
                            else:
                                wh_time = 'N/A'
                            
                            html_content += f"""
                            <div style="border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: #2c3e50;">Phiếu xuất #{idx}</h4>
                                    <span style="background: #9b59b6; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                        Xuất kho
                                    </span>
                                </div>
                                <p style="margin: 5px 0; color: #7f8c8d;">⏰ Thời gian: {wh_time}</p>
                            """
                            
                            # Người xử lý
                            handled_by = warehouse.get('handledBy', 'N/A')
                            html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>👤 Người xử lý: {handled_by}</p>"
                            
                            # Nhà cung cấp
                            supplier = warehouse.get('supplier', '')
                            if supplier:
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>🏢 Nhà cung cấp: {supplier}</p>"
                            
                            # Thông tin xuất đến
                            exported_to = warehouse.get('exportedTo', {})
                            if exported_to and exported_to.get('name'):
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>📍 Xuất đến: {exported_to.get('name', '')}"
                                if exported_to.get('phone'):
                                    html_content += f" - {exported_to.get('phone')}"
                                html_content += "</p>"
                                if exported_to.get('address'):
                                    html_content += f"<p style='margin: 5px 0; color: #7f8c8d; font-size: 13px;'>   Địa chỉ: {exported_to.get('address')}</p>"
                                if exported_to.get('email'):
                                    html_content += f"<p style='margin: 5px 0; color: #7f8c8d; font-size: 13px;'>   Email: {exported_to.get('email')}</p>"
                            
                            html_content += f"<p style='margin: 5px 0;'><strong>Tổng tiền: {warehouse.get('total', 0):,} VNĐ</strong></p>"
                            
                            # Thông tin sản phẩm trong phiếu xuất
                            products = warehouse.get('products', [])
                            if products:
                                html_content += "<h5 style='margin: 10px 0 5px 0; color: #34495e;'>📦 Sản phẩm:</h5>"
                                html_content += "<div style='background: #f8f9fa; padding: 8px; border-radius: 4px;'>"
                                
                                for product in products:
                                    product_name = product.get('name', 'Sản phẩm không rõ tên')
                                    quantity = product.get('quantity', 0)
                                    price = product.get('price', 0)
                                    variant = product.get('variant', '')
                                    total_price = product.get('totalPrice', quantity * price)
                                    
                                    html_content += f"""
                                    <div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px;">
                                        <div style="font-weight: bold; color: #2c3e50; margin-bottom: 3px;">
                                            {product_name}
                                            {f" ({variant})" if variant else ""}
                                        </div>
                                        <div style="font-size: 14px; color: #7f8c8d;">
                                            Số lượng: {quantity} × {price:,} VNĐ = <strong style="color: #e74c3c;">{total_price:,} VNĐ</strong>
                                        </div>
                                    </div>
                                    """
                                
                                html_content += "</div>"
                            
                            html_content += "</div>"
                        
                        html_content += "</div>"
                    
                    html_content += """
                        <div style="text-align: center; margin-top: 20px; padding: 15px; background: #ecf0f1; border-radius: 6px;">
                            <p style="margin: 0; color: #7f8c8d;">Hãy nhập thêm câu hỏi nếu có, tôi luôn sẵn sàng giải đáp các thắc mắc của bạn</p>
                        </div>
                    </div>
                    """
                    
                    return html_content
                
                # Kiểm tra xem message có chứa từ "chi tiết" không
                show_detail = any(keyword in message.lower() for keyword in ["chi tiết", "thông tin cụ thể", "doanh thu cụ thể"])
                
                if show_detail:
                    # Hiển thị chi tiết đầy đủ
                    response = create_detailed_response()
                else:
                    # Hiển thị thông tin tổng quan
                    responses = [
                        f"Doanh thu tháng {formatted_month} là: {total_revenue:,} VNĐ. Trong đó doanh thu từ đơn hàng là {total_order_revenue:,} VNĐ, doanh thu từ phiếu xuất là {total_warehouse_revenue:,} VNĐ. Ngoài ra, bạn còn câu hỏi nào khác không?",
                        f"Tổng doanh thu tháng {formatted_month}: {total_revenue:,} VNĐ. Doanh thu từ đơn hàng là {total_order_revenue:,} VNĐ. Doanh thu từ phiếu xuất là {total_warehouse_revenue:,} VNĐ. Bạn cần tôi giải đáp gì nữa không?",
                        (
                            f"<p>Doanh thu tháng {formatted_month}:</p>"
                            f"<ul>"
                            f"<li>Doanh thu đơn hàng: {total_order_revenue:,} VNĐ ({len(orders)} đơn)</li>"
                            f"<li>Doanh thu phiếu xuất sản phẩm: {total_warehouse_revenue:,} VNĐ ({len(warehouses)} phiếu)</li>"
                            f"<li><strong>Tổng cộng: {total_revenue:,} VNĐ</strong></li>"
                            f"</ul>"
                            f"<p>Bạn có muốn xem chi tiết doanh thu theo ngày không? Ví dụ: 'doanh thu chi tiết tháng {formatted_month}'</p>"
                        )
                    ]
                    response = random.choice(responses)
                
                return JSONResponse(content={"response": response})
                
            except Exception as e:
                return JSONResponse(content={"response": f"Đã xảy ra lỗi khi truy vấn doanh thu theo tháng: {str(e)}"})
        # Doanh thu năm
        if tag == 'doanh_thu_theo_nam':
            try:
                # Extract năm từ message
                year_patterns = [
                    r"doanh thu theo năm (.+)",
                    r"doanh thu năm (.+)",
                    r"năm (.+) là bao nhiêu",
                    r"doanh thu năm (.+) là bao nhiêu",
                    r"doanh thu của năm (.+)",
                    r"năm (.+)",
                ]
                
                year_query = extract_value(year_patterns, message)
                # Parse năm từ query
                target_year = parse_year_from_query(year_query)
                
                if not target_year:
                    error_responses = [
                        "Không thể hiểu định dạng năm. Vui lòng nhập theo format: YYYY hoặc 'năm này'",
                        "Format năm không đúng. Bạn có thể nhập như: 2024, 2025, năm này, năm trước",
                        "Tôi không hiểu năm bạn muốn xem. Vui lòng thử lại với format: YYYY"
                    ]
                    return JSONResponse(content={"response": random.choice(error_responses)})
                
                # Tạo start và end time cho năm cụ thể
                start_of_year = datetime(target_year, 1, 1)
                end_of_year = datetime(target_year + 1, 1, 1)
                
                # Query đơn hàng trong năm cụ thể (chỉ tính đơn thành công)
                orders = list(order_collection.find({
                    "status": {"$in": ["Succeed", "Delivering"]},
                    "createdAt": { 
                        "$gte": start_of_year,
                        "$lt": end_of_year
                    }
                }))
                
                # Query phiếu xuất trong năm cụ thể (warehouse với type = 'export')
                warehouses = list(warehouses_collection.find({
                    "type": "export",
                    "createdAt": { 
                        "$gte": start_of_year,
                        "$lt": end_of_year
                    }
                }))

                if not orders and not warehouses:
                    response = [
                        f"Không tìm thấy doanh thu nào trong năm {target_year}. Bạn còn câu hỏi nào khác không?",
                        f"Doanh thu chưa xuất hiện trong năm {target_year}. Nếu có câu hỏi khác, tôi sẵn sàng hỗ trợ bạn.",
                        f"Doanh thu năm {target_year} là 0 VNĐ. Bạn có thể truy vấn doanh thu của các năm khác nếu muốn."
                    ]
                    return JSONResponse(content={"response": random.choice(response)})

                # Tính tổng doanh thu năm
                total_order_revenue = sum(order.get("total", 0) for order in orders)
                total_warehouse_revenue = sum(wh.get("total", 0) for wh in warehouses)
                total_revenue = total_order_revenue + total_warehouse_revenue
                
                # Hiển thị trạng thái
                status_mapping = {
                    "Processing": "Đang xử lý",
                    "Delivering": "Đang giao hàng",
                    "Succeed": "Thành công",
                    "Cancelled": "Đã hủy",
                    "Confirm": "Đã xác nhận"
                }
                
                def create_detailed_response():
                    html_content = f"""
                    <div style="background: #f8f9fa; border-radius: 8px; padding: 20px;">
                        <h2 style="color: #2c3e50; margin-bottom: 20px;">📊 Báo cáo doanh thu năm {target_year}</h2>
                        
                        <div style="background: white; padding: 15px; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #27ae60; margin-top: 0;">💰 Tổng quan doanh thu</h3>
                            <p><strong>Tổng doanh thu: </strong><span style="color: #e74c3c; font-size: 18px; font-weight: bold;">{total_revenue:,} VNĐ</span></p>
                            <p>• Doanh thu từ đơn hàng: {total_order_revenue:,} VNĐ ({len(orders)} đơn)</p>
                            <p>• Doanh thu từ phiếu xuất: {total_warehouse_revenue:,} VNĐ ({len(warehouses)} phiếu)</p>
                            <p>• Trung bình mỗi tháng: {(total_revenue / 12):,.0f} VNĐ</p>
                        </div>
                    """
                    
                    # Chi tiết đơn hàng
                    if orders:
                        html_content += """
                        <div style="background: white; padding: 15px; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #3498db; margin-top: 0;">🛒 Chi tiết đơn hàng</h3>
                        """
                        for idx, order in enumerate(orders, 1):
                            order_date = order.get('createdAt')
                            if isinstance(order_date, datetime):
                                order_time = order_date.strftime('%d/%m/%Y %H:%M')
                            else:
                                order_time = 'N/A'
                            
                            order_status = order.get('status', 'N/A')
                            status_color = {
                                'Succeed': '#27ae60',
                                'Processing': '#f39c12', 
                                'Delivering': '#3498db',
                                'Cancelled': '#e74c3c',
                                'Confirm': '#9b59b6'
                            }.get(order_status, '#95a5a6')
                            
                            html_content += f"""
                            <div style="border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: #2c3e50;">Đơn hàng #{idx}</h4>
                                    <span style="background: {status_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                        {status_mapping.get(order_status, order_status)}
                                    </span>
                                </div>
                                <p style="margin: 5px 0; color: #7f8c8d;">⏰ Thời gian: {order_time}</p>
                                <p style="margin: 5px 0;"><strong>Tổng tiền: {order.get('total', 0):,} VNĐ</strong></p>
                            """
                            
                            # Thông tin khách hàng
                            order_by_id = order.get('orderBy')
                            if order_by_id:
                                try:
                                    if isinstance(order_by_id, str):
                                        order_by_id = ObjectId(order_by_id)
                                    customer = user_collection.find_one({"_id": order_by_id})
                                    if customer:
                                        customer_name = customer.get('name', 'N/A')
                                        customer_phone = customer.get('phone', '')
                                        html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>👤 Khách hàng: {customer_name}"
                                        if customer_phone:
                                            html_content += f" - {customer_phone}"
                                        html_content += "</p>"
                                except:
                                    pass
                            
                            # Voucher (nếu có)
                            voucher_id = order.get('applyVoucher')
                            if voucher_id:
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>🎫 Đã áp dụng voucher</p>"
                            
                            # Vị trí giao hàng
                            location = order.get('location')
                            if location and location.get('lat') and location.get('lng'):
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d; font-size: 13px;'>📍 Vị trí: ({location['lat']}, {location['lng']})</p>"
                            
                            # Thông tin sản phẩm trong đơn hàng
                            products = order.get('products', [])
                            if products:
                                html_content += "<h5 style='margin: 10px 0 5px 0; color: #34495e;'>📦 Sản phẩm:</h5>"
                                html_content += "<div style='background: #f8f9fa; padding: 8px; border-radius: 4px;'>"
                                
                                for product in products:
                                    product_name = product.get('name', 'Sản phẩm không rõ tên')
                                    quantity = product.get('quantity', 0)
                                    price = product.get('price', 0)
                                    variant = product.get('variant', '')
                                    subtotal = quantity * price
                                    
                                    html_content += f"""
                                    <div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px;">
                                        <div style="font-weight: bold; color: #2c3e50; margin-bottom: 3px;">
                                            {product_name}
                                            {f" ({variant})" if variant else ""}
                                        </div>
                                        <div style="font-size: 14px; color: #7f8c8d;">
                                            Số lượng: {quantity} × {price:,} VNĐ = <strong style="color: #e74c3c;">{subtotal:,} VNĐ</strong>
                                        </div>
                                    </div>
                                    """
                                
                                html_content += "</div>"
                            
                            html_content += "</div>"
                        
                        html_content += "</div>"
                    
                    # Chi tiết phiếu xuất
                    if warehouses:
                        html_content += """
                        <div style="background: white; padding: 15px; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #9b59b6; margin-top: 0;">📋 Chi tiết phiếu xuất</h3>
                        """
                        for idx, warehouse in enumerate(warehouses, 1):
                            wh_date = warehouse.get('createdAt')
                            if isinstance(wh_date, datetime):
                                wh_time = wh_date.strftime('%d/%m/%Y %H:%M')
                            else:
                                wh_time = 'N/A'
                            
                            html_content += f"""
                            <div style="border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: #2c3e50;">Phiếu xuất #{idx}</h4>
                                    <span style="background: #9b59b6; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                        Xuất kho
                                    </span>
                                </div>
                                <p style="margin: 5px 0; color: #7f8c8d;">⏰ Thời gian: {wh_time}</p>
                            """
                            
                            # Người xử lý
                            handled_by = warehouse.get('handledBy', 'N/A')
                            html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>👤 Người xử lý: {handled_by}</p>"
                            
                            # Nhà cung cấp
                            supplier = warehouse.get('supplier', '')
                            if supplier:
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>🏢 Nhà cung cấp: {supplier}</p>"
                            
                            # Thông tin xuất đến
                            exported_to = warehouse.get('exportedTo', {})
                            if exported_to and exported_to.get('name'):
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>📍 Xuất đến: {exported_to.get('name', '')}"
                                if exported_to.get('phone'):
                                    html_content += f" - {exported_to.get('phone')}"
                                html_content += "</p>"
                                if exported_to.get('address'):
                                    html_content += f"<p style='margin: 5px 0; color: #7f8c8d; font-size: 13px;'>   Địa chỉ: {exported_to.get('address')}</p>"
                                if exported_to.get('email'):
                                    html_content += f"<p style='margin: 5px 0; color: #7f8c8d; font-size: 13px;'>   Email: {exported_to.get('email')}</p>"
                            
                            html_content += f"<p style='margin: 5px 0;'><strong>Tổng tiền: {warehouse.get('total', 0):,} VNĐ</strong></p>"
                            
                            # Thông tin sản phẩm trong phiếu xuất
                            products = warehouse.get('products', [])
                            if products:
                                html_content += "<h5 style='margin: 10px 0 5px 0; color: #34495e;'>📦 Sản phẩm:</h5>"
                                html_content += "<div style='background: #f8f9fa; padding: 8px; border-radius: 4px;'>"
                                
                                for product in products:
                                    product_name = product.get('name', 'Sản phẩm không rõ tên')
                                    quantity = product.get('quantity', 0)
                                    price = product.get('price', 0)
                                    variant = product.get('variant', '')
                                    total_price = product.get('totalPrice', quantity * price)
                                    
                                    html_content += f"""
                                    <div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px;">
                                        <div style="font-weight: bold; color: #2c3e50; margin-bottom: 3px;">
                                            {product_name}
                                            {f" ({variant})" if variant else ""}
                                        </div>
                                        <div style="font-size: 14px; color: #7f8c8d;">
                                            Số lượng: {quantity} × {price:,} VNĐ = <strong style="color: #e74c3c;">{total_price:,} VNĐ</strong>
                                        </div>
                                    </div>
                                    """
                                
                                html_content += "</div>"
                            
                            html_content += "</div>"
                        
                        html_content += "</div>"
                    
                    html_content += """
                        <div style="text-align: center; margin-top: 20px; padding: 15px; background: #ecf0f1; border-radius: 6px;">
                            <p style="margin: 0; color: #7f8c8d;">Hãy nhập thêm câu hỏi nếu có, tôi luôn sẵn sàng giải đáp các thắc mắc của bạn</p>
                        </div>
                    </div>
                    """
                    
                    return html_content
                
                # Kiểm tra xem message có chứa từ "chi tiết" không
                show_detail = any(keyword in message.lower() for keyword in ["chi tiết", "thông tin cụ thể", "doanh thu cụ thể"])
                
                if show_detail:
                    # Hiển thị chi tiết đầy đủ
                    response = create_detailed_response()
                else:
                    # Tính trung bình tháng
                    avg_monthly_revenue = total_revenue / 12 if total_revenue > 0 else 0

                    responses = [
                        f"Doanh thu năm {target_year} là: {total_revenue:,} VNĐ. Trong đó doanh thu từ đơn hàng là {total_order_revenue:,} VNĐ, doanh thu từ phiếu xuất là {total_warehouse_revenue:,} VNĐ. Trung bình mỗi tháng là {avg_monthly_revenue:,.0f} VNĐ. Ngoài ra, bạn còn câu hỏi nào khác không?",
                        f"Tổng doanh thu năm {target_year}: {total_revenue:,} VNĐ. Doanh thu từ đơn hàng là {total_order_revenue:,} VNĐ. Doanh thu từ phiếu xuất là {total_warehouse_revenue:,} VNĐ. Trung bình mỗi tháng: {avg_monthly_revenue:,.0f} VNĐ. Bạn cần tôi giải đáp gì nữa không?",
                        (
                            f"<p>Doanh thu năm {target_year}:</p>"
                            f"<ul>"
                            f"<li>Doanh thu đơn hàng: {total_order_revenue:,} VNĐ ({len(orders)} đơn)</li>"
                            f"<li>Doanh thu phiếu xuất sản phẩm: {total_warehouse_revenue:,} VNĐ ({len(warehouses)} phiếu)</li>"
                            f"<li><strong>Tổng cộng: {total_revenue:,} VNĐ</strong></li>"
                            f"<li>Trung bình mỗi tháng: {avg_monthly_revenue:,.0f} VNĐ</li>"
                            f"</ul>"
                            f"<p>Bạn có muốn xem chi tiết theo tháng hoặc so sánh với năm khác không?</p>"
                        )
                    ]
                    response = random.choice(responses)
                
                return JSONResponse(content={"response": response})

            except Exception as e:
                return JSONResponse(content={"response": f"Đã xảy ra lỗi khi truy vấn doanh thu theo năm: {str(e)}"})
        # Doanh thu khoản thời gian theo ngày
        if tag == 'doanh_thu_khoang_thoi_gian_theo_ngay':
            try:
                date_range_patterns = [
                    r"doanh thu từ ngày (.+?) đến ngày (.+?)(?:\s|$|\.|\?|!)",
                    r"doanh thu từ (.+?) đến (.+?)(?:\s|$|\.|\?|!)",
                    r"từ ngày (.+?) đến ngày (.+?)(?:\s|$|\.|\?|!)",
                    r"từ (.+?) đến (.+?)(?:\s|$|\.|\?|!)",
                    r"từ ngày (.+?) đến (.+?)(?:\s|$|\.|\?|!)",
                    r"từ (.+?) đến ngày (.+?)(?:\s|$|\.|\?|!)",
                ]
                start_date, end_date = extract_date_range(date_range_patterns, message)
                target_start_date = parse_date_from_query(start_date)
                target_end_date = parse_date_from_query(end_date)
                
                if not target_start_date or not target_end_date:
                    error_responses = [
                        "Không thể hiểu định dạng ngày. Vui lòng nhập theo format: DD/MM/YYYY, DD/MM hoặc 'hôm nay'",
                        "Format ngày không đúng. Bạn có thể nhập như: 15/7, 15/7/2025, hôm nay, hôm qua",
                        "Tôi không hiểu ngày bạn muốn xem. Vui lòng thử lại với format: DD/MM hoặc DD/MM/YYYY"
                    ]
                    return JSONResponse(content={"response": random.choice(error_responses)})
                
                # Tạo khoảng thời gian
                start_of_day = datetime.combine(target_start_date, datetime.min.time())
                end_of_day = datetime.combine(target_end_date, datetime.max.time())
                
                # Query đơn hàng trong khoảng thời gian (chỉ tính đơn thành công)
                orders = list(order_collection.find({
                    "status": {"$in": ["Succeed", "Delivering"]},
                    "createdAt": { 
                        "$gte": start_of_day,
                        "$lte": end_of_day
                    }
                }))
                
                # Query phiếu xuất trong khoảng thời gian (warehouse với type = 'export')
                warehouses = list(warehouses_collection.find({
                    "type": "export",
                    "createdAt": { 
                        "$gte": start_of_day,
                        "$lte": end_of_day
                    }
                }))
                
                # Format ngày để hiển thị
                formatted_start_date = target_start_date.strftime("%d/%m/%Y")
                formatted_end_date = target_end_date.strftime("%d/%m/%Y")
                
                if not orders and not warehouses:
                    response = [
                        f"Không tìm thấy bất kỳ khoản doanh thu nào từ ngày {formatted_start_date} đến ngày {formatted_end_date}. Bạn còn câu hỏi nào khác không?",
                        f"Từ ngày {formatted_start_date} đến ngày {formatted_end_date} chưa xuất hiện bất kỳ khoản doanh thu nào. Nếu có câu hỏi khác, tôi sẵn sàng hỗ trợ bạn.",
                        f"Doanh thu từ ngày {formatted_start_date} đến ngày {formatted_end_date} là 0 VNĐ. Bạn có thể truy vấn doanh thu theo khoảng thời gian khác nếu muốn."
                    ]
                    return JSONResponse(content={"response": random.choice(response)})

                # Tính tổng doanh thu
                total_order_revenue = sum(order.get("total", 0) for order in orders)
                total_warehouse_revenue = sum(wh.get("total", 0) for wh in warehouses)
                total_revenue = total_order_revenue + total_warehouse_revenue
                
                # Tính số ngày trong khoảng
                days_count = (target_end_date - target_start_date).days + 1
                avg_daily_revenue = total_revenue / days_count if days_count > 0 else 0
                
                # Hiển thị trạng thái
                status_mapping = {
                    "Processing": "Đang xử lý",
                    "Delivering": "Đang giao hàng",
                    "Succeed": "Thành công",
                    "Cancelled": "Đã hủy",
                    "Confirm": "Đã xác nhận"
                }
                
                def create_detailed_response():
                    html_content = f"""
                    <div style="background: #f8f9fa; border-radius: 8px; padding: 20px;">
                        <h2 style="color: #2c3e50; margin-bottom: 20px;">📊 Báo cáo doanh thu từ ngày {formatted_start_date} đến ngày {formatted_end_date}</h2>
                        
                        <div style="background: white; padding: 15px; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #27ae60; margin-top: 0;">💰 Tổng quan doanh thu</h3>
                            <p><strong>Tổng doanh thu: </strong><span style="color: #e74c3c; font-size: 18px; font-weight: bold;">{total_revenue:,} VNĐ</span></p>
                            <p>• Doanh thu từ đơn hàng: {total_order_revenue:,} VNĐ ({len(orders)} đơn)</p>
                            <p>• Doanh thu từ phiếu xuất: {total_warehouse_revenue:,} VNĐ ({len(warehouses)} phiếu)</p>
                            <p>• Khoảng thời gian: {days_count} ngày</p>
                            <p>• Trung bình mỗi ngày: {avg_daily_revenue:,.0f} VNĐ</p>
                        </div>
                    """
                    
                    # Chi tiết đơn hàng
                    if orders:
                        html_content += """
                        <div style="background: white; padding: 15px; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #3498db; margin-top: 0;">🛒 Chi tiết đơn hàng</h3>
                        """
                        for idx, order in enumerate(orders, 1):
                            order_date = order.get('createdAt')
                            if isinstance(order_date, datetime):
                                order_time = order_date.strftime('%d/%m/%Y %H:%M')
                            else:
                                order_time = 'N/A'
                            
                            order_status = order.get('status', 'N/A')
                            status_color = {
                                'Succeed': '#27ae60',
                                'Processing': '#f39c12', 
                                'Delivering': '#3498db',
                                'Cancelled': '#e74c3c',
                                'Confirm': '#9b59b6'
                            }.get(order_status, '#95a5a6')
                            
                            html_content += f"""
                            <div style="border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: #2c3e50;">Đơn hàng #{idx}</h4>
                                    <span style="background: {status_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                        {status_mapping.get(order_status, order_status)}
                                    </span>
                                </div>
                                <p style="margin: 5px 0; color: #7f8c8d;">⏰ Thời gian: {order_time}</p>
                                <p style="margin: 5px 0;"><strong>Tổng tiền: {order.get('total', 0):,} VNĐ</strong></p>
                            """
                            
                            # Thông tin khách hàng
                            order_by_id = order.get('orderBy')
                            if order_by_id:
                                try:
                                    if isinstance(order_by_id, str):
                                        order_by_id = ObjectId(order_by_id)
                                    customer = user_collection.find_one({"_id": order_by_id})
                                    if customer:
                                        customer_name = customer.get('name', 'N/A')
                                        customer_phone = customer.get('phone', '')
                                        html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>👤 Khách hàng: {customer_name}"
                                        if customer_phone:
                                            html_content += f" - {customer_phone}"
                                        html_content += "</p>"
                                except:
                                    pass
                            
                            # Voucher (nếu có)
                            voucher_id = order.get('applyVoucher')
                            if voucher_id:
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>🎫 Đã áp dụng voucher</p>"
                            
                            # Vị trí giao hàng
                            location = order.get('location')
                            if location and location.get('lat') and location.get('lng'):
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d; font-size: 13px;'>📍 Vị trí: ({location['lat']}, {location['lng']})</p>"
                            
                            # Thông tin sản phẩm trong đơn hàng
                            products = order.get('products', [])
                            if products:
                                html_content += "<h5 style='margin: 10px 0 5px 0; color: #34495e;'>📦 Sản phẩm:</h5>"
                                html_content += "<div style='background: #f8f9fa; padding: 8px; border-radius: 4px;'>"
                                
                                for product in products:
                                    product_name = product.get('name', 'Sản phẩm không rõ tên')
                                    quantity = product.get('quantity', 0)
                                    price = product.get('price', 0)
                                    variant = product.get('variant', '')
                                    subtotal = quantity * price
                                    
                                    html_content += f"""
                                    <div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px;">
                                        <div style="font-weight: bold; color: #2c3e50; margin-bottom: 3px;">
                                            {product_name}
                                            {f" ({variant})" if variant else ""}
                                        </div>
                                        <div style="font-size: 14px; color: #7f8c8d;">
                                            Số lượng: {quantity} × {price:,} VNĐ = <strong style="color: #e74c3c;">{subtotal:,} VNĐ</strong>
                                        </div>
                                    </div>
                                    """
                                
                                html_content += "</div>"
                            
                            html_content += "</div>"
                        
                        html_content += "</div>"
                    
                    # Chi tiết phiếu xuất
                    if warehouses:
                        html_content += """
                        <div style="background: white; padding: 15px; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #9b59b6; margin-top: 0;">📋 Chi tiết phiếu xuất</h3>
                        """
                        for idx, warehouse in enumerate(warehouses, 1):
                            wh_date = warehouse.get('createdAt')
                            if isinstance(wh_date, datetime):
                                wh_time = wh_date.strftime('%d/%m/%Y %H:%M')
                            else:
                                wh_time = 'N/A'
                            
                            html_content += f"""
                            <div style="border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: #2c3e50;">Phiếu xuất #{idx}</h4>
                                    <span style="background: #9b59b6; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                        Xuất kho
                                    </span>
                                </div>
                                <p style="margin: 5px 0; color: #7f8c8d;">⏰ Thời gian: {wh_time}</p>
                            """
                            
                            # Người xử lý
                            handled_by = warehouse.get('handledBy', 'N/A')
                            html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>👤 Người xử lý: {handled_by}</p>"
                            
                            # Nhà cung cấp
                            supplier = warehouse.get('supplier', '')
                            if supplier:
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>🏢 Nhà cung cấp: {supplier}</p>"
                            
                            # Thông tin xuất đến
                            exported_to = warehouse.get('exportedTo', {})
                            if exported_to and exported_to.get('name'):
                                html_content += f"<p style='margin: 5px 0; color: #7f8c8d;'>📍 Xuất đến: {exported_to.get('name', '')}"
                                if exported_to.get('phone'):
                                    html_content += f" - {exported_to.get('phone')}"
                                html_content += "</p>"
                                if exported_to.get('address'):
                                    html_content += f"<p style='margin: 5px 0; color: #7f8c8d; font-size: 13px;'>   Địa chỉ: {exported_to.get('address')}</p>"
                                if exported_to.get('email'):
                                    html_content += f"<p style='margin: 5px 0; color: #7f8c8d; font-size: 13px;'>   Email: {exported_to.get('email')}</p>"
                            
                            html_content += f"<p style='margin: 5px 0;'><strong>Tổng tiền: {warehouse.get('total', 0):,} VNĐ</strong></p>"
                            
                            # Thông tin sản phẩm trong phiếu xuất
                            products = warehouse.get('products', [])
                            if products:
                                html_content += "<h5 style='margin: 10px 0 5px 0; color: #34495e;'>📦 Sản phẩm:</h5>"
                                html_content += "<div style='background: #f8f9fa; padding: 8px; border-radius: 4px;'>"
                                
                                for product in products:
                                    product_name = product.get('name', 'Sản phẩm không rõ tên')
                                    quantity = product.get('quantity', 0)
                                    price = product.get('price', 0)
                                    variant = product.get('variant', '')
                                    total_price = product.get('totalPrice', quantity * price)
                                    
                                    html_content += f"""
                                    <div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px;">
                                        <div style="font-weight: bold; color: #2c3e50; margin-bottom: 3px;">
                                            {product_name}
                                            {f" ({variant})" if variant else ""}
                                        </div>
                                        <div style="font-size: 14px; color: #7f8c8d;">
                                            Số lượng: {quantity} × {price:,} VNĐ = <strong style="color: #e74c3c;">{total_price:,} VNĐ</strong>
                                        </div>
                                    </div>
                                    """
                                
                                html_content += "</div>"
                            
                            html_content += "</div>"
                        
                        html_content += "</div>"
                    
                    html_content += """
                        <div style="text-align: center; margin-top: 20px; padding: 15px; background: #ecf0f1; border-radius: 6px;">
                            <p style="margin: 0; color: #7f8c8d;">Hãy nhập thêm câu hỏi nếu có, tôi luôn sẵn sàng giải đáp các thắc mắc của bạn</p>
                        </div>
                    </div>
                    """
                    
                    return html_content
                
                # Kiểm tra xem message có chứa từ "chi tiết" không
                show_detail = any(keyword in message.lower() for keyword in ["chi tiết", "thông tin cụ thể", "doanh thu cụ thể"])
                
                if show_detail:
                    # Hiển thị chi tiết đầy đủ
                    response = create_detailed_response()
                else:
                    responses = [
                        f"Doanh thu từ ngày {formatted_start_date} đến ngày {formatted_end_date} là: {total_revenue:,} VNĐ. Trong đó doanh thu từ đơn hàng là {total_order_revenue:,} VNĐ, doanh thu từ phiếu xuất là {total_warehouse_revenue:,} VNĐ. Trung bình mỗi ngày: {avg_daily_revenue:,.0f} VNĐ. Ngoài ra, bạn còn câu hỏi nào khác không?",
                        f"Tổng doanh thu từ ngày {formatted_start_date} đến ngày {formatted_end_date}: {total_revenue:,} VNĐ ({days_count} ngày). Doanh thu từ đơn hàng là {total_order_revenue:,} VNĐ. Doanh thu từ phiếu xuất là {total_warehouse_revenue:,} VNĐ. Bạn cần tôi giải đáp gì nữa không?",
                        (
                            f"<p>Doanh thu từ ngày {formatted_start_date} đến ngày {formatted_end_date}:</p>"
                            f"<ul>"
                            f"<li>Doanh thu đơn hàng: {total_order_revenue:,} VNĐ ({len(orders)} đơn)</li>"
                            f"<li>Doanh thu phiếu xuất sản phẩm: {total_warehouse_revenue:,} VNĐ ({len(warehouses)} phiếu)</li>"
                            f"<li><strong>Tổng cộng: {total_revenue:,} VNĐ</strong></li>"
                            f"<li>Khoảng thời gian: {days_count} ngày</li>"
                            f"<li>Trung bình mỗi ngày: {avg_daily_revenue:,.0f} VNĐ</li>"
                            f"</ul>"
                            f"<p>Bạn có muốn xem chi tiết từng ngày không?</p>"
                        )
                    ]
                    response = random.choice(responses)
                
                return JSONResponse(content={"response": response})

            except Exception as e:
                return JSONResponse(content={"response": f"Đã xảy ra lỗi khi truy vấn doanh thu theo khoảng thời gian: {str(e)}"})
        # Thống kế doanh thu từ tháng đến tháng
        if tag == 'doanh_thu_khoang_thoi_gian_theo_thang':
            try:
                month_range_patterns = [
                    r"doanh thu từ tháng (.+?) đến tháng (.+?)(?:\s|$|\.|\?|!)",
                    r"doanh thu từ (.+?) đến (.+?)(?:\s|$|\.|\?|!)",
                    r"từ tháng (.+?) đến tháng (.+?)(?:\s|$|\.|\?|!)",
                    r"từ (.+?) đến (.+?)(?:\s|$|\.|\?|!)",
                    r"từ tháng (.+?) đến (.+?)(?:\s|$|\.|\?|!)",
                    r"từ (.+?) đến tháng (.+?)(?:\s|$|\.|\?|!)",
                ]
                start_month, end_month = extract_date_range(month_range_patterns, message)
                target_start_month = parse_month_from_query(start_month)
                target_end_month = parse_month_from_query(end_month)
                
                if not target_start_month and target_end_month:
                    if not parsed_month:
                        error_responses = [
                            "Không thể hiểu định dạng tháng. Vui lòng nhập theo format: MM/YYYY, MM hoặc 'tháng này'",
                            "Format tháng không đúng. Bạn có thể nhập như: 7/2025, 7, tháng này, tháng trước",
                        ]
                        return JSONResponse(content={"response": random.choice(error_responses)})
                
                
                # tháng
                year_start, month_start = target_start_month
                year_end, month_end = target_end_month
                start_of_month = datetime(year_start, month_start, 1)
                print(start_of_month)
                end_of_month = datetime(year_end, month_end, 1)
                print(end_of_month)
                
                # Query đơn hàng trong năm cụ thể
                orders = list(order_collection.find({
                    "status": {"$in": ['Delivering', 'Succeed', 'Confirm']},
                    "createdAt": { 
                        "$gte": start_of_month,
                        "$lt": end_of_month
                    }
                }))
                
                # Query phiếu xuất trong năm cụ thể
                receipts = list(warehouses_collection.find({
                    "type": "export",
                    "createdAt": { 
                        "$gte": start_of_month,
                        "$lt": end_of_month
                    }
                }))
                if not orders and not receipts:
                    response = [
                        f"Không tìm thấy bất kỳ khoản doanh thu nào từ tháng {month_start}/{year_start} đến tháng {month_end}/{year_end}. Bạn còn câu hỏi nào khác không?",
                        f"Chưa có doanh thu nào xuất hiện kể từ tháng {month_start}/{year_start} đến tháng {month_end}/{year_end}. Nếu có câu hỏi khác, tôi sẵn sàng hỗ trợ bạn.",
                        f"Doanh thu từ tháng {month_start}/{year_start} đến tháng {month_end}/{year_end} là 0 VNĐ. Ngoài ra, bạn còn câu hỏi nào khác không?"
                    ]
                    return JSONResponse(content={"response": random.choice(response)})

                # Tính tổng doanh thu năm
                total_order_revenue = sum(order.get("total", 0) for order in orders)
                total_receipt_revenue = sum(receipt.get("total", 0) for receipt in receipts)
                total_revenue = total_order_revenue + total_receipt_revenue
                # Hiển thị trạng thái
                status_mapping = {
                        "Processing": "Đang xử lý",
                        "Delivering": "Đang giao hàng",
                        "Received": "Đã nhận hàng",
                        "Succeed": "Đã nhận hàng",
                        "Cancelled": "Đã hủy"
                    }
                def create_detailed_response():
                    html_content = f"""
                    <div style="background: #f8f9fa; border-radius: 8px;">
                        <h2 style="color: #2c3e50; margin-bottom: 20px;">📊 Báo cáo doanh thu từ tháng {month_start}/{year_start} đến tháng {month_end}/{year_end}</h2>
                        
                            <h3 style="color: #27ae60; margin-top: 0;">💰 Tổng quan doanh thu</h3>
                            <p><strong>Tổng doanh thu: </strong><span style="color: #e74c3c; font-size: 18px; font-weight: bold;">{total_revenue:,} VNĐ</span></p>
                            <p>• Doanh thu từ đơn hàng: {total_order_revenue:,} VNĐ ({len(orders)} đơn)</p>
                            <p>• Doanh thu từ phiếu xuất: {total_receipt_revenue:,} VNĐ ({len(receipts)} phiếu)</p>
                        </div>
                    """
                    # Chi tiết đơn hàng
                    if orders:
                        html_content += """
                        <div style="background: white; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #3498db; margin-top: 0;">🛒 Chi tiết đơn hàng</h3>
                        """
                        for idx, order in enumerate(orders, 1):
                            order_time = order.get('createdAt', '').strftime('%H:%M') if order.get('createdAt') else 'N/A'
                            status_color = {
                                'Confirm': '#27ae60',
                                'Succeed': '#27ae60',
                                'Processing': '#f39c12', 
                                'Delivering': '#3498db',
                                'Cancelled': "#d81a13"
                            }.get(order.get('status', ''), '#95a5a6')
                            
                            html_content += f"""
                            <div style="border: 1px solid #ddd; margin-bottom: 15px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: #2c3e50;">#{order.get('_id', 'N/A')}</h4>
                                    <span style="background: {status_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                        {status_mapping.get(order.get('status', 'N/A') ,order.get('status', 'N/A'))}
                                    </span>
                                </div>
                                <p style="margin: 5px 0; color: #7f8c8d;">⏰ Thời gian: {order_time}</p>
                                <p style="margin: 5px 0;"><strong>Tổng tiền: {order.get('total', 0):,} VNĐ</strong></p>
                            """
                            
                            # Thông tin sản phẩm trong đơn hàng
                            products = order.get('products', [])
                            if products:
                                html_content += "<h5 style='margin: 10px 0 5px 0; color: #34495e;'>📦 Sản phẩm:</h5>"
                                html_content += "<div style='background: #f8f9fa; padding: 8px; border-radius: 4px;'>"
                                
                                for product in products:
                                    product_name = product.get('name', 'Sản phẩm không rõ tên')
                                    quantity = product.get('quantity', 0)
                                    price = product.get('price', 0)
                                    subtotal = quantity * price
                                    
                                    html_content += f"""
                                    <div style="margin-bottom: 8px;  background: white; border-radius: 3px;">
                                        <div style="font-weight: bold; color: #2c3e50; margin-bottom: 3px;">{product_name}</div>
                                        <div style="font-size: 14px; color: #7f8c8d;">
                                            Số lượng: {quantity} × {price:,} VNĐ = <strong style="color: #e74c3c;">{subtotal:,} VNĐ</strong>
                                        </div>
                                    </div>
                                    """
                                
                                html_content += "</div>"
                            
                            html_content += "</div>"
                        
                        html_content += "</div>"
                    
                    # Chi tiết phiếu xuất
                    if receipts:
                        html_content += """
                        <div style="background: white;  margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #9b59b6; margin-top: 0;">📋 Chi tiết phiếu xuất</h3>
                        """
                        for idx, receipt in enumerate(receipts, 1):
                            receipt_time = receipt.get('createdAt', '').strftime('%H:%M') if receipt.get('createdAt') else 'N/A'
                            
                            html_content += f"""
                            <div style="border: 1px solid #ddd; margin-bottom: 15px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: #2c3e50;">#{receipt.get('_id', 'N/A')}</h4>
                                </div>
                                <p style="margin: 5px 0; color: #7f8c8d;">⏰ Thời gian: {receipt_time}</p>
                                <p style="margin: 5px 0;"><strong>Tổng tiền: {receipt.get('total', 0):,} VNĐ</strong></p>
                            """
                            
                            # Thông tin sản phẩm trong đơn hàng
                            products = receipt.get('products', [])
                            if products:
                                html_content += "<h5 style='margin: 10px 0 5px 0; color: #34495e;'>📦 Sản phẩm:</h5>"
                                html_content += "<div style='background: #f8f9fa; padding: 8px; border-radius: 4px;'>"
                                
                                for product in products:
                                    product_name = product.get('name', 'Sản phẩm không rõ tên')
                                    quantity = product.get('quantity', 0)
                                    price = product.get('price', 0)
                                    subtotal = quantity * price
                                    
                                    html_content += f"""
                                    <div style="margin-bottom: 8px;  background: white; border-radius: 3px;">
                                        <div style="font-weight: bold; color: #2c3e50; margin-bottom: 3px;">{product_name}</div>
                                        <div style="font-size: 14px; color: #7f8c8d;">
                                            Số lượng: {quantity} × {price:,} VNĐ = <strong style="color: #e74c3c;">{subtotal:,} VNĐ</strong>
                                        </div>
                                    </div>
                                    """
                                
                                html_content += "</div>"
                            
                            html_content += "</div>"
                        
                        html_content += "</div>"
                    
                    html_content += """
                        <div style="text-align: center; margin-top: 5px; background: #ecf0f1; border-radius: 6px;">
                            <p style="margin: 0; color: #7f8c8d;">Hãy nhập thêm câu hỏi nếu có, tôi luôn sẵn sàng giải đáp các thắc mắc của bạn</p>
                        </div>
                    </div>
                    """
                    
                    return html_content
                
                # Kiểm tra xem message có chứa từ "chi tiết" không
                show_detail = any(keyword in message.lower() for keyword in ["chi tiết", "thông tin cụ thể", "doanh thu cụ thể"])
                if show_detail:
                    # Hiển thị chi tiết đầy đủ
                    response = create_detailed_response()
                else:
                    responses = [
                        f"Doanh từ tháng {month_start}/{year_start} đến tháng {month_end}/{year_end} là: {total_revenue:,} VNĐ. Trong đó doanh thu từ đơn hàng là {total_order_revenue:,} VNĐ, doanh thu từ phiếu xuất là {total_receipt_revenue:,} VNĐ. Ngoài ra, bạn còn câu hỏi nào khác không?",
                        f"Tổng doanh thu từ tháng {month_start}/{year_start} đến tháng {month_end}/{year_end}: {total_revenue:,} VNĐ. Doanh thu từ đơn hàng là {total_order_revenue:,} VNĐ. Doanh thu từ phiếu xuất là {total_receipt_revenue:,} VNĐ. Bạn cần tôi giải đáp gì nữa không?",
                        (
                            f"<p>Doanh thu từ tháng {month_start}/{year_start} đến tháng {month_end}/{year_end}:</p>"
                            f"<ul>"
                            f"<li>Doanh thu đơn hàng: {total_order_revenue:,} VNĐ ({len(orders)} đơn)</li>"
                            f"<li>Doanh thu phiếu xuất sản phẩm: {total_receipt_revenue:,} VNĐ ({len(receipts)} phiếu)</li>"
                            f"<li><strong>Tổng cộng: {total_revenue:,} VNĐ</strong></li>"
                            f"</ul>"
                            f"<p>Bạn có muốn xem chi tiết theo ngày hoặc so sánh với năm khác không?</p>"
                        )
                    ]
                    response = random.choice(responses)
                return JSONResponse(content={"response": response})

            except Exception as e:
                return JSONResponse(content={"response": f"Đã xảy ra lỗi khi truy vấn doanh thu theo tháng: {str(e)}"})
        # Thống kế doanh thu từ năm đến năm
        if tag == 'doanh_thu_khoang_thoi_gian_theo_nam':
            try:
                year_range_patterns = [
                    r"doanh thu từ năm (.+?) đến năm (.+?)(?:\s|$|\.|\?|!)",
                    r"doanh thu từ (.+?) đến (.+?)(?:\s|$|\.|\?|!)",
                    r"từ năm (.+?) đến năm (.+?)(?:\s|$|\.|\?|!)",
                    r"từ (.+?) đến (.+?)(?:\s|$|\.|\?|!)",
                    r"từ năm (.+?) đến (.+?)(?:\s|$|\.|\?|!)",
                    r"từ (.+?) đến năm (.+?)(?:\s|$|\.|\?|!)",
                ]
                year_query_start, year_query_end = extract_date_range(year_range_patterns, message)
                target_year_start = parse_year_from_query(year_query_start) 
                target_year_end = parse_year_from_query(year_query_end) 
                
                if not target_year_start or not target_year_end:
                    error_responses = [
                        "Không thể hiểu định dạng năm. Vui lòng nhập theo format: YYYY hoặc 'năm này'",
                        "Format năm không đúng. Bạn có thể nhập như: 2024, 2025, năm này, năm trước",
                        "Tôi không hiểu năm bạn muốn xem. Vui lòng thử lại với format: YYYY"
                    ]
                    return JSONResponse(content={"response": random.choice(error_responses)})
                start_of_year = datetime(target_year_start, 1, 1)
                end_of_year = datetime(target_year_end, 1, 1)
                
                # Query đơn hàng trong năm cụ thể
                orders = list(order_collection.find({
                    "status": {"$in": ['Delivering', 'Succeed', 'Confirm']},
                    "createdAt": { 
                        "$gte": start_of_year,
                        "$lt": end_of_year
                    }
                }))
                
                # Query phiếu xuất trong năm cụ thể
                receipts = list(warehouses_collection.find({
                    "type": "export",
                    "createdAt": { 
                        "$gte": start_of_year,
                        "$lt": end_of_year
                    }
                }))
                if not orders or not receipts:
                    response = [
                        f"Không tìm thấy đơn hàng nào từ năm {year_query_start} đến năm {year_query_end}. Bạn còn câu hỏi nào khác không?",
                        f"Chưa có đơn hàng nào được tạo từ năm {year_query_start} đến năm {year_query_end}. Nếu có câu hỏi khác, tôi sẵn sàng hỗ trợ bạn.",
                        f"Doanh thu từ năm {year_query_start} đến năm {year_query_end} là 0 VNĐ."
                    ]
                    return JSONResponse(content={"response": random.choice(response)})

                # Tính tổng doanh thu năm
                total_order_revenue = sum(order.get("total", 0) for order in orders)
                total_receipt_revenue = sum(receipt.get("total", 0) for receipt in receipts)
                total_revenue = total_order_revenue + total_receipt_revenue
                # Hiển thị trạng thái
                status_mapping = {
                        "Processing": "Đang xử lý",
                        "Delivering": "Đang giao hàng",
                        "Received": "Đã nhận hàng",
                        "Succeed": "Đã nhận hàng",
                        "Cancelled": "Đã hủy"
                    }
                def create_detailed_response():
                    html_content = f"""
                    <div style="background: #f8f9fa; border-radius: 8px;">
                        <h2 style="color: #2c3e50; margin-bottom: 20px;">📊 Báo cáo doanh thu từ năm {year_query_start} đến năm {year_query_end}</h2>
                        
                            <h3 style="color: #27ae60; margin-top: 0;">💰 Tổng quan doanh thu</h3>
                            <p><strong>Tổng doanh thu: </strong><span style="color: #e74c3c; font-size: 18px; font-weight: bold;">{total_revenue:,} VNĐ</span></p>
                            <p>• Doanh thu từ đơn hàng: {total_order_revenue:,} VNĐ ({len(orders)} đơn)</p>
                            <p>• Doanh thu từ phiếu xuất: {total_receipt_revenue:,} VNĐ ({len(receipts)} phiếu)</p>
                        </div>
                    """
                    # Chi tiết đơn hàng
                    if orders:
                        html_content += """
                        <div style="background: white; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #3498db; margin-top: 0;">🛒 Chi tiết đơn hàng</h3>
                        """
                        for idx, order in enumerate(orders, 1):
                            order_time = order.get('createdAt', '').strftime('%H:%M') if order.get('createdAt') else 'N/A'
                            status_color = {
                                'Confirm': '#27ae60',
                                'Succeed': '#27ae60',
                                'Processing': '#f39c12', 
                                'Delivering': '#3498db',
                                'Cancelled': "#d81a13"
                            }.get(order.get('status', ''), '#95a5a6')
                            
                            html_content += f"""
                            <div style="border: 1px solid #ddd; margin-bottom: 15px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: #2c3e50;">#{order.get('_id', 'N/A')}</h4>
                                    <span style="background: {status_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                        {status_mapping.get(order.get('status', 'N/A') ,order.get('status', 'N/A'))}
                                    </span>
                                </div>
                                <p style="margin: 5px 0; color: #7f8c8d;">⏰ Thời gian: {order_time}</p>
                                <p style="margin: 5px 0;"><strong>Tổng tiền: {order.get('total', 0):,} VNĐ</strong></p>
                            """
                            
                            # Thông tin sản phẩm trong đơn hàng
                            products = order.get('products', [])
                            if products:
                                html_content += "<h5 style='margin: 10px 0 5px 0; color: #34495e;'>📦 Sản phẩm:</h5>"
                                html_content += "<div style='background: #f8f9fa; padding: 8px; border-radius: 4px;'>"
                                
                                for product in products:
                                    product_name = product.get('name', 'Sản phẩm không rõ tên')
                                    quantity = product.get('quantity', 0)
                                    price = product.get('price', 0)
                                    subtotal = quantity * price
                                    
                                    html_content += f"""
                                    <div style="margin-bottom: 8px;  background: white; border-radius: 3px;">
                                        <div style="font-weight: bold; color: #2c3e50; margin-bottom: 3px;">{product_name}</div>
                                        <div style="font-size: 14px; color: #7f8c8d;">
                                            Số lượng: {quantity} × {price:,} VNĐ = <strong style="color: #e74c3c;">{subtotal:,} VNĐ</strong>
                                        </div>
                                    </div>
                                    """
                                
                                html_content += "</div>"
                            
                            html_content += "</div>"
                        
                        html_content += "</div>"
                    
                    # Chi tiết phiếu xuất
                    if receipts:
                        html_content += """
                        <div style="background: white;  margin-bottom: 20px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #9b59b6; margin-top: 0;">📋 Chi tiết phiếu xuất</h3>
                        """
                        for idx, receipt in enumerate(receipts, 1):
                            receipt_time = receipt.get('createdAt', '').strftime('%H:%M') if receipt.get('createdAt') else 'N/A'
                            status_color = {
                                'pending': '#f39c12',
                                'confirmed': '#27ae60', 
                                'cancelled': '#d81a13'
                            }.get(order.get('status', ''), '#95a5a6')
                            
                            html_content += f"""
                            <div style="border: 1px solid #ddd; margin-bottom: 15px; border-radius: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h4 style="margin: 0; color: #2c3e50;">#{receipt.get('_id', 'N/A')}</h4>
                                </div>
                                <p style="margin: 5px 0; color: #7f8c8d;">⏰ Thời gian: {receipt_time}</p>
                                <p style="margin: 5px 0;"><strong>Tổng tiền: {receipt.get('total', 0):,} VNĐ</strong></p>
                            """
                            
                            # Thông tin sản phẩm trong đơn hàng
                            products = receipt.get('products', [])
                            if products:
                                html_content += "<h5 style='margin: 10px 0 5px 0; color: #34495e;'>📦 Sản phẩm:</h5>"
                                html_content += "<div style='background: #f8f9fa; padding: 8px; border-radius: 4px;'>"
                                
                                for product in products:
                                    product_name = product.get('name', 'Sản phẩm không rõ tên')
                                    quantity = product.get('quantity', 0)
                                    price = product.get('price', 0)
                                    subtotal = quantity * price
                                    
                                    html_content += f"""
                                    <div style="margin-bottom: 8px;  background: white; border-radius: 3px;">
                                        <div style="font-weight: bold; color: #2c3e50; margin-bottom: 3px;">{product_name}</div>
                                        <div style="font-size: 14px; color: #7f8c8d;">
                                            Số lượng: {quantity} × {price:,} VNĐ = <strong style="color: #e74c3c;">{subtotal:,} VNĐ</strong>
                                        </div>
                                    </div>
                                    """
                                
                                html_content += "</div>"
                            
                            html_content += "</div>"
                        
                        html_content += "</div>"
                    
                    html_content += """
                        <div style="text-align: center; margin-top: 5px; background: #ecf0f1; border-radius: 6px;">
                            <p style="margin: 0; color: #7f8c8d;">Hãy nhập thêm câu hỏi nếu có, tôi luôn sẵn sàng giải đáp các thắc mắc của bạn</p>
                        </div>
                    </div>
                    """
                    
                    return html_content
                
                # Kiểm tra xem message có chứa từ "chi tiết" không
                show_detail = any(keyword in message.lower() for keyword in ["chi tiết", "thông tin cụ thể", "doanh thu cụ thể"])
                if show_detail:
                    # Hiển thị chi tiết đầy đủ
                    response = create_detailed_response()
                else:
                    responses = [
                        f"Doanh từ năm {year_query_start} đến năm {year_query_end} là: {total_revenue:,} VNĐ. Trong đó doanh thu từ đơn hàng là {total_order_revenue:,} VNĐ, doanh thu từ phiếu xuất là {total_receipt_revenue:,} VNĐ. Ngoài ra, bạn còn câu hỏi nào khác không?",
                        f"Tổng doanh thu từ năm {year_query_start} đến năm {year_query_end}: {total_revenue:,} VNĐ. Doanh thu từ đơn hàng là {total_order_revenue:,} VNĐ. Doanh thu từ phiếu xuất là {total_receipt_revenue:,} VNĐ. Bạn cần tôi giải đáp gì nữa không?",
                        (
                            f"<p>Doanh thu từ năm {year_query_start} đến năm {year_query_end}:</p>"
                            f"<ul>"
                            f"<li>Doanh thu đơn hàng: {total_order_revenue:,} VNĐ ({len(orders)} đơn)</li>"
                            f"<li>Doanh thu phiếu xuất sản phẩm: {total_receipt_revenue:,} VNĐ ({len(receipts)} phiếu)</li>"
                            f"<li><strong>Tổng cộng: {total_revenue:,} VNĐ</strong></li>"
                            f"</ul>"
                            f"<p>Bạn có muốn xem chi tiết theo ngày hoặc so sánh với năm khác không?</p>"
                        )
                    ]
                    response = random.choice(response)
                return JSONResponse(content={"response": response})

            except Exception as e:
                return JSONResponse(content={"response": f"Đã xảy ra lỗi khi truy vấn doanh thu theo tháng: {str(e)}"})

        for intent in intents_chatbot_data['intents']:
            if tag == intent["tag"]:
                return JSONResponse(content={"response": random.choice(intent['responses'])})

    return JSONResponse(content={"response": "Tôi không hiểu bạn nói gì. Vui lòng nhập các thông tin rõ ràng hơn để tôi có thể hiểu, cảm ơn rất nhiều."})



@app.get("/")
async def run_server():
    try:
        return JSONResponse(content={"status": "run server"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "server is failed", "error": str(e)}, status_code=500)

# Trích xuất thông từ tên,....
def extract_value(patterns, message):
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1).strip()
        return ""
#  Trích xuất hai giá trị
def extract_date_range(patterns, message):
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            if match.groups() and len(match.groups()) >= 2:
                return match.group(1).strip(), match.group(2).strip()
    return "", ""

# Tải mô hình TensorFlow đã huấn luyện
model_path = "identify/model_seafood.keras"
loaded_model = load_model(model_path)

# Tải LabelEncoder để chuyển đổi nhãn từ mô hình
data_csv_path = 'A:/LV/ai/identify/dataset/dataset.csv'
data_csv = pd.read_csv(data_csv_path)
labels = [str(word).strip() for word in data_csv['name'].to_numpy()]
label_encoder = LabelEncoder()
label_encoder.fit(labels)

def get_price_range_for_label(label: str) -> dict:
    """Lấy khoảng giá cho nhãn"""
    price_info = price.PRICE_MAPPING.get(label, {'min': 0, 'max': 0})
    
    # Trả về giá cơ bản nếu không có khoảng giá
    if isinstance(price_info, (int, float)):
        return {'min': price_info, 'max': price_info}
    
    return price_info


SUPPORTED_FORMATS = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/bmp': '.bmp',
    'image/tiff': '.tiff'
}

def convert_to_rgb(image_bytes):
    """Chuyển đổi ảnh sang RGB format"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Chuyển đổi sang RGB nếu cần
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        # Nếu là ảnh grayscale, chuyển sang RGB
        if img.mode == 'L':
            img = img.convert('RGB')
        # Lưu ảnh vào buffer
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
    except Exception as e:
        raise ValueError(f"Lỗi khi chuyển đổi ảnh: {str(e)}")

def load_single_image(img_path: str):
    """Tải và xử lý ảnh từ đường dẫn"""
    try:
        # Đọc file ảnh
        image = tf.io.read_file(img_path)
        
        # Decode ảnh
        try:
            image = tf.image.decode_image(image, channels=3)
        except:
            raise ValueError("Không thể decode ảnh")

        # Xử lý ảnh
        image = tf.image.resize(image, [250, 250])
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, axis=0)
        
        # Data augmentation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        
        return image

    except Exception as e:
        raise ValueError(f"Lỗi khi xử lý ảnh: {str(e)}")

def predict_single_image(model, img_path, label_encoder):
    """Thực hiện dự đoán trên một ảnh"""
    try:
        # Tải và xử lý ảnh
        image = load_single_image(img_path)

        # Dự đoán
        predictions = model.predict(image)
        
        # Xử lý kết quả
        predicted_idx = np.argmax(predictions[0])
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(predictions[0][predicted_idx])
        predicted_price_range = get_price_range_for_label(predicted_label)

        # Lấy top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                "label": label_encoder.inverse_transform([idx])[0],
                "confidence": float(predictions[0][idx]),
                "price": get_price_range_for_label(label_encoder.inverse_transform([idx])[0])
            }
            for idx in top_3_idx
        ]

        return predicted_label, confidence, top_3_predictions, predicted_price_range

    except Exception as e:
        raise ValueError(f"Lỗi khi dự đoán: {str(e)}")

@app.post("/image")
async def predict_image(file: UploadFile = File(...)):
    """API endpoint cho dự đoán ảnh"""
    
    # Kiểm tra định dạng file
    if file.content_type not in SUPPORTED_FORMATS:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Định dạng không được hỗ trợ",
                "supported_formats": list(SUPPORTED_FORMATS.keys())
            }
        )
    
    try:
        # Đọc nội dung file
        contents = await file.read()
        
        # Chuyển đổi ảnh sang RGB nếu cần
        contents = convert_to_rgb(contents)
        
        # Tạo file tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Thực hiện dự đoán
        predicted_label, confidence, top_3_predictions, predicted_price_range = predict_single_image(
            loaded_model,  # Sử dụng mô hình TensorFlow
            temp_file_path,
            label_encoder
        )

        # Xóa file tạm
        os.unlink(temp_file_path)

        # Lấy thông tin về file ảnh
        with Image.open(io.BytesIO(contents)) as img:
            image_info = {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
            }

        return JSONResponse(content={
            "status": "success",
            "predicted_class": predicted_label,
            "confidence": confidence,
            "predicted_price_range": predicted_price_range,
            "top_3_predictions": top_3_predictions,
            "file_type": file.content_type,
            "image_info": image_info
        })

    except Exception as e:
        # Đảm bảo xóa file tạm nếu có lỗi
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
            
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e)
            }
        )
@app.get("/supported-formats")
async def get_supported_formats():
    """Trả về danh sách các định dạng được hỗ trợ"""
    return {
        "supported_formats": list(SUPPORTED_FORMATS.keys()),
        "status": "success"
    }