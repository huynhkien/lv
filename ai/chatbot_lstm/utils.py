import numpy as np
import re
import torch
from collections import Counter
from torch.utils.data import Dataset
import unicodedata
import random

class TextPreprocessor:
    """
    Xử lý văn bản tiếng Việt cho chatbot
    """
    def __init__(self, max_vocab_size=5000, max_seq_len=30):
        self.max_vocab_size = max_vocab_size # Số lượng từ tối đa mà chương trình sẽ nhớ
        self.max_seq_len = max_seq_len # Độ dài tối đa của một câu hỏi
        self.word2idx = {'<PAD>': 0, '<UNK>': 1} # Chuyển đổi từ chữ => số
        # Ví dụ: self.word2idx = {'<PAD>': 0, '<UNK': 1, 'Xin': 2, 'Cam': 3} 
        self.idx2word = {0: '<PAD>', 1: '<UNK>'} # Chuyển đổi từ số => chữ
        # Ví dụ: self.word2idx = {0:'<PAD>', 1:'<UNK', 2:'Xin',3:'Cam'} 
        self.tag2idx = {} # Tương tự => chuyển đổi tag của từng tập kịch bản
        self.idx2tag = {}
        
    def clean_text(self, text):
        """Làm sạch văn bản tiếng Việt được cải thiện"""
        # Chuẩn hóa unicode
        text = unicodedata.normalize('NFC', text)
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Xử lý số: "123" -> "<NUM>"
        text = re.sub(r'\d+', '<NUM>', text)
        
        # Giữ lại ký tự tiếng Việt và một số ký tự đặc biệt
        text = re.sub(r'[^\w\s\?\!\.\,àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', text)
        
        # Xử lý dấu câu
        text = re.sub(r'([.!?])', r' \1 ', text)
        text = re.sub(r'([,])', r' \1 ', text)
        
        # Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_vocab_from_intents(self, intents_data):
        """Xây dựng từ điển từ dữ liệu intents"""
        all_words = []
        tags = set()
        
        for intent in intents_data:
            tag = intent['tag']
            tags.add(tag)
            
            # Xử lý patterns
            for pattern in intent['patterns']:
                cleaned = self.clean_text(pattern)
                words = cleaned.split() # tách thành các từ trong câu
                all_words.extend(words) # Thêm tất cả các từ vào all_word
        
        # Xây dựng tag mapping
        for i, tag in enumerate(sorted(tags)):
            self.tag2idx[tag] = i
            self.idx2tag[i] = tag
        
        # Đếm tần suất từ
        word_freq = Counter(all_words)
        
        # Lấy top từ phổ biến nhất
        most_common = word_freq.most_common(self.max_vocab_size - 2)  # -2 cho PAD và UNK
        
        # Thêm vào từ điển
        for word, freq in most_common:
            if freq >= 1:  # Chỉ giữ từ xuất hiện ít nhất 1 lần
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def text_to_sequence(self, text):
        """Chuyển văn bản thành một dãy số"""
        cleaned = self.clean_text(text) 
        words = cleaned.split()
        
        sequence = []
        for word in words:
            if word in self.word2idx: #Kiểm tra từ đã có trong từ điển hay không?
                sequence.append(self.word2idx[word]) # Có => thêm số 
            else:
                sequence.append(self.word2idx['<UNK>']) # Không => thêm UNK (1)
        
        # Kiểm tra độ dài của  sequence
        if len(sequence) < self.max_seq_len: # Nếu bé hơn độ dài mặc định đã cho => thêm dãy để để đủ độ dài 30 từ
            sequence.extend([self.word2idx['<PAD>']] * (self.max_seq_len - len(sequence)))
        else: # Nếu lớn hơn => mặc đinh lấy 30 từ như đã cho
            sequence = sequence[:self.max_seq_len]
            
        return sequence
    
    def prepare_training_data(self, intents_data):
        """Chuẩn bị dữ liệu training"""
        X = []
        y = []
        
        for intent in intents_data:
            tag = intent['tag']
            tag_idx = self.tag2idx[tag]
            
            for pattern in intent['patterns']:
                sequence = self.text_to_sequence(pattern)
                X.append(sequence)
                y.append(tag_idx)
        
        return np.array(X), np.array(y)
    
class IntentDataset(Dataset):
    """Tạo dataset"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.X[idx],
            'labels': self.y[idx]
        }
    
class DataAugmentation:
    """Tăng cường dữ liệu"""
    @staticmethod
    def synonym_replacement(text):
        """Thay thế từ đồng nghĩa
           Thêm các từ đồng nghĩa vào tập train
        """
        synonyms = {
            'sản phẩm': ['hàng hóa', 'mặt hàng', 'món hàng'],
            # 'báo cáo': ['thống kê', 'report', 'bảng kê'],
            # 'doanh số': ['doanh thu', 'số liệu bán hàng', 'kết quả kinh doanh'],
            # 'nhân viên': ['staff', 'người làm', 'cán bộ'],
            # 'khách hàng': ['user', 'người dùng', 'tài khoản'],
            'cao': ['nhiều', 'lớn'],
            'thấp': ['kém', 'không cao', 'không lớn', 'không nhiều'],
            # 'doanh thu': ['thu nhập', 'lợi nhuận', 'tổng thu'],
            # 'thông tin': ['dữ liệu', 'tin tức', 'nội dung', 'số liệu', 'data'],
            'địa chỉ': ['nơi ở', 'chỗ ở', 'vị trí', 'địa điểm', 'vị trí cư trú', 'nơi cư trú', 'thông tin nơi ở', 'thông tin liên hệ', 'location', 'address'],
            # 'hiển thị': ['trình bày', 'xuất hiện', 'hiện ra', 'cho thấy', 'render', 'hiện thị', 'hiện lên', 'trưng bày', 'thể hiện'],
            'số điện thoại': ['sdt', 'số máy', 'số liên lạc', 'số gọi', 'số mobile', 'mobile', 'số liên hệ', 'điện thoại', 'phone number', 'contact number'],
            'email': ['địa chỉ email', 'mail', 'địa chỉ mail', 'thư điện tử', 'hòm thư', 'email address', 'contact email', 'mail cá nhân', 'mail liên hệ'],
            'tìm': ['tìm kiếm', 'tra', 'tra cứu', 'kiếm', 'lục', 'tìm thấy', 'xem', 'khám phá', 'truy vấn', 'tìm ra'],
            # 'xử lý': ['đang xử lý', 'đã xử lý', 'chờ xử lý', 'đang thực hiện', 'đang giải quyết', 'đang kiểm tra', 'trong quá trình xử lý', 'xử lý đơn hàng', 'đang tiến hành'],
            # 'chi tiết': ['thông tin cụ thể', 'nội dung đầy đủ', 'mô tả', 'dữ liệu chi tiết', 'phân tích', 'chi tiết đầy đủ'],
            # 'đang giao': ['đang vận chuyển', 'đang giao hàng', 'trên đường giao', 'đơn đang được giao', 'đang chuyển', 'shipper đang giao', 'đang gửi hàng', 'hàng đang trên đường', 'giao hàng đang diễn ra'],
            # 'đã nhận': ['đã nhận hàng', 'đã giao xong', 'giao thành công', 'khách đã nhận', 'đã hoàn tất', 'đã hoàn thành', 'đã xong', 'đã kết thúc đơn hàng', 'đơn đã hoàn tất'],
            # 'đã huỷ': ['đơn bị huỷ', 'huỷ đơn', 'đơn đã huỷ', 'huỷ thành công', 'đã hủy đơn hàng', 'đơn hàng bị hủy', 'đã huỷ xong', 'đã huỷ thành công', 'đơn không còn hiệu lực'],
        }
        text_lower = text.lower()
        has_keyword = False
        for keyword in synonyms.keys():
            if keyword in text_lower:
                has_keyword = True
                break
        
        # Nếu không có từ khóa nào, chỉ trả về text gốc
        if not has_keyword:
            return [text]
        
        variations = [text]
        sorted_phrases = sorted(synonyms.keys(), key=len, reverse=True)
        
        for original_phrase in sorted_phrases:
            if original_phrase in text_lower:
                for synonym in synonyms[original_phrase]:
                    pattern = re.compile(re.escape(original_phrase), re.IGNORECASE)
                    new_text = pattern.sub(synonym, text)
                    
                    if new_text not in variations and new_text != text:
                        variations.append(new_text)
        
        return variations