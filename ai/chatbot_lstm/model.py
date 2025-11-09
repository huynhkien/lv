import torch
import torch.nn as nn
import torch.nn.functional as F


class ChatbotLSTM(nn.Module):
    """
    Mô hình LSTM được tối ưu cho tiếng Việt
    vocab_size: số lượng từ trong từ điểm
    embedding_dim: kích thước vector nhúng
    hidden_size: số node ẩn trong mỗi LSTM layer
    num_layers: số lớp LSTM
    num_classes: số lượng lớp đầu ra
    drop_out: xác xuất tránh overfitting
    """
    def __init__(self, vocab_size, embedding_dim=200, hidden_size=100, 
                 num_layers=2, num_classes=57, dropout=0.4):
        super(ChatbotLSTM, self).__init__()
        # Chuyển đổi từ index => vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Dropout => tránh overfitting
        self.embedding_dropout = nn.Dropout(0.2)
        # Lớp layer LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers,  
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Tính trọng số attention cho mỗi bước thời gian
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Bộ phân loại classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),      # Giảm lớp hidden_size 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),     
            nn.ReLU(),
            nn.Dropout(dropout // 2),                     # Giảm dropout cuối
            nn.Linear(hidden_size // 2, num_classes)      #
        )
        
        # Khởi tạo trọng số
        self._init_weights()
        
    def _init_weights(self):
        """Khởi tạo trọng số đơn giản"""
        # Chỉ init những layer quan trọng
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)  # Padding token = 0
        
        # Init classifier weights
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Simple attention (hiệu quả và ổn định)
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(context_vector)
        
        return output, attention_weights