import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from chatbot_lstm.utils import TextPreprocessor, IntentDataset, DataAugmentation
from chatbot_lstm.model import ChatbotLSTM  
import json

class ChatbotTrainer:
    """Huấn luyện chatbot"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Sử dụng thêm label smoothing để giúp mô hình ổn định hơn
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
        # Sử dụng AdamW để giảm overfitting
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=3e-4,           # Learning rate vừa phải
            weight_decay=1e-4, # Weight decay nhẹ
        )
        
        # Giảm learning rate nếu acc không cải thiện sau một số lần lặp
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=7, factor=0.5, verbose=True
        )
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs, _ = self.model(input_ids)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(dataloader), correct / total
    # Đánh giá mô hình
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs, _ = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(dataloader), correct / total
    # Thực hiện huấn luyện mô hình
    def train(self, train_dataloader, val_dataloader, epochs=80, patience=15):
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_dataloader)
            val_loss, val_acc = self.evaluate(val_dataloader)
            
            # Dừng khi sau một số lần lặp độ chính xác không được cải thiện
            self.scheduler.step(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 50)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
                
        print(f"Final best accuracy: {best_val_acc:.4f}")

class ChatbotPredictor:
    """Class để predict intent từ input text"""
    def __init__(self, model, preprocessor, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.preprocessor = preprocessor
        self.device = device
        self.model.eval()
    
    def predict(self, text, return_confidence=True):
        """Predict intent cho input text"""
        # Xử lý input
        sequence = self.preprocessor.text_to_sequence(text)
        input_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs, attention_weights = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_tag = self.preprocessor.idx2tag[predicted.item()]
            confidence_score = confidence.item()
        
        if return_confidence:
            return predicted_tag, confidence_score, attention_weights.cpu().numpy()
        else:
            return predicted_tag

def main():
    """Hàm main được cải thiện hoàn chỉnh"""
    
    # 1. Đọc dữ liệu từ file intents.json
    with open('intents_chatbot.json', 'r', encoding='utf-8') as f:
        intents_data = json.load(f)
    augmenter = DataAugmentation()
    augmented_intents = []

    for intent in intents_data['intents']:
        new_patterns = []
        for pattern in intent['patterns']:
            variations = augmenter.synonym_replacement(pattern)
            new_patterns.extend(variations)
        
        # Loại bỏ trùng lặp
        unique_patterns = list(set(new_patterns))

        # Gán lại vào intent mới
        augmented_intents.append({
            'tag': intent['tag'],
            'patterns': unique_patterns,
            'responses': intent['responses']
        })
    # Điều chỉnh parameters
    preprocessor = TextPreprocessor(
        max_vocab_size=5000,  
        max_seq_len=40        
    )
    
    # 3. Xây dựng vocabulary
    preprocessor.build_vocab_from_intents(augmented_intents)
    
    # 4. Chuẩn bị dữ liệu
    X, y = preprocessor.prepare_training_data(augmented_intents)
    # 5. Chia tập train , tập val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  
    )
    # 6. Tạo tập dataset để thực hiện huấn luyện mô hình
    train_dataset = IntentDataset(X_train, y_train)
    val_dataset = IntentDataset(X_val, y_val)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)   
    # 7. Huấn luyện mô hình
    model = ChatbotLSTM(  
        vocab_size=len(preprocessor.word2idx),
        embedding_dim=300,        
        hidden_size=256,          
        num_layers=3,             
        num_classes=len(preprocessor.tag2idx),
        dropout=0.5               
    )
    
    print(f"Vocab size: {len(preprocessor.word2idx)}")
    print(f"Number of classes: {len(preprocessor.tag2idx)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    trainer = ChatbotTrainer(model)
    trainer.train(
        train_dataloader, 
        val_dataloader, 
        epochs=150,   
        patience=25   
    )
    # Lưu model
    data = {
    "model_state": model.state_dict(),
    "vocab_size": len(preprocessor.word2idx),
    "embedding_dim": 300,
    "hidden_size": 256,
    "num_layers": 3,
    "num_classes": len(preprocessor.tag2idx),
    "word2idx": preprocessor.word2idx,
    "tag2idx": preprocessor.tag2idx,
    "idx2tag": preprocessor.idx2tag,
    "max_seq_len": preprocessor.max_seq_len,
}

    torch.save(data, "chatbot_public_model.pth")
    # Lấy model đã lưu
    # model = torch.load('moderate_chatbot_model.pth')
    predictor = ChatbotPredictor(model, preprocessor)
    
    # Test
    test_texts = [
        "sản phẩm nào còn dư thừa trong kho",
        "xin chào, tôi cần hỗ trợ",
        "kho hàng nào đang nhiều nhất",
        "tôi cần tìm thông tin nhân viên"
    ]
    
    print("\n=== Test Predictions ===")
    for text in test_texts:
        pred_tag, confidence, attention = predictor.predict(text)
        print(f"Input: '{text}'")
        print(f"Predicted: {pred_tag} (confidence: {confidence:.4f})")
        
    
    return model, preprocessor, predictor

if __name__ == "__main__":
    model, preprocessor, predictor = main()