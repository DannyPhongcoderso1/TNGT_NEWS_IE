import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from .config import MODEL_PATHS, DEVICE

class PhoBERTFeatureExtractor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print(f"--- [INFO] Loading Vectorizer Base ({MODEL_PATHS['VECTORIZER_BASE']})...")
            cls._instance = super(PhoBERTFeatureExtractor, cls).__new__(cls)
            
            # 1. Tuân thủ file evaluate: dùng use_fast=False
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATHS["VECTORIZER_BASE"], 
                use_fast=False 
            )
            cls._instance.model = AutoModel.from_pretrained(MODEL_PATHS["VECTORIZER_BASE"]).to(DEVICE)
            cls._instance.model.eval()
        return cls._instance

    def vectorize_token_level(self, text):
        """
        Logic học hỏi từ hàm extract_features trong evaluate-ner-re.ipynb
        """
        # Bước 1: Tách từ (Giả định text đầu vào ngăn cách bởi khoảng trắng)
        # Trong evaluate file, input là 'tkns' (list). Ở đây ta có string nên cần split.
        tkns = text.split() 

        # Bước 2: Tokenize cả câu (để lấy input_ids và embeddings)
        # is_split_into_words=True báo cho tokenizer biết input đã là list các từ
        inputs = self.tokenizer(
            tkns, 
            is_split_into_words=True, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Lấy last_hidden_state của batch đầu tiên [seq_len, hidden_dim]
        embs = outputs.last_hidden_state[0].cpu().numpy()

        wids = [None] # Bắt đầu bằng [CLS] -> None
        
        for i, w in enumerate(tkns):
            # Encode riêng từng từ để xem nó bị tách thành bao nhiêu subwords
            # add_special_tokens=False để không thêm CLS/SEP vào từng từ con
            subword_ids = self.tokenizer.encode(w, add_special_tokens=False)
            wids.extend([i] * len(subword_ids))
            
        # Cắt hoặc thêm None cho khớp với độ dài input_ids (do truncation hoặc padding)
        # inputs['input_ids'].shape[1] là chiều dài thực tế sau khi qua tokenizer
        seq_len = inputs['input_ids'].shape[1]
        wids = (wids + [None])[:seq_len] # Thêm [SEP] -> None và cắt đuôi
        
        # Bước 4: Mapping Embedding về Word
        word_vectors = []
        seen_ids = set() # Đánh dấu các từ đã lấy vector (chỉ lấy subword đầu tiên)
        
        for idx, wid in enumerate(wids):
            # idx: vị trí trong chuỗi subwords (tương ứng với embs)
            # wid: chỉ số của từ gốc (0, 1, 2...)
            
            if wid is not None and wid not in seen_ids and idx < len(embs):
                # Logic: X_flat.append(embs[idx]) trong file evaluate
                word_vectors.append(embs[idx])
                seen_ids.add(wid)
        
        # Đảm bảo số lượng vector trả về bằng số lượng từ (xử lý edge case)
        # Nếu model cắt bớt (truncation), ta cũng chỉ trả về vector của những từ còn lại
        return np.array(word_vectors)

    def vectorize_sentence_level(self, text):
        """Vector hóa cả câu (dùng cho RE) - Giữ nguyên"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    
    def extract_crf_features(self, text):
        """Dùng cho CRF"""
        vectors = self.vectorize_token_level(text)
        features = []
        for vec in vectors:
            features.append({f'dim_{i}': v for i, v in enumerate(vec)})
        return [features]