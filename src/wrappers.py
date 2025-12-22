import torch
import numpy as np
from transformers import pipeline

# --- HÀM PHỤ TRỢ: GỘP BIO TAGS THÀNH ENTITY ---
def aggregate_entities(tokens, tags):
    """
    Input: 
      tokens = ['Tai', 'nạn', 'tại', 'Hà', 'Nội']
      tags   = ['O',   'O',   'O',   'B-LOC', 'I-LOC']
    Output:
      [{'word': 'Hà Nội', 'entity_group': 'LOC'}]
    """
    entities = []
    current_entity = None
    
    for token, tag in zip(tokens, tags):
        if tag == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
            
        # Tách B-LOC thành prefix=B, label=LOC
        parts = tag.split('-')
        label = parts[1] if len(parts) > 1 else tag
        prefix = parts[0] if len(parts) > 1 else ''
        
        if prefix == 'B':
            if current_entity:
                entities.append(current_entity)
            current_entity = {"word": token, "entity_group": label}
        elif prefix == 'I':
            if current_entity and current_entity['entity_group'] == label:
                current_entity['word'] += " " + token
            else:
                # Trường hợp I- nằm lẻ loi (coi như bắt đầu mới)
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"word": token, "entity_group": label}
        else:
            # Trường hợp nhãn không có B/I (ít gặp nhưng đề phòng)
            if current_entity:
                entities.append(current_entity)
            current_entity = {"word": token, "entity_group": tag}
             
    if current_entity:
        entities.append(current_entity)
    return entities

# --- CÁC CLASS WRAPPER ---
class BasePredictor:
    def __init__(self, model_type):
        self.model_type = model_type

class NERPredictor(BasePredictor):
    def __init__(self, model_type, model, tokenizer=None, feature_extractor=None, label_map=None):
        super().__init__(model_type)
        self.model = model
        
        if model_type == 'DL':
            self.pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, 
                                 aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
        else:
            self.feature_extractor = feature_extractor
            self.label_map = label_map # {0: 'O', 1: 'B-LOC'}

    def predict(self, text):
        if self.model_type == 'DL':
            # HuggingFace pipeline đã tự gộp entity
            return self.pipe(text)
        
        else:
            # --- LOGIC CHO ML (SVM/LogReg/CRF) ---
            
            # 1. Lấy vector
            # Lưu ý: feature_extractor.vectorize_token_level dùng split() nên ta cũng dùng split() để lấy token gốc
            tokens = text.split()
            
            if hasattr(self.model, "predict_marginals") or "CRF" in str(type(self.model)):
                vectors = self.feature_extractor.extract_crf_features(text)
                # CRF predict trả về list of list labels luôn
                preds = self.model.predict(vectors)[0] 
            else:
                vectors = self.feature_extractor.vectorize_token_level(text)
                # SVM/LogReg predict trả về mảng số
                pred_ids = self.model.predict(vectors)
                
                # Map ID -> Label String
                preds = []
                for pid in pred_ids:
                    # Xử lý trường hợp label_map key là string hoặc int
                    label = self.label_map.get(pid) or self.label_map.get(str(pid)) or 'O'
                    preds.append(label)

            # 2. Gộp Tokens + Tags thành Entities
            # Cắt ngắn nếu số lượng token và pred không khớp (do tokenizer)
            min_len = min(len(tokens), len(preds))
            entities = aggregate_entities(tokens[:min_len], preds[:min_len])
            
            return entities

class REPredictor(BasePredictor):
    def __init__(self, model_type, model, tokenizer=None, feature_extractor=None, label_encoder=None):
        super().__init__(model_type)
        self.model = model
        
        if model_type == 'DL':
            self.tokenizer = tokenizer
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.feature_extractor = feature_extractor
            self.label_encoder = label_encoder

    def predict(self, text):
        if self.model_type == 'DL':
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            pred_id = logits.argmax().item()
            return self.model.config.id2label[pred_id]
        
        else:
            vec = self.feature_extractor.vectorize_sentence_level(text)
            pred_id = self.model.predict([vec])[0]
            
            if self.label_encoder:
                if hasattr(self.label_encoder, 'inverse_transform'):
                    return self.label_encoder.inverse_transform([pred_id])[0]
                elif isinstance(self.label_encoder, dict):
                    return self.label_encoder.get(pred_id) or self.label_encoder.get(str(pred_id)) or "UNKNOWN"
            return pred_id