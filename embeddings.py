# embeddings.py
from sentence_transformers import SentenceTransformer
import json
import pickle
import os

class EmbeddingHandler:
    def __init__(self):
        # Sử dụng Vietnamese_Embedding từ AITeamVN
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    def load_json_data(self, file_path):
        """Tải dữ liệu từ file JSON."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"File {file_path} không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
            return []
        except json.JSONDecodeError:
            print(f"File {file_path} không đúng định dạng JSON.")
            return []

    def generate_embeddings(self, texts):
        """Tạo embeddings từ danh sách các văn bản."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings

    def process_handbook(self, file_path, output_path='embeddings_data.pkl'):
        """Xử lý dữ liệu từ file JSON, tạo embeddings và lưu vào file để tái sử dụng."""
        # Tải dữ liệu
        data = self.load_json_data(file_path)
        if not data:
            return [], [], []

        # Kết hợp regulation và content với ký tự xuống dòng \n
        texts = [f"{item.get('regulation', '')}\n{item.get('content', '')}" for item in data]
        
        # Tạo embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Lưu dữ liệu vào file để tái sử dụng
        output_data = {
            'texts': texts,
            'embeddings': embeddings,
            'metadata': data
        }
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"Đã lưu dữ liệu vào {output_path} để tái sử dụng.")

        return texts, embeddings, data

    def load_saved_embeddings(self, file_path='embeddings_data.pkl'):
        """Tải dữ liệu embeddings đã lưu trước đó."""
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Đã tải dữ liệu từ {file_path}.")
            return data['texts'], data['embeddings'], data['metadata']
        else:
            print(f"File {file_path} không tồn tại. Vui lòng chạy process_handbook trước.")
            return [], [], []

if __name__ == "__main__":
    # Đường dẫn đến file JSON
    json_file_path = os.path.join('data', 'data_raw.json')
    output_file_path = 'embeddings_data.pkl'

    handler = EmbeddingHandler()
    
    # Kiểm tra xem file embeddings đã tồn tại chưa
    if os.path.exists(output_file_path):
        print("Tải embeddings từ file đã lưu...")
        texts, embeddings, data = handler.load_saved_embeddings(output_file_path)
    else:
        print("Tạo embeddings mới...")
        texts, embeddings, data = handler.process_handbook(json_file_path, output_file_path)
    
    print(f"Đã xử lý {len(embeddings)} embeddings.")
    # In thử một mẫu để kiểm tra định dạng
    if texts:
        print("\nMẫu nội dung đầu tiên:")
        print(texts[16])