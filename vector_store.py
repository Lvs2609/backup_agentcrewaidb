from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import os
from dotenv import load_dotenv
from embeddings import EmbeddingHandler

load_dotenv()

class VectorStore:
    def __init__(self, collection_name="ictu_handbook"):
        # Khởi tạo client Qdrant với timeout tăng lên
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60  # Tăng timeout lên 60 giây
        )
        self.collection_name = collection_name
        self.vector_size = 768  # Kích thước vector

    def create_collection(self):
        """Tạo hoặc tái tạo collection trong Qdrant."""
        try:
            # Kiểm tra xem collection đã tồn tại chưa
            if self.client.collection_exists(collection_name=self.collection_name):
                print(f"Collection {self.collection_name} đã tồn tại. Xóa và tạo lại...")
                self.client.delete_collection(collection_name=self.collection_name)
            
            # Tạo collection mới
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Đã tạo collection {self.collection_name} thành công với vector_size = {self.vector_size}.")
        except Exception as e:
            print(f"❌ Lỗi khi tạo collection: {e}")
            raise

    def upsert_vectors(self, texts, embeddings, metadata, batch_size=100):
        """Lưu trữ vectors vào Qdrant theo batch."""
        if embeddings is None or len(embeddings) == 0:
            print("❌ Không có embeddings để lưu trữ.")
            return
        
        try:
            # Kiểm tra kích thước của embedding đầu tiên
            if embeddings is not None and len(embeddings) > 0:
                print(f"Kích thước của embedding đầu tiên: {len(embeddings[0])} chiều.")

            # Chia dữ liệu thành các batch
            total_vectors = len(embeddings)
            for start_idx in range(0, total_vectors, batch_size):
                end_idx = min(start_idx + batch_size, total_vectors)
                batch_texts = texts[start_idx:end_idx]
                batch_embeddings = embeddings[start_idx:end_idx]
                batch_metadata = metadata[start_idx:end_idx]

                # Tạo points cho batch hiện tại
                points = [
                    PointStruct(
                        id=idx,
                        vector=embedding.tolist(),
                        payload={"text": text, "metadata": meta}
                    )
                    for idx, (text, embedding, meta) in enumerate(zip(batch_texts, batch_embeddings, batch_metadata), start=start_idx)
                ]

                # Upsert batch
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"✅ Đã lưu trữ batch {start_idx // batch_size + 1}: {len(points)} vectors (từ {start_idx} đến {end_idx - 1}).")

            print(f"✅ Đã lưu trữ toàn bộ {total_vectors} vectors vào Qdrant thành công.")
            # In thông tin mẫu của vector đầu tiên
            if total_vectors > 0:
                print("\nMẫu vector đầu tiên đã lưu trữ:")
                print(f"ID: 4")
                print(f"Text: {texts[30][:100]}...")  # In 100 ký tự đầu tiên
                print(f"Metadata: {metadata[30]}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu trữ vectors: {e}")
            raise

    def search(self, query_embedding, top_k=5):
        """Tìm kiếm vectors trong Qdrant."""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            print(f"✅ Tìm kiếm thành công. Tìm thấy {len(results)} kết quả.")
            return results
        except Exception as e:
            print(f"❌ Lỗi khi tìm kiếm: {e}")
            return []

if __name__ == "__main__":
    print("🚀 Bắt đầu chương trình lưu trữ vectors vào Qdrant...")
    
    # Đường dẫn đến file JSON và file embeddings đã lưu
    json_file_path = os.path.join('data', 'split_datanew.json')
    output_file_path = 'embeddings_data.pkl'

    # Tạo hoặc tải embeddings
    handler = EmbeddingHandler()
    if os.path.exists(output_file_path):
        print("📂 Tải embeddings từ file đã lưu...")
        texts, embeddings, data = handler.load_saved_embeddings(output_file_path)
    else:
        print("🛠️ Tạo embeddings mới...")
        texts, embeddings, data = handler.process_handbook(json_file_path, output_file_path)
    
    # Kiểm tra embeddings
    if embeddings is None or len(embeddings) == 0:
        print("❌ Không có embeddings để lưu trữ. Thoát chương trình.")
        exit()

    print(f"📊 Tổng số embeddings: {len(embeddings)}")
    if len(embeddings) > 0:
        print(f"Kích thước của embedding đầu tiên: {len(embeddings[0])} chiều.")

    # Lưu trữ vào Qdrant
    store = VectorStore()
    store.create_collection()
    store.upsert_vectors(texts, embeddings, data, batch_size=100)

    print("🎉 Hoàn tất chương trình!")