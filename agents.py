from crewai import Agent, Task, Crew
from embeddings import EmbeddingHandler
from vector_store import VectorStore
from llm_config import get_llm
from keywords import KEYWORDS
import re

class ChatBotAgent:
    def __init__(self):
        self.embedding_handler = EmbeddingHandler()
        self.vector_store = VectorStore()
        self.llm = get_llm()
        self.keywords = KEYWORDS

        # Khởi tạo độ ưu tiên cho các chủ đề
        self.priority_weights = {
            "khoa": 5,
            "ngành học": 5,
            "học phí": 5,
            "xếp loại học lực": 2,
            "xếp hạng học lực": 2,
            "đăng nhập": 1,
            "đổi mật khẩu": 1,
            "đăng ký học": 1,
            "thanh toán học phí": 1,
            "hoạt động ngoại khóa": 1,
            "xem lịch thi": 1,
            "tra cứu điểm": 1,
            "hủy học phần": 1,
            "gửi tin nhắn": 1,
            "tổng số tín chỉ": 2,
        }
        # Lưu trữ độ ưu tiên cao nhất từ các câu hỏi trước
        self.max_priority = 0
        # Đếm số lượng câu hỏi để reset max_priority nếu cần
        self.query_count = 0
        # Thêm danh sách để lưu trữ lịch sử chat không giới hạn
        self.chat_history = []  # Danh sách lưu trữ các cặp (query, response)

        # Lưu trữ bảng chương trình đào tạo đại trà
        self.training_program = {
            "II: Nghệ thuật": {
                "majors": ["Thiết kế đồ họa"],
                "Khóa 17": 372000,
                "Khóa 18-19": 369200,
                "Khóa 20-21": 387000
            },
            "III: Kinh doanh và quản lý, pháp luật": {
                "majors": ["Hệ thống thông tin quản lý", "Quản trị văn phòng", "Thương mại điện tử", "Kinh tế số"],
                "Khóa 17": 387500,
                "Khóa 18-19": 384600,
                "Khóa 20-21": 403200
            },
            "V: Máy tính và công nghệ thông tin, Công nghệ kỹ thuật…": {
                "majors": [
                    "Công nghệ thông tin", "Khoa học máy tính", "Truyền thông và mạng máy tính", "Kỹ thuật phần mềm",
                    "Hệ thống thông tin", "An toàn thông tin", "Công nghệ kỹ thuật điện, điện tử",
                    "Công nghệ ô tô và Giao thông thông minh", "Công nghệ kỹ thuật điều khiển và tự động hóa",
                    "Công nghệ kỹ thuật máy tính", "Công nghệ kỹ thuật điện tử, viễn thông", "Kỹ thuật y sinh",
                    "Kỹ thuật cơ điện tử thông minh và robot"
                ],
                "Khóa 17": 453000,
                "Khóa 18-19": 450000,
                "Khóa 20-21": 467700
            },
            "VII: Báo chí và thông tin…": {
                "majors": ["Truyền thông đa phương tiện", "Công nghệ truyền thông"],
                "Khóa 17": 372000,
                "Khóa 18-19": 369200,
                "Khóa 20-21": 387000
            }
        }
        # print("🔍 self.training_program:", self.training_program)

        self.agent = Agent(
            role="ICTU Handbook Assistant",
            goal="Truy xuất và diễn giải thông tin từ sổ tay ICTU để trả lời câu hỏi của sinh viên",
            backstory="Tôi là trợ lý ảo hỗ trợ sinh viên ICTU, truy xuất thông tin từ sổ tay sinh viên và cung cấp câu trả lời chính xác, ngắn gọn.",
            verbose=True,
            llm=self.llm
        )
        
    def is_true_false_question(self, query):
        """Kiểm tra xem câu hỏi có phải dạng đúng/sai không"""
        true_false_keywords = ["đúng không", "có phải", "không phải", "có đúng", "có thật", "thật không"]
        return any(keyword in query.lower() for keyword in true_false_keywords)

    def is_comparison_question(self, query):
        """Kiểm tra xem câu hỏi có phải dạng so sánh không"""
        comparison_keywords = ["so sánh", "khác nhau", "khác biệt", "giữa", "và", "cao hơn", "thấp hơn", "hơn"]
        query_lower = query.lower()
        comparison_topics = [
            "học phí", "nội trú", "ngoại trú", "điều kiện nhập học", "chương trình đào tạo", 
            "hoạt động ngoại khóa"
        ]
        return (
            any(keyword in query_lower for keyword in comparison_keywords) and
            any(topic in query_lower for topic in comparison_topics)
        )

    def compare_tuition_fees(self, query):
        query_lower = query.lower()
        majors = [m for block in self.training_program.values() for m in block["majors"]]
        
        # Debug: In câu hỏi và danh sách majors
        print(f"🔍 Query Lower: {query_lower}")
        print(f"🔍 Majors: {majors}")
        
        # Tìm ngành bằng regex, giữ thứ tự xuất hiện
        found_majors = []
        for major in majors:
            # Tạo pattern regex cho tên ngành, thoát các ký tự đặc biệt
            pattern = re.compile(re.escape(major.lower()), re.IGNORECASE)
            match = pattern.search(query_lower)
            if match and major not in found_majors:
                found_majors.append((major, match.start()))
        
        # Sắp xếp theo vị trí xuất hiện
        found_majors.sort(key=lambda x: x[1])
        found_majors = [major for major, _ in found_majors[:2]]
        
        # Debug: In các ngành tìm thấy
        print(f"🔍 Found Majors: {found_majors}")
        
        if len(found_majors) < 2:
            return "Không tìm thấy đủ hai ngành để so sánh học phí. Vui lòng kiểm tra tên ngành trong câu hỏi."
        
        # Xác định khóa
        course = "Khóa 18-19"  # Mặc định
        if "khóa 17" in query_lower:
            course = "Khóa 17"
        elif "khóa 20-21" in query_lower or "khóa 20" in query_lower or "khóa 21" in query_lower:
            course = "Khóa 20-21"
        
        # Debug: In khóa được chọn
        print(f"🔍 Selected Course: {course}")
        
        # Lấy học phí
        fees = []
        for major in found_majors:
            found = False
            for block_name, block in self.training_program.items():
                if major in block["majors"]:
                    fee = block.get(course, block.get("Khóa 18-19"))
                    if fee is not None:
                        fees.append((major, course, fee))
                        found = True
                        break
            if not found:
                return f"Không tìm thấy thông tin học phí cho ngành {major} khóa {course}."
        
        # Debug: In học phí tìm thấy
        print(f"🔍 Fees: {fees}")
        
        if len(fees) < 2:
            return "Không tìm thấy thông tin học phí cho các ngành hoặc khóa yêu cầu."
        
        # So sánh học phí
        result = []
        fee1, fee2 = fees[0], fees[1]  # fee1: ngành A, fee2: ngành B
        result.append(f"Ngành {fee1[0]} {fee1[1]}: {fee1[2]:,} đồng/tín chỉ")
        result.append(f"Ngành {fee2[0]} {fee2[1]}: {fee2[2]:,} đồng/tín chỉ")
        
        diff = fee1[2] - fee2[2]
        if diff > 0:
            result.append(f"Kết quả: Học phí ngành {fee1[0]} {fee1[1]} lớn hơn ngành {fee2[0]} {fee2[1]} (chênh lệch {abs(diff):,} đồng/tín chỉ).")
        elif diff < 0:
            result.append(f"Kết quả: Học phí ngành {fee1[0]} {fee1[1]} nhỏ hơn ngành {fee2[0]} {fee2[1]} (chênh lệch {abs(diff):,} đồng/tín chỉ).")
        else:
            result.append(f"Kết quả: Học phí ngành {fee1[0]} {fee1[1]} bằng ngành {fee2[0]} {fee2[1]}.")
        
        return "\n".join(result)
    
    def compare_resident_nonresident(self, query, search_results):
        """So sánh điểm khác nhau giữa sinh viên nội trú và ngoại trú dựa trên vector_store"""
        query_lower = query.lower()
        
        # Tìm kiếm thông tin từ vector_store
        resident_keywords = ["nội trú", "ký túc xá", "ở trong trường"]
        nonresident_keywords = ["ngoại trú", "thuê trọ", "ở ngoài trường"]
        resident_info = []
        nonresident_info = []
        
        for result in search_results:
            result_text = result["text"].lower()
            if any(keyword in result_text for keyword in resident_keywords):
                resident_info.append(result["text"])
            if any(keyword in result_text for keyword in nonresident_keywords):
                nonresident_info.append(result["text"])

        # Tạo câu trả lời
        result = ["**So sánh sinh viên nội trú và ngoại trú**:"]
        
        if resident_info or nonresident_info:
            if resident_info:
                result.append("Sinh viên nội trú:")
                result.extend([f"- {info}" for info in resident_info[:3]])  # Tăng giới hạn lên 3 để có thêm thông tin
            if nonresident_info:
                result.append("Sinh viên ngoại trú:")
                result.extend([f"- {info}" for info in nonresident_info[:3]])
        else:
            result.append("🙁 Không tìm thấy thông tin về sinh viên nội trú hoặc ngoại trú trong sổ tay ICTU.")

        return "\n".join(result)

    def compare_generic(self, query, search_results, topic):
        """So sánh chung cho các chủ đề khác (ngoài học phí và nội trú/ngoại trú)"""
        query_lower = query.lower()
        result = [f"**So sánh về {topic}**:"]
        
        # Xác định từ khóa cho chủ đề
        topic_keywords = {
            "điều kiện nhập học": ["điều kiện nhập học", "yêu cầu nhập học", "điểm chuẩn", "xét tuyển"],
            "chương trình đào tạo": ["chương trình đào tạo", "tín chỉ", "môn học", "kỹ sư", "cử nhân"],
            "hoạt động ngoại khóa": ["hoạt động ngoại khóa", "điểm ngoại khóa", "sự kiện sinh viên"],
            # Thêm các chủ đề khác nếu cần
        }.get(topic, [])

        # Tìm kiếm thông tin liên quan từ search_results
        topic_info = []
        for result in search_results:
            result_text = result["text"].lower()
            if any(keyword in result_text for keyword in topic_keywords):
                topic_info.append(result["text"])

        if topic_info:
            # Tách thông tin theo các đối tượng so sánh (nếu có)
            entities = []
            for major in [m for block in self.training_program.values() for m in block["majors"]]:
                if major.lower() in query_lower:
                    entities.append(major)
            if not entities:
                entities = [keyword for keyword in topic_keywords if keyword in query_lower]

            if len(entities) >= 2:
                # So sánh giữa hai đối tượng cụ thể
                for entity in entities:
                    entity_info = [info for info in topic_info if entity.lower() in info.lower()]
                    if entity_info:
                        result.append(f"{entity}:")
                        result.extend([f"- {info}" for info in entity_info[:2]])
                    else:
                        result.append(f"{entity}: Không tìm thấy thông tin chi tiết.")
            else:
                # Liệt kê thông tin chung cho chủ đề
                result.append(f"Thông tin về {topic}:")
                result.extend([f"- {info}" for info in topic_info[:3]])
        else:
            result.append(f"🙁 Không tìm thấy thông tin về {topic} trong sổ tay ICTU.")

        return "\n".join(result)

    def extract_statement(self, query):
        """Trích xuất tuyên bố từ câu hỏi đúng/sai"""
        for keyword in ["đúng không", "có phải", "không phải", "có đúng", "có thật", "thật không"]:
            if keyword in query.lower():
                statement = query.lower().replace(keyword, "").strip(" ?")
                return statement
        return query

    def get_relevant_info(self, query):
        """Lấy thông tin liên quan từ training_program"""
        relevant_info = {}
        query_lower = query.lower()

        # Kiểm tra các từ khóa trong câu hỏi để lấy dữ liệu liên quan
        if "học phí" in query_lower:
            for block_name, block in self.training_program.items():
                for major in block["majors"]:
                    if major.lower() in query_lower:
                        relevant_info[major] = {
                            "Khóa 17": block["Khóa 17"],
                            "Khóa 18-19": block["Khóa 18-19"],
                            "Khóa 20-21": block["Khóa 20-21"]
                        }
        return relevant_info

    def verify_statement(self, statement, relevant_info, search_results):
        """Xác minh tuyên bố đúng hay sai"""
        statement_lower = statement.lower()

        # Kiểm tra thông tin từ training_program (relevant_info)
        for major, info in relevant_info.items():
            for course, fee in info.items():
                # Kiểm tra nếu tuyên bố chứa thông tin về học phí
                if major.lower() in statement_lower and course.lower() in statement_lower:
                    fee_str = f"{fee:,} đồng/tín chỉ"
                    if fee_str in statement_lower:
                        return True, f"Đúng, {statement}. (Nguồn: Dữ liệu chương trình đào tạo)"
                    else:
                        # Tìm số tiền trong tuyên bố để so sánh
                        import re
                        numbers = re.findall(r'\d+', statement_lower.replace(",", ""))
                        if numbers:
                            stated_fee = int(numbers[0].replace(",", ""))
                            if stated_fee == fee:
                                return True, f"Đúng, {statement}. (Nguồn: Dữ liệu chương trình đào tạo)"
                            else:
                                return False, f"Sai, {statement}. Thực tế: Ngành {major} {course}: {fee:,} đồng/tín chỉ. (Nguồn: Dữ liệu chương trình đào tạo)"
        
        # Kiểm tra thông tin từ vector_store (search_results)
        for result in search_results:
            result_text = result["text"].lower()
            if statement_lower in result_text:
                return True, f"Đúng, {statement}. (Nguồn: Sổ tay ICTU)"
        
        # Nếu không tìm thấy thông tin khớp, trả về Sai
        return False, f"Sai, không tìm thấy thông tin xác nhận cho {statement}. (Nguồn: Dữ liệu chương trình đào tạo)"

    def search_data(self, query, top_k=50):
        try:
            query_embedding = self.embedding_handler.generate_embeddings([query])[0]
            search_results = self.vector_store.search(query_embedding, top_k=top_k)
            
            if not search_results:
                return None, "🙁 Không tìm thấy thông tin liên quan trong sổ tay ICTU."

            # Đếm số lượng câu hỏi và reset max_priority nếu cần
            self.query_count += 1
            if self.query_count > 10:  # Reset sau 10 câu hỏi để tránh độ ưu tiên tăng quá cao
                self.max_priority = 0
                self.query_count = 0

            query_lower = query.lower()
            relevant_keywords = []
            for topic, keywords in self.keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    relevant_keywords.extend(keywords)
                    break

            # Thêm từ khóa và tăng độ ưu tiên cho các chủ đề
            if "khoa" in query_lower or "ngành học" in query_lower or "liệt kê" in query_lower:
                relevant_keywords.extend([
                    "khoa", "ngành học", "cơ cấu tổ chức", "khoa học cơ bản", "công nghệ thông tin",
                    "công nghệ điện tử và truyền thông", "hệ thống thông tin kinh tế", "công nghệ tự động hóa",
                    "truyền thông đa phương tiện", "thương mại điện tử"
                ])
                self.priority_weights["khoa"] = max(self.priority_weights["khoa"] + 2, self.max_priority)
                self.priority_weights["ngành học"] = max(self.priority_weights["ngành học"] + 2, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["khoa"], self.priority_weights["ngành học"])

            # Thêm từ khóa cho các ngành học từ bảng chương trình đào tạo
            majors = []
            for block in self.training_program.values():
                majors.extend(block["majors"])
            for major in majors:
                if major.lower() in query_lower:
                    relevant_keywords.append(major)

            # Thêm từ khóa cho học phí và khóa học
            if "học phí" in query_lower or "mỗi tín chỉ" in query_lower or "bao nhiêu tiền" in query_lower:
                relevant_keywords.extend(["học phí", "đồng/tín chỉ", "mỗi tín chỉ", "khóa 17", "khóa 18-19", "khóa 20-21"])
                self.priority_weights["học phí"] = max(self.priority_weights["học phí"] + 2, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["học phí"])
            if "khóa 17" in query_lower:
                relevant_keywords.append("khóa 17")
            if "khóa 18" in query_lower or "khóa 19" in query_lower or "khóa 18-19" in query_lower:
                relevant_keywords.extend(["khóa 18", "khóa 19", "khóa 18-19"])
            if "khóa 20" in query_lower or "khóa 21" in query_lower or "khóa 20-21" in query_lower:
                relevant_keywords.extend(["khóa 20", "khóa 21", "khóa 20-21"])

            # Mở rộng từ khóa cho các chủ đề khác
            if "điều 3" in query_lower:
                relevant_keywords.extend(["điều 3", "chương trình đào tạo", "kỹ sư", "trình độ bậc 7", "cử nhân"])
            if "cử nhân" in query_lower:
                relevant_keywords.extend(["cử nhân", "chương trình đào tạo", "đại học"])
            if "kỹ sư" in query_lower:
                relevant_keywords.extend(["kỹ sư", "trình độ bậc 7", "chương trình đào tạo", "tín chỉ", "khối lượng học tập"])
            if "xếp loại học lực" in query_lower:
                relevant_keywords.extend(["xếp loại học lực", "xuất sắc", "giỏi", "khá", "trung bình", "yếu", "kém", "điểm trung bình tích lũy"])
                self.priority_weights["xếp loại học lực"] = max(self.priority_weights["xếp loại học lực"] + 1, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["xếp loại học lực"])
            if "xếp hạng học lực" in query_lower:
                relevant_keywords.extend(["xếp hạng học lực", "hạng bình thường", "hạng yếu", "điểm trung bình tích lũy"])
                self.priority_weights["xếp hạng học lực"] = max(self.priority_weights["xếp hạng học lực"] + 1, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["xếp hạng học lực"])
            if "đăng nhập" in query_lower:
                relevant_keywords.extend(["đăng nhập", "mã sinh viên", "mật khẩu", "ngày tháng năm sinh", "viết hoa", "hệ thống"])
                self.priority_weights["đăng nhập"] = max(self.priority_weights["đăng nhập"] + 1, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["đăng nhập"])
            if "đổi mật khẩu" in query_lower:
                relevant_keywords.extend(["đổi mật khẩu", "thay đổi mật khẩu", "cập nhật mật khẩu", "quản lý tài khoản", "phòng đào tạo", "hệ thống"])
                self.priority_weights["đổi mật khẩu"] = max(self.priority_weights["đổi mật khẩu"] + 1, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["đổi mật khẩu"])
            if "đăng ký học" in query_lower:
                relevant_keywords.extend(["đăng ký học", "học phần", "hệ thống", "đăng ký tín chỉ"])
                self.priority_weights["đăng ký học"] = max(self.priority_weights["đăng ký học"] + 1, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["đăng ký học"])
            if "thanh toán học phí" in query_lower or "nộp tiền trước" in query_lower:
                relevant_keywords.extend(["thanh toán học phí", "online", "đăng ký học kỳ tới", "nộp tiền trước", "chức năng thanh toán", "hệ thống"])
                self.priority_weights["thanh toán học phí"] = max(self.priority_weights["thanh toán học phí"] + 1, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["thanh toán học phí"])
            if "hoạt động ngoại khóa" in query_lower:
                relevant_keywords.extend(["hoạt động ngoại khóa", "điểm ngoại khóa", "tham gia hoạt động", "đánh giá hoạt động"])
                self.priority_weights["hoạt động ngoại khóa"] = max(self.priority_weights["hoạt động ngoại khóa"] + 1, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["hoạt động ngoại khóa"])
            if "xem lịch thi" in query_lower:
                relevant_keywords.extend(["xem lịch thi", "học kỳ", "đợt học"])
                self.priority_weights["xem lịch thi"] = max(self.priority_weights["xem lịch thi"] + 1, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["xem lịch thi"])
            if "tra cứu điểm" in query_lower:
                relevant_keywords.extend(["tra cứu điểm", "học kỳ"])
                self.priority_weights["tra cứu điểm"] = max(self.priority_weights["tra cứu điểm"] + 1, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["tra cứu điểm"])
            if "hủy học phần" in query_lower:
                relevant_keywords.extend(["hủy học phần", "đăng ký nhầm"])
                self.priority_weights["hủy học phần"] = max(self.priority_weights["hủy học phần"] + 1, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["hủy học phần"])
            if "gửi tin nhắn" in query_lower:
                relevant_keywords.extend(["gửi tin nhắn", "người quản trị"])
                self.priority_weights["gửi tin nhắn"] = max(self.priority_weights["gửi tin nhắn"] + 1, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["gửi tin nhắn"])
            if "tổng số tín chỉ" in query_lower or "khối lượng học tập tối thiểu" in query_lower:
                relevant_keywords.extend(["tổng số tín chỉ", "khối lượng học tập tối thiểu", "chương trình đào tạo", "kỹ sư", "trình độ bậc 7", "cử nhân", "điều 3"])
                self.priority_weights["tổng số tín chỉ"] = max(self.priority_weights["tổng số tín chỉ"] + 1, self.max_priority)
                self.max_priority = max(self.max_priority, self.priority_weights["tổng số tín chỉ"])

            filtered_results = []
            for result in search_results:
                result_text = result.payload["text"].lower()
                keyword_score = sum(1 for keyword in relevant_keywords if keyword in result_text)
                
                # Áp dụng độ ưu tiên theo chủ đề
                if "khoa" in query_lower and "khoa" in result_text:
                    keyword_score += self.priority_weights["khoa"]
                if "ngành học" in query_lower and any(major.lower() in result_text for major in majors):
                    keyword_score += self.priority_weights["ngành học"]
                if "liệt kê" in query_lower and "khoa" in result_text:
                    keyword_score += self.priority_weights["khoa"]
                if "cơ cấu tổ chức" in result_text:
                    keyword_score += 10  # Ưu tiên cao cho các đoạn văn chứa danh sách khoa
                if "học phí" in query_lower and "đồng/tín chỉ" in result_text:
                    keyword_score += self.priority_weights["học phí"]
                if "xếp loại học lực" in query_lower and "điểm trung bình tích lũy" in result_text:
                    keyword_score += self.priority_weights["xếp loại học lực"]
                if "xếp hạng học lực" in query_lower and "điểm trung bình tích lũy" in result_text:
                    keyword_score += self.priority_weights["xếp hạng học lực"]
                if "đăng nhập" in query_lower and "hệ thống" in result_text:
                    keyword_score += self.priority_weights["đăng nhập"]
                if "đổi mật khẩu" in query_lower and "quản lý tài khoản" in result_text:
                    keyword_score += self.priority_weights["đổi mật khẩu"]
                if "đăng ký học" in query_lower and "hệ thống" in result_text:
                    keyword_score += self.priority_weights["đăng ký học"]
                if "thanh toán học phí" in query_lower and "hệ thống" in result_text:
                    keyword_score += self.priority_weights["thanh toán học phí"]
                if "tổng số tín chỉ" in result_text or "khối lượng học tập tối thiểu" in result_text:
                    keyword_score += self.priority_weights["tổng số tín chỉ"]
                if "trình độ bậc 7 kỹ sư" in result_text and "tín chỉ" in result_text:
                    keyword_score += 3

                combined_score = result.score + (keyword_score * 0.7)
                if combined_score >= 0.1:  # Giảm ngưỡng từ 0.3 xuống 0.1
                    filtered_results.append({
                        "id": result.id,
                        "text": result.payload["text"],
                        "metadata": result.payload["metadata"],
                        "score": combined_score
                    })

            filtered_results = sorted(filtered_results, key=lambda x: x["score"], reverse=True)
            if not filtered_results:
                return None, "🙁 Không tìm thấy thông tin liên quan trong sổ tay ICTU (score quá thấp)."
            return filtered_results[:top_k], None
        except Exception as e:
            return None, f"❌ Lỗi khi tìm kiếm dữ liệu: {str(e)}"

    def create_task(self, query, search_results):
        query_lower = query.lower()
        context = "\n\n".join([result["text"] for result in search_results]) if search_results else ""
        context = context.replace("Tải về:", "").replace(".xlsm", "")
        for page in ["133", "134", "135", "136", "137", "138", "139", "140", "141", "142"]:
            context = context.replace(page, "")

        # Thu thập từ khóa liên quan
        relevant_keywords = []
        for topic, keywords in self.keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_keywords.extend(keywords)
                break

        majors = [m for block in self.training_program.values() for m in block["majors"]]
        if "khoa" in query_lower or "ngành học" in query_lower or "liệt kê" in query_lower:
            relevant_keywords.extend([
                "khoa", "ngành học", "cơ cấu tổ chức", "khoa học cơ bản", "công nghệ thông tin",
                "công nghệ điện tử và truyền thông", "hệ thống thông tin kinh tế", "công nghệ tự động hóa",
                "truyền thông đa phương tiện", "thương mại điện tử"
            ])
        for major in majors:
            if major.lower() in query_lower:
                relevant_keywords.append(major)
        if "học phí" in query_lower or "mỗi tín chỉ" in query_lower or "bao nhiêu tiền" in query_lower:
            relevant_keywords.extend(["học phí", "đồng/tín chỉ", "mỗi tín chỉ", "khóa 17", "khóa 18-19", "khóa 20-21"])
        if "khóa 17" in query_lower:
            relevant_keywords.append("khóa 17")
        if "khóa 18" in query_lower or "khóa 19" in query_lower or "khóa 18-19" in query_lower:
            relevant_keywords.extend(["khóa 18", "khóa 19", "khóa 18-19"])
        if "khóa 20" in query_lower or "khóa 21" in query_lower or "khóa 20-21" in query_lower:
            relevant_keywords.extend(["khóa 20", "khóa 21", "khóa 20-21"])
        if "nội trú" in query_lower or "ngoại trú" in query_lower:
            relevant_keywords.extend(["nội trú", "ngoại trú", "ký túc xá", "thuê trọ", "ở trong trường", "ở ngoài trường"])
        if "điều kiện nhập học" in query_lower:
            relevant_keywords.extend(["điều kiện nhập học", "yêu cầu nhập học", "điểm chuẩn", "xét tuyển"])
        if "chương trình đào tạo" in query_lower:
            relevant_keywords.extend(["chương trình đào tạo", "tín chỉ", "môn học", "kỹ sư", "cử nhân"])
        if "hoạt động ngoại khóa" in query_lower:
            relevant_keywords.extend(["hoạt động ngoại khóa", "điểm ngoại khóa", "sự kiện sinh viên"])

        def has_relevant_content(keywords, context_lower, search_results):
            for keyword in keywords:
                if keyword in context_lower:
                    return True
            for result in search_results:
                result_text = result["text"].lower()
                if "khoa" in query_lower and "khoa" in result_text:
                    return True
                if "ngành học" in query_lower and any(major.lower() in result_text for major in majors):
                    return True
                if "điểm trung bình tích lũy" in result_text and ("xếp loại học lực" in query_lower or "xếp hạng học lực" in query_lower):
                    return True
                if "hệ thống" in result_text and ("đăng nhập" in query_lower or "đổi mật khẩu" in query_lower or "đăng ký học" in query_lower or "thanh toán học phí" in query_lower):
                    return True
                if "quản lý tài khoản" in result_text and "đổi mật khẩu" in query_lower:
                    return True
                if any(major.lower() in result_text for major in majors) and "đồng/tín chỉ" in result_text:
                    return True
                if ("nội trú" in query_lower or "ngoại trú" in query_lower) and ("ký túc xá" in result_text or "thuê trọ" in result_text):
                    return True
                if "điều kiện nhập học" in query_lower and any(k in result_text for k in ["điều kiện nhập học", "yêu cầu nhập học", "điểm chuẩn"]):
                    return True
                if "chương trình đào tạo" in query_lower and any(k in result_text for k in ["chương trình đào tạo", "tín chỉ", "môn học"]):
                    return True
                if "hoạt động ngoại khóa" in query_lower and any(k in result_text for k in ["hoạt động ngoại khóa", "điểm ngoại khóa"]):
                    return True
            return False

        # Xử lý câu hỏi so sánh
        if self.is_comparison_question(query):
            if "học phí" in query_lower:
                return Task(
                    description=f"""
                        So sánh học phí dựa trên câu hỏi: '{query}'.
                        ⚠️ Yêu cầu:
                        - Trích xuất học phí từ dữ liệu chương trình đào tạo (self.training_program) cho các ngành hoặc khóa được đề cập trong câu hỏi.
                        - Xác định ngành A (ngành được đề cập ĐẦU TIÊN trong câu hỏi) và ngành B (ngành được đề cập THỨ HAI trong câu hỏi) dựa trên thứ tự xuất hiện trong câu hỏi.
                        - Nếu câu hỏi nêu cụ thể các ngành (ví dụ: Công nghệ thông tin, Thương mại điện tử), liệt kê học phí của ngành A trước, sau đó ngành B, theo khóa được chỉ định (hoặc mặc định Khóa 18-19 nếu không có khóa).
                        - Nếu câu hỏi nêu các khóa (ví dụ: Khóa 17, Khóa 20-21), so sánh học phí của ngành được đề cập qua các khóa, nhưng vẫn ưu tiên thứ tự ngành A trước.
                        - So sánh và nêu rõ điểm khác nhau, luôn lấy ngành A làm tham chiếu:
                        - Nếu học phí ngành A lớn hơn ngành B, trả về: "Học phí ngành A lớn hơn ngành B (chênh lệch X đồng/tín chỉ)."
                        - Nếu học phí ngành A nhỏ hơn ngành B, trả về: "Học phí ngành A nhỏ hơn ngành B (chênh lệch X đồng/tín chỉ)."
                        - Nếu học phí bằng nhau, trả về: "Học phí ngành A bằng ngành B."
                        - Định dạng câu trả lời rõ ràng, ví dụ:
                        - Ngành Công nghệ thông tin Khóa 20-21: 467,700 đồng/tín chỉ
                        - Ngành Thương mại điện tử Khóa 20-21: 403,200 đồng/tín chỉ
                        - Điểm khác nhau: Học phí ngành Công nghệ thông tin lớn hơn ngành Thương mại điện tử (chênh lệch 64,500 đồng/tín chỉ).
                        - Nếu không tìm thấy thông tin, trả về: "Không tìm thấy thông tin học phí cho các ngành hoặc khóa yêu cầu."
                        - KHÔNG suy đoán, thêm thông tin ngoài, hoặc thay đổi thứ tự ngành.
                        - PHẢI trả về CHÍNH XÁC kết quả từ self.compare_tuition_fees(query) mà KHÔNG được diễn giải lại, chỉnh sửa, hoặc thay đổi thứ tự.
                        - Kết quả đã được tính toán sẵn trong expected_output, chỉ cần sao chép nguyên văn.
                        - Đảm bảo thứ tự: ngành A (đầu tiên trong câu hỏi) luôn được liệt kê và so sánh trước ngành B.
                        Chỉ trả lời bằng tiếng Việt.
                    """,
                    agent=self.agent,
                    expected_output=self.compare_tuition_fees(query)
                )
            elif "nội trú" in query_lower or "ngoại trú" in query_lower:
                return Task(
                    description=f"""
                        So sánh sinh viên nội trú và ngoại trú dựa trên câu hỏi: '{query}' từ nội dung sau:
                        {context}

                        ⚠️ Yêu cầu:
                        - Trích xuất thông tin về sinh viên nội trú (liên quan đến "nội trú", "ký túc xá", "ở trong trường") và ngoại trú (liên quan đến "ngoại trú", "thuê trọ", "ở ngoài trường") từ nội dung cung cấp.
                        - Liệt kê các đặc điểm của nội trú và ngoại trú, nêu rõ điểm khác nhau (ví dụ: chi phí, điều kiện sinh hoạt, quyền lợi, hạn chế).
                        - Định dạng câu trả lời rõ ràng, ví dụ:
                        - Sinh viên nội trú:
                            - [Thông tin 1, ví dụ: Chi phí thấp hơn, 500,000-1,000,000 đồng/tháng]
                            - [Thông tin 2, ví dụ: Gần trường, tiện di chuyển]
                        - Sinh viên ngoại trú:
                            - [Thông tin 1, ví dụ: Chi phí cao hơn, 1,500,000-3,000,000 đồng/tháng]
                            - [Thông tin 2, ví dụ: Tự do thời gian]
                        - Điểm khác nhau: [Tóm tắt sự khác biệt, ví dụ: Nội trú chi phí thấp hơn nhưng có giờ giới nghiêm, ngoại trú tự do hơn nhưng chi phí cao].
                        - Chỉ trích xuất tối đa 3 thông tin cho mỗi loại để câu trả lời ngắn gọn.
                        - Nếu không tìm thấy thông tin, trả lời: "Không tìm thấy thông tin về sinh viên nội trú hoặc ngoại trú trong sổ tay ICTU."
                        - Không thêm thông tin ngoài nội dung cung cấp.
                        Chỉ trả lời bằng tiếng Việt.
                    """,
                    agent=self.agent,
                    expected_output=self.compare_resident_nonresident(query, search_results)
                )
            elif "điều kiện nhập học" in query_lower:
                return Task(
                    description=f"""
                        So sánh điều kiện nhập học dựa trên câu hỏi: '{query}' từ nội dung sau:
                        {context}

                        ⚠️ Yêu cầu:
                        - Trích xuất thông tin về điều kiện nhập học (liên quan đến "điều kiện nhập học", "yêu cầu nhập học", "điểm chuẩn", "xét tuyển") từ nội dung cung cấp.
                        - Nếu câu hỏi nêu cụ thể các ngành (ví dụ: Công nghệ thông tin, Thương mại điện tử), liệt kê điều kiện nhập học của từng ngành và nêu điểm khác nhau (ví dụ: điểm chuẩn, tổ hợp môn, phương thức xét tuyển).
                        - Nếu không có ngành cụ thể, cung cấp thông tin chung về điều kiện nhập học và so sánh các khía cạnh (ví dụ: xét tuyển học bạ vs thi THPT).
                        - Định dạng câu trả lời rõ ràng, ví dụ:
                        - Ngành X: [Điều kiện, ví dụ: Điểm chuẩn 25.5, tổ hợp A00]
                        - Ngành Y: [Điều kiện, ví dụ: Điểm chuẩn 24.0, tổ hợp A00, D01]
                        - Điểm khác nhau: [Tóm tắt, ví dụ: Ngành X có điểm chuẩn cao hơn, không chấp nhận D01].
                        - Chỉ trích xuất tối đa 3 thông tin cho mỗi ngành hoặc chủ đề để ngắn gọn.
                        - Nếu không tìm thấy thông tin, trả lời: "Không tìm thấy thông tin về điều kiện nhập học trong sổ tay ICTU."
                        - Không thêm thông tin ngoài nội dung cung cấp.
                        Chỉ trả lời bằng tiếng Việt.
                    """,
                    agent=self.agent,
                    expected_output=self.compare_generic(query, search_results, "điều kiện nhập học")
                )
            elif "chương trình đào tạo" in query_lower:
                return Task(
                    description=f"""
                        So sánh chương trình đào tạo dựa trên câu hỏi: '{query}' từ nội dung sau:
                        {context}

                        ⚠️ Yêu cầu:
                        - Trích xuất thông tin về chương trình đào tạo (liên quan đến "chương trình đào tạo", "tín chỉ", "môn học", "kỹ sư", "cử nhân") từ nội dung cung cấp.
                        - Nếu câu hỏi nêu cụ thể các ngành (ví dụ: Kỹ thuật phần mềm, Khoa học máy tính), liệt kê đặc điểm chương trình đào tạo của từng ngành và nêu điểm khác nhau (ví dụ: số tín chỉ, môn học chính, định hướng nghề nghiệp).
                        - Nếu không có ngành cụ thể, cung cấp thông tin chung về chương trình đào tạo và so sánh các khía cạnh (ví dụ: kỹ sư vs cử nhân).
                        - Định dạng câu trả lời rõ ràng, ví dụ:
                        - Ngành X: [Tổng 150 tín chỉ, tập trung vào...]
                        - Ngành Y: [Tổng 145 tín chỉ, bao gồm môn...]
                        - Điểm khác nhau: [Tóm tắt, ví dụ: Ngành X có nhiều tín chỉ hơn, tập trung thực hành].
                        - Chỉ trích xuất tối đa 3 thông tin cho mỗi ngành hoặc chủ đề để ngắn gọn.
                        - Nếu không tìm thấy thông tin, trả lời: "Không tìm thấy thông tin về chương trình đào tạo trong sổ tay ICTU."
                        - Không thêm thông tin ngoài nội dung cung cấp.
                        Chỉ trả lời bằng tiếng Việt.
                    """,
                    agent=self.agent,
                    expected_output=self.compare_generic(query, search_results, "chương trình đào tạo")
                )
            elif "hoạt động ngoại khóa" in query_lower:
                return Task(
                    description=f"""
                        So sánh hoạt động ngoại khóa dựa trên câu hỏi: '{query}' từ nội dung sau:
                        {context}

                        ⚠️ Yêu cầu:
                        - Trích xuất thông tin về hoạt động ngoại khóa (liên quan đến "hoạt động ngoại khóa", "điểm ngoại khóa", "sự kiện sinh viên") từ nội dung cung cấp.
                        - Nếu câu hỏi nêu cụ thể các ngành hoặc nhóm (ví dụ: Công nghệ thông tin, Thương mại điện tử), liệt kê các hoạt động ngoại khóa liên quan đến từng ngành hoặc nhóm và nêu điểm khác nhau (ví dụ: loại sự kiện, tần suất).
                        - Nếu không có ngành cụ thể, cung cấp thông tin chung về hoạt động ngoại khóa và so sánh các khía cạnh (ví dụ: hoạt động của khoa vs toàn trường).
                        - Định dạng câu trả lời rõ ràng, ví dụ:
                        - Ngành X: [Hoạt động 1, ví dụ: Cuộc thi lập trình]
                        - Ngành Y: [Hoạt động 2, ví dụ: Hội thảo thương mại]
                        - Điểm khác nhau: [Tóm tắt, ví dụ: Ngành X tập trung kỹ thuật, Ngành Y thiên về kinh doanh].
                        - Chỉ trích xuất tối đa 3 thông tin cho mỗi ngành hoặc chủ đề để ngắn gọn.
                        - Nếu không tìm thấy thông tin, trả lời: "Không tìm thấy thông tin về hoạt động ngoại khóa trong sổ tay ICTU."
                        - Không thêm thông tin ngoài nội dung cung cấp.
                        Chỉ trả lời bằng tiếng Việt.
                    """,
                    agent=self.agent,
                    expected_output=self.compare_generic(query, search_results, "hoạt động ngoại khóa")
                )

        # Xử lý câu hỏi học phí đơn lẻ
        if any(major.lower() in query_lower for major in majors) and ("học phí" in query_lower or "mỗi tín chỉ" in query_lower or "bao nhiêu tiền" in query_lower):
            for major in majors:
                if major.lower() in query_lower:
                    selected_major = major
                    break
            for block_name, block in self.training_program.items():
                if selected_major in block["majors"]:
                    if "khóa 17" in query_lower:
                        tuition_fee = block["Khóa 17"]
                        return Task(
                            description=f"""
                                Trả lời học phí của ngành {selected_major} cho Khóa 17.
                                ⚠️ Yêu cầu:
                                - Trích xuất học phí từ dữ liệu chương trình đào tạo.
                                - Định dạng câu trả lời: "Ngành {selected_major} Khóa 17: {tuition_fee:,} đồng/tín chỉ."
                                - Chỉ sử dụng dữ liệu từ self.training_program.
                                Chỉ trả lời bằng tiếng Việt.
                            """,
                            agent=self.agent,
                            expected_output=f"Ngành {selected_major} Khóa 17: {tuition_fee:,} đồng/tín chỉ."
                        )
                    elif "khóa 18-19" in query_lower or "khóa 18" in query_lower or "khóa 19" in query_lower:
                        tuition_fee = block["Khóa 18-19"]
                        return Task(
                            description=f"""
                                Trả lời học phí của ngành {selected_major} cho Khóa 18-19.
                                ⚠️ Yêu cầu:
                                - Trích xuất học phí từ dữ liệu chương trình đào tạo.
                                - Định dạng câu trả lời: "Ngành {selected_major} Khóa 18-19: {tuition_fee:,} đồng/tín chỉ."
                                - Chỉ sử dụng dữ liệu từ self.training_program.
                                Chỉ trả lời bằng tiếng Việt.
                            """,
                            agent=self.agent,
                            expected_output=f"Ngành {selected_major} Khóa 18-19: {tuition_fee:,} đồng/tín chỉ."
                        )
                    elif "khóa 20-21" in query_lower or "khóa 20" in query_lower or "khóa 21" in query_lower:
                        tuition_fee = block["Khóa 20-21"]
                        return Task(
                            description=f"""
                                Trả lời học phí của ngành {selected_major} cho Khóa 20-21.
                                ⚠️ Yêu cầu:
                                - Trích xuất học phí từ dữ liệu chương trình đào tạo.
                                - Định dạng câu trả lời: "Ngành {selected_major} Khóa 20-21: {tuition_fee:,} đồng/tín chỉ."
                                - Chỉ sử dụng dữ liệu từ self.training_program.
                                Chỉ trả lời bằng tiếng Việt.
                            """,
                            agent=self.agent,
                            expected_output=f"Ngành {selected_major} Khóa 20-21: {tuition_fee:,} đồng/tín chỉ."
                        )
                    else:
                        tuition_fee = block["Khóa 18-19"]
                        return Task(
                            description=f"""
                                Trả lời học phí của ngành {selected_major} cho khóa không xác định (mặc định Khóa 18-19).
                                ⚠️ Yêu cầu:
                                - Trích xuất học phí từ dữ liệu chương trình đào tạo.
                                - Định dạng câu trả lời: "Ngành {selected_major}: {tuition_fee:,} đồng/tín chỉ (theo đơn giá Khóa 18-19 cho các khóa không xác định)."
                                - Chỉ sử dụng dữ liệu từ self.training_program.
                                Chỉ trả lời bằng tiếng Việt.
                            """,
                            agent=self.agent,
                            expected_output=f"Ngành {selected_major}: {tuition_fee:,} đồng/tín chỉ (theo đơn giá Khóa 18-19 cho các khóa không xác định)."
                        )

        # Xử lý câu hỏi liệt kê các ngành học
        if "liệt kê các ngành học" in query_lower or "danh sách ngành học" in query_lower:
            all_majors = []
            for block_name, block in self.training_program.items():
                all_majors.extend(block["majors"])
            majors_list = "\n".join([f"- {major}" for major in all_majors])
            return Task(
                description=f"""
                    Liệt kê các ngành học từ chương trình đào tạo đại trà.
                    ⚠️ Yêu cầu:
                    - Trích xuất danh sách ngành học từ dữ liệu chương trình đào tạo (self.training_program).
                    - Định dạng câu trả lời:
                    - Danh sách các ngành học tại ICTU:
                        - [Ngành 1]
                        - [Ngành 2]
                        ...
                    - Không thêm thông tin ngoài dữ liệu cung cấp.
                    Chỉ trả lời bằng tiếng Việt.
                """,
                agent=self.agent,
                expected_output=f"Danh sách các ngành học tại ICTU:\n{majors_list}"
            )

        # Xử lý các câu hỏi khác
        task_description = f"""
            Trích xuất thông tin chính xác để trả lời câu hỏi: '{query}' từ nội dung sau:
            {context}

            ⚠️ Yêu cầu:
            - Chỉ trích xuất đúng nội dung liên quan nhất từ đoạn văn trên.
            - Ưu tiên các đoạn văn chứa các từ khóa: {', '.join(relevant_keywords)}.
            - Nếu câu hỏi hỏi về "liệt kê các khoa" hoặc "các khoa tại ICTU", trích xuất danh sách các khoa từ các đoạn văn chứa từ khóa "khoa" và liệt kê đầy đủ.
            - Nếu câu hỏi hỏi về "ngành học" hoặc "liệt kê các ngành học", trích xuất thông tin về các ngành học nếu có. Nếu không, suy luận từ tên khoa (ví dụ: Khoa Công nghệ Thông tin có thể có ngành Công nghệ Thông tin) và ghi rõ là suy luận.
            - Nếu câu hỏi hỏi về "xếp loại học lực", trích xuất các đoạn chứa "xếp loại học lực", "xuất sắc", "giỏi", "khá", "trung bình", "yếu", "kém", hoặc "điểm trung bình tích lũy".
            - Nếu câu hỏi hỏi về "xếp hạng học lực", trích xuất các đoạn chứa "xếp hạng học lực", "hạng bình thường", "hạng yếu", hoặc "điểm trung bình tích lũy".
            - Nếu câu hỏi hỏi về "đăng nhập", trích xuất các đoạn chứa "đăng nhập", "mã sinh viên", "mật khẩu", "ngày tháng năm sinh", "viết hoa", "hệ thống".
            - Nếu câu hỏi hỏi về "đổi mật khẩu", trích xuất các đoạn chứa "đổi mật khẩu", "thay đổi mật khẩu", "cập nhật mật khẩu", "quản lý tài khoản", "phòng đào tạo", "hệ thống".
            - Nếu câu hỏi hỏi về "đăng ký học", trích xuất các đoạn chứa "đăng ký học", "học phần", "hệ thống", "đăng ký tín chỉ".
            - Nếu câu hỏi hỏi về "thanh toán học phí" hoặc "nộp tiền trước", trích xuất các đoạn chứa "thanh toán học phí", "online", "đăng ký học kỳ tới", "nộp tiền trước", "chức năng thanh toán", "hệ thống".
            - Nếu câu hỏi hỏi về "nội trú" hoặc "ngoại trú", trích xuất các đoạn văn chứa "nội trú", "ngoại trú", "ký túc xá", "thuê trọ", "ở trong trường", "ở ngoài trường".
            - Nếu câu hỏi hỏi về "điều kiện nhập học", trích xuất các đoạn văn chứa "điều kiện nhập học", "yêu cầu nhập học", "điểm chuẩn", "xét tuyển".
            - Nếu câu hỏi hỏi về "chương trình đào tạo", trích xuất các đoạn văn chứa "chương trình đào tạo", "tín chỉ", "môn học", "kỹ sư", "cử nhân".
            - Nếu câu hỏi hỏi về "hoạt động ngoại khóa", trích xuất các đoạn văn chứa "hoạt động ngoại khóa", "điểm ngoại khóa", "sự kiện sinh viên".
            - Nếu câu hỏi không thuộc các chủ đề trên, trích xuất các đoạn văn liên quan nhất dựa trên từ khóa và ngữ nghĩa của câu hỏi.
            - KHÔNG được viết lại, diễn giải lại hay sáng tạo thêm thông tin ngoài nội dung cung cấp.
            - Nếu trong đoạn văn có câu trả lời trực tiếp, trích nguyên văn.
            - Nếu không tìm thấy thông tin phù hợp, trả lời: **"Không tìm thấy thông tin về chủ đề này trong sổ tay ICTU."**

            Chỉ trả lời bằng tiếng Việt.
        """
        task = Task(
            description=task_description,
            agent=self.agent,
            expected_output="Trích xuất nguyên văn thông tin liên quan từ đoạn văn, không tóm tắt, không diễn giải."
        )
        return task
    
    def run(self, query, top_k=5):
        # Kiểm tra nếu câu hỏi là dạng đúng/sai
        if self.is_true_false_question(query):
            statement = self.extract_statement(query)
            search_results, error = self.search_data(query, top_k)
            if error:
                print(error)
                return error
            relevant_info = self.get_relevant_info(query)
            is_true, explanation = self.verify_statement(statement, relevant_info, search_results)
            self.chat_history.append((query, explanation))
            return explanation

        # Xử lý các câu hỏi thông tin bình thường
        search_results, error = self.search_data(query, top_k)
        print("\n💬 Kết quả tìm kiếm từ Qdrant:")
        if error:
            print(error)
            return error
        else:
            for i, result in enumerate(search_results):
                print(f"\nKết quả {i + 1} (Score: {result['score']:.4f}):")
                print(f"ID: {result['id']}")
                print("Text (toàn bộ nội dung):")
                print(result['text'])
                print(f"Metadata: {result['metadata']}")

        print("\n📝 Diễn giải và trả lời câu hỏi bằng LLM...")
        try:
            task = self.create_task(query, search_results)
            # Debug: In expected_output
            print("\n🔍 Expected Output từ compare_tuition_fees:")
            print(task.expected_output)
            crew = Crew(agents=[self.agent], tasks=[task], verbose=2)
            result = crew.kickoff()
            # Lấy ID từ kết quả có nội dung khớp với câu trả lời
            if search_results:
                relevant_result = None
                for res in search_results:
                    if result.strip() in res['text'].strip():
                        relevant_result = res
                        break
                if not relevant_result:
                    for res in search_results:
                        if "Đoàn thanh niên" in res['text'] and "Nhà điều hành C1" in res['text']:
                            relevant_result = res
                            break
                if not relevant_result:
                    relevant_result = search_results[0]
                result_with_id = result
            else:
                result_with_id = result
            self.chat_history.append((query, result_with_id))
            return result_with_id
        except Exception as e:
            error_msg = f"❌ Lỗi khi diễn giải dữ liệu: {str(e)}"
            print(error_msg)
            return error_msg


if __name__ == "__main__":
    print("🚀 Khởi động ChatBotAgent...")
    try:
        bot = ChatBotAgent()
        print("👋 Chào bạn! Tôi là trợ lý ảo hỗ trợ sinh viên ICTU.")
        print("Hãy nhập câu hỏi của bạn (nhập 'exit' để thoát):")

        while True:
            # Nhận câu hỏi từ người dùng qua terminal
            query = input("\n📝 Câu hỏi của bạn: ").strip()

            # Kiểm tra điều kiện thoát
            if query.lower() == "exit":
                print("\n👋 Tạm biệt! Hẹn gặp lại bạn.")
                break

            # Kiểm tra nếu câu hỏi rỗng
            if not query:
                print("⚠️ Vui lòng nhập câu hỏi!")
                continue

            # Xử lý câu hỏi và in phản hồi
            print(f"\n📝 Đang xử lý câu hỏi: {query}")
            response = bot.run(query, top_k=12)
            print("\n💬 Phản hồi từ trợ lý:")
            print(response)
            print("\n" + "="*50)

    except Exception as e:
        print(f"❌ Lỗi khi khởi động hoặc chạy agent: {e}")
    print("🎉 Hoàn tất!")