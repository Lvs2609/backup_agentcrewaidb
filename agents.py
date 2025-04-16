from crewai import Agent, Task, Crew
from embeddings import EmbeddingHandler
from vector_store import VectorStore
from llm_config import get_llm
from keywords import KEYWORDS

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

        self.agent = Agent(
            role="ICTU Handbook Assistant",
            goal="Truy xuất và diễn giải thông tin từ sổ tay ICTU để trả lời câu hỏi của sinh viên",
            backstory="Tôi là trợ lý ảo hỗ trợ sinh viên ICTU, truy xuất thông tin từ sổ tay sinh viên và cung cấp câu trả lời chính xác, ngắn gọn.",
            verbose=True,
            llm=self.llm
        )

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
        context = "\n\n".join([result["text"] for result in search_results])
        context = context.replace("Tải về:", "").replace(".xlsm", "")
        for page in ["133", "134", "135", "136", "137", "138", "139", "140", "141", "142"]:
            context = context.replace(page, "")
        
        query_lower = query.lower()
        relevant_keywords = []
        for topic, keywords in self.keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_keywords.extend(keywords)
                break

        # Thêm từ khóa cho các khoa và ngành học
        if "khoa" in query_lower or "ngành học" in query_lower or "liệt kê" in query_lower:
            relevant_keywords.extend([
                "khoa", "ngành học", "cơ cấu tổ chức", "khoa học cơ bản", "công nghệ thông tin",
                "công nghệ điện tử và truyền thông", "hệ thống thông tin kinh tế", "công nghệ tự động hóa",
                "truyền thông đa phương tiện", "thương mại điện tử"
            ])

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
        if "khóa 17" in query_lower:
            relevant_keywords.append("khóa 17")
        if "khóa 18" in query_lower or "khóa 19" in query_lower or "khóa 18-19" in query_lower:
            relevant_keywords.extend(["khóa 18", "khóa 19", "khóa 18-19"])
        if "khóa 20" in query_lower or "khóa 21" in query_lower or "khóa 20-21" in query_lower:
            relevant_keywords.extend(["khóa 20", "khóa 21", "khóa 20-21"])

        # Hàm kiểm tra nội dung liên quan
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
            return False

        # Xử lý câu hỏi về học phí của một ngành cụ thể
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
                            description=f"Trả lời học phí của ngành {selected_major} cho Khóa 17.",
                            agent=self.agent,
                            expected_output=f"Ngành {selected_major} Khóa 17: {tuition_fee:,} đồng/tín chỉ."
                        )
                    elif "khóa 18-19" in query_lower or "khóa 18" in query_lower or "khóa 19" in query_lower:
                        tuition_fee = block["Khóa 18-19"]
                        return Task(
                            description=f"Trả lời học phí của ngành {selected_major} cho Khóa 18-19.",
                            agent=self.agent,
                            expected_output=f"Ngành {selected_major} Khóa 18-19: {tuition_fee:,} đồng/tín chỉ."
                        )
                    elif "khóa 20-21" in query_lower or "khóa 20" in query_lower or "khóa 21" in query_lower:
                        tuition_fee = block["Khóa 20-21"]
                        return Task(
                            description=f"Trả lời học phí của ngành {selected_major} cho Khóa 20-21.",
                            agent=self.agent,
                            expected_output=f"Ngành {selected_major} Khóa 20-21: {tuition_fee:,} đồng/tín chỉ."
                        )
                    else:
                        # Đối với các khóa không có trong bảng, lấy học phí của Khóa 18-19
                        tuition_fee = block["Khóa 18-19"]
                        return Task(
                            description=f"Trả lời học phí của ngành {selected_major} cho khóa không xác định (mặc định Khóa 18-19).",
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
                description="Liệt kê các ngành học từ chương trình đào tạo đại trà.",
                agent=self.agent,
                expected_output=f"Danh sách các ngành học tại ICTU:\n{majors_list}"
            )

        # Kiểm tra câu hỏi về khoa và ngành học
        if "khoa" in query_lower or "ngành học" in query_lower or "liệt kê" in query_lower:
            if not has_relevant_content(["khoa", "ngành học", "cơ cấu tổ chức"], context.lower(), search_results):
                return Task(description="Không tìm thấy thông tin phù hợp.", agent=self.agent, expected_output="**Không tìm thấy thông tin về chủ đề này trong sổ tay ICTU.**")

        # Nới lỏng kiểm tra cho các chủ đề khác
        if "điều 3" in query_lower and not has_relevant_content(["điều 3", "chương trình đào tạo"], context.lower(), search_results):
            return Task(description="Không tìm thấy thông tin phù hợp.", agent=self.agent, expected_output="**Không tìm thấy thông tin về chủ đề này trong sổ tay ICTU.**")
        if "cử nhân" in query_lower and not has_relevant_content(["cử nhân", "chương trình đào tạo"], context.lower(), search_results):
            return Task(description="Không tìm thấy thông tin phù hợp.", agent=self.agent, expected_output="**Không tìm thấy thông tin về chủ đề này trong sổ tay ICTU.**")
        if "kỹ sư" in query_lower and not has_relevant_content(["kỹ sư", "trình độ bậc 7", "chương trình đào tạo"], context.lower(), search_results):
            return Task(description="Không tìm thấy thông tin phù hợp.", agent=self.agent, expected_output="**Không tìm thấy thông tin về chủ đề này trong sổ tay ICTU.**")
        if "xếp loại học lực" in query_lower and not has_relevant_content(["xếp loại học lực", "xuất sắc", "giỏi", "khá", "trung bình", "yếu", "kém", "điểm trung bình tích lũy"], context.lower(), search_results):
            return Task(description="Không tìm thấy thông tin phù hợp.", agent=self.agent, expected_output="**Không tìm thấy thông tin về chủ đề này trong sổ tay ICTU.**")
        if "xếp hạng học lực" in query_lower and not has_relevant_content(["xếp hạng học lực", "hạng bình thường", "hạng yếu", "điểm trung bình tích lũy"], context.lower(), search_results):
            return Task(description="Không tìm thấy thông tin phù hợp.", agent=self.agent, expected_output="**Không tìm thấy thông tin về chủ đề này trong sổ tay ICTU.**")
        if "đăng nhập" in query_lower and not has_relevant_content(["đăng nhập", "mã sinh viên", "mật khẩu", "hệ thống"], context.lower(), search_results):
            return Task(description="Không tìm thấy thông tin phù hợp.", agent=self.agent, expected_output="**Không tìm thấy thông tin về chủ đề này trong sổ tay ICTU.**")
        if "đổi mật khẩu" in query_lower and not has_relevant_content(["đổi mật khẩu", "thay đổi mật khẩu", "cập nhật mật khẩu", "quản lý tài khoản", "phòng đào tạo", "hệ thống"], context.lower(), search_results):
            return Task(description="Không tìm thấy thông tin phù hợp.", agent=self.agent, expected_output="**Không tìm thấy thông tin về chủ đề này trong sổ tay ICTU.**")
        if "đăng ký học" in query_lower and not has_relevant_content(["đăng ký học", "học phần", "hệ thống", "đăng ký tín chỉ"], context.lower(), search_results):
            return Task(description="Không tìm thấy thông tin phù hợp.", agent=self.agent, expected_output="**Không tìm thấy thông tin về chủ đề này trong sổ tay ICTU.**")
        if "thanh toán học phí" in query_lower or "nộp tiền trước" in query_lower:
            if not has_relevant_content(["thanh toán học phí", "online", "đăng ký học kỳ tới", "nộp tiền trước", "chức năng thanh toán", "hệ thống"], context.lower(), search_results):
                return Task(description="Không tìm thấy thông tin phù hợp.", agent=self.agent, expected_output="**Không tìm thấy thông tin về chủ đề này trong sổ tay ICTU.**")
        if any(major.lower() in query_lower for major in majors) and ("học phí" in query_lower or "mỗi tín chỉ" in query_lower or "bao nhiêu tiền" in query_lower):
            if not has_relevant_content(majors + ["học phí", "đồng/tín chỉ", "mỗi tín chỉ"], context.lower(), search_results):
                return Task(description="Không tìm thấy thông tin phù hợp.", agent=self.agent, expected_output="**Không tìm thấy thông tin về chủ đề này trong sổ tay ICTU.**")

        task = Task(
            description=f"""
                Trích xuất thông tin chính xác để trả lời câu hỏi: '{query}' từ nội dung sau:
                {context}

                ⚠️ Yêu cầu:
                - Chỉ trích xuất đúng nội dung liên quan nhất từ đoạn văn trên.
                - Ưu tiên các đoạn văn chứa các từ khóa: {', '.join(relevant_keywords)}.
                - Nếu câu hỏi hỏi về "liệt kê các khoa" hoặc "các khoa tại ICTU", trích xuất danh sách các khoa từ các đoạn văn chứa từ khóa "khoa" và liệt kê đầy đủ.
                - Nếu câu hỏi hỏi về "ngành học" hoặc "liệt kê các ngành học", trích xuất thông tin về các ngành học nếu có (ví dụ: "Thương mại điện tử" từ các đoạn văn liên quan). Nếu không có thông tin chi tiết về ngành học, suy luận ngành học dựa trên tên khoa (ví dụ: Khoa Công nghệ Thông tin có thể đào tạo ngành Công nghệ Thông tin) và ghi rõ rằng đây là suy luận.
                - Nếu câu hỏi hỏi về "số lượng ngành học", đếm số lượng ngành học dựa trên thông tin có sẵn hoặc suy luận từ tên khoa, và ghi rõ nếu là suy luận.
                - Nếu câu hỏi hỏi về "xếp loại học lực", chỉ trích xuất các đoạn chứa "xếp loại học lực", "xuất sắc", "giỏi", "khá", "trung bình", "yếu", "kém", hoặc "điểm trung bình tích lũy".
                - Nếu câu hỏi hỏi về "xếp hạng học lực", chỉ trích xuất các đoạn chứa "xếp hạng học lực", "hạng bình thường", "hạng yếu", hoặc "điểm trung bình tích lũy".
                - Nếu câu hỏi hỏi về cả "xếp loại học lực" và "xếp hạng học lực", trích xuất cả hai nội dung và trình bày rõ ràng, phân tách bằng dấu xuống dòng.
                - Nếu câu hỏi hỏi về "đăng nhập", chỉ trích xuất các đoạn chứa "đăng nhập", "mã sinh viên", "mật khẩu", "ngày tháng năm sinh", "viết hoa", "hệ thống".
                - Nếu câu hỏi hỏi về "đổi mật khẩu", chỉ trích xuất các đoạn chứa "đổi mật khẩu", "thay đổi mật khẩu", "cập nhật mật khẩu", "quản lý tài khoản", "phòng đào tạo", "hệ thống".
                - Nếu câu hỏi hỏi về "đăng ký học", chỉ trích xuất các đoạn chứa "đăng ký học", "học phần", "hệ thống", "đăng ký tín chỉ".
                - Nếu câu hỏi hỏi về "thanh toán học phí" hoặc "nộp tiền trước", chỉ trích xuất các đoạn chứa "thanh toán học phí", "online", "đăng ký học kỳ tới", "nộp tiền trước", "chức năng thanh toán", "hệ thống". Nếu câu hỏi có cụm "nộp tiền trước", ưu tiên các đoạn văn chứa "nộp tiền trước" và mô tả quy trình nộp tiền.
                - Nếu câu hỏi hỏi về "hủy học phần", chỉ trích xuất các đoạn chứa "hủy học phần", "đăng ký nhầm".
                - Nếu câu hỏi hỏi về "xem lịch thi", chỉ trích xuất các đoạn chứa "xem lịch thi", "học kỳ", "đợt học".
                - Nếu câu hỏi hỏi về "tra cứu điểm", chỉ trích xuất các đoạn chứa "tra cứu điểm", "học kỳ".
                - Nếu câu hỏi không thuộc các chủ đề trên, trích xuất các đoạn văn liên quan nhất dựa trên từ khóa và ngữ nghĩa của câu hỏi.
                - KHÔNG được viết lại, diễn giải lại hay sáng tạo thêm.
                - KHÔNG cần tóm tắt.
                - Nếu trong đoạn văn có câu trả lời trực tiếp, hãy trích nguyên văn.
                - Nếu không tìm thấy thông tin phù hợp, trả lời: **"Không tìm thấy thông tin về chủ đề này trong sổ tay ICTU."**

                Chỉ trả lời bằng tiếng Việt.
                """,
            agent=self.agent,
            expected_output="Câu trả lời chính xác, không tóm tắt, không viết lại, chỉ trích xuất đúng thông tin từ đoạn văn."
        )
        return task

    def run(self, query, top_k=50):
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
            crew = Crew(agents=[self.agent], tasks=[task], verbose=2)
            result = crew.kickoff()
            # Lấy ID từ kết quả có nội dung khớp với câu trả lời
            if search_results:
                relevant_result = None
                for res in search_results:
                    # So sánh chính xác hơn bằng cách kiểm tra xem result có phải là một phần của res['text']
                    if result.strip() in res['text'].strip():
                        relevant_result = res
                        break
                if not relevant_result:
                    # Nếu không tìm thấy khớp chính xác, thử tìm kiếm dựa trên từ khóa chính trong result
                    for res in search_results:
                        if "Đoàn thanh niên" in res['text'] and "Nhà điều hành C1" in res['text']:
                            relevant_result = res
                            break
                if not relevant_result:
                    relevant_result = search_results[0]  # Mặc định lấy kết quả có score cao nhất nếu không tìm thấy
                result_with_id = f"{result} (ID: {relevant_result['metadata']['id']})"
            else:
                result_with_id = result
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
            response = bot.run(query, top_k=18)
            print("\n💬 Phản hồi từ trợ lý:")
            print(response)
            print("\n" + "="*50)

    except Exception as e:
        print(f"❌ Lỗi khi khởi động hoặc chạy agent: {e}")
    print("🎉 Hoàn tất!")