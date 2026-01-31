"""
Self-Learning Coding AI Assistant - All-in-One Version
Một AI Assistant tự học từ conversations
"""

import os
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()


# ============================================================================
# MEMORY SYSTEM
# ============================================================================

class MemorySystem:
    """Lưu trữ và truy xuất kiến thức từ conversations"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.conversations_dir = self.data_dir / "conversations"
        self.knowledge_dir = self.data_dir / "knowledge_base"
        
        # Tạo thư mục
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.data_dir / "embeddings")
        )
        
        # Collections
        self.conversations_collection = self.chroma_client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.code_collection = self.chroma_client.get_or_create_collection(
            name="code_examples",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Encoder
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def save_conversation(self, user_message, assistant_response, metadata=None):
        """Lưu conversation và tạo embedding"""
        timestamp = datetime.now().isoformat()
        conversation_id = f"conv_{timestamp.replace(':', '-')}"
        
        # Lưu JSON
        conversation_data = {
            "id": conversation_id,
            "timestamp": timestamp,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "metadata": metadata or {}
        }
        
        filepath = self.conversations_dir / f"{conversation_id}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        
        # Tạo embedding
        combined_text = f"Q: {user_message}\nA: {assistant_response}"
        
        self.conversations_collection.add(
            documents=[combined_text],
            metadatas=[{
                "timestamp": timestamp,
                "type": "conversation",
                **(metadata or {})
            }],
            ids=[conversation_id]
        )
        
        # Lưu code nếu có
        if "```" in assistant_response:
            self._extract_and_save_code(user_message, assistant_response, conversation_id)
        
        return conversation_id
    
    def _extract_and_save_code(self, question, response, conversation_id):
        """Trích xuất code blocks"""
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', response, re.DOTALL)
        
        for idx, (language, code) in enumerate(code_blocks):
            code_id = f"{conversation_id}_code_{idx}"
            
            self.code_collection.add(
                documents=[code],
                metadatas=[{
                    "language": language or "unknown",
                    "question": question,
                    "conversation_id": conversation_id
                }],
                ids=[code_id]
            )
    
    def search_similar_conversations(self, query, n_results=5):
        """Tìm conversations tương tự"""
        results = self.conversations_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    
    def search_code_examples(self, query, language=None, n_results=3):
        """Tìm code examples"""
        where_filter = {"language": language} if language else None
        
        results = self.code_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        return results
    
    def get_conversation_stats(self):
        """Thống kê"""
        total_conversations = len(list(self.conversations_dir.glob("*.json")))
        
        return {
            "total_conversations": total_conversations,
            "conversations_in_db": self.conversations_collection.count(),
            "code_examples": self.code_collection.count()
        }
    
    def export_knowledge_base(self, output_file="knowledge_export.json"):
        """Export knowledge"""
        all_data = []
        
        for conv_file in self.conversations_dir.glob("*.json"):
            with open(conv_file, 'r', encoding='utf-8') as f:
                all_data.append(json.load(f))
        
        export_path = self.knowledge_dir / output_file
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        return export_path


# ============================================================================
# LEARNING ENGINE
# ============================================================================

class LearningEngine:
    """Tự động học từ interactions"""
    
    def __init__(self, memory_system, learning_threshold=10):
        self.memory = memory_system
        self.learning_threshold = learning_threshold
        self.patterns_file = Path("data/knowledge_base/learned_patterns.json")
        self.patterns_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.patterns = self._load_patterns()
        
    def _load_patterns(self):
        """Load patterns"""
        if self.patterns_file.exists():
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "topics": {},
            "code_patterns": {},
            "error_solutions": {},
            "best_practices": []
        }
    
    def _save_patterns(self):
        """Lưu patterns"""
        with open(self.patterns_file, 'w', encoding='utf-8') as f:
            json.dump(self.patterns, f, ensure_ascii=False, indent=2)
    
    def analyze_conversation(self, user_message, assistant_response):
        """Phân tích và học"""
        
        # Phát hiện topics
        topics = self._extract_topics(user_message)
        for topic in topics:
            self.patterns["topics"][topic] = self.patterns["topics"].get(topic, 0) + 1
        
        # Học code patterns
        if "```" in assistant_response:
            self._learn_code_pattern(user_message, assistant_response)
        
        # Học error solutions
        if any(keyword in user_message.lower() for keyword in ["error", "bug", "lỗi", "sai"]):
            self._learn_error_solution(user_message, assistant_response)
        
        # Best practices
        if any(keyword in user_message.lower() for keyword in ["best practice", "tốt nhất", "nên"]):
            self._extract_best_practice(assistant_response)
        
        self._save_patterns()
    
    def _extract_topics(self, text):
        """Trích xuất topics"""
        programming_keywords = {
            "python", "javascript", "java", "c++", "react", "node",
            "database", "sql", "api", "algorithm", "data structure",
            "machine learning", "ai", "web", "backend", "frontend",
            "docker", "kubernetes", "git", "testing", "debug"
        }
        
        text_lower = text.lower()
        found_topics = [keyword for keyword in programming_keywords if keyword in text_lower]
        return found_topics
    
    def _learn_code_pattern(self, question, response):
        """Học code patterns"""
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', response, re.DOTALL)
        
        for language, code in code_blocks:
            if not language:
                continue
            
            pattern_key = f"{language}_pattern"
            
            if pattern_key not in self.patterns["code_patterns"]:
                self.patterns["code_patterns"][pattern_key] = []
            
            self.patterns["code_patterns"][pattern_key].append({
                "question": question,
                "code_snippet": code[:500],
                "timestamp": datetime.now().isoformat()
            })
            
            # Giữ tối đa 50
            if len(self.patterns["code_patterns"][pattern_key]) > 50:
                self.patterns["code_patterns"][pattern_key] = \
                    self.patterns["code_patterns"][pattern_key][-50:]
    
    def _learn_error_solution(self, error_description, solution):
        """Học error solutions"""
        error_key = error_description.lower()[:100]
        
        if error_key not in self.patterns["error_solutions"]:
            self.patterns["error_solutions"][error_key] = []
        
        self.patterns["error_solutions"][error_key].append({
            "solution": solution[:1000],
            "timestamp": datetime.now().isoformat(),
            "count": 1
        })
    
    def _extract_best_practice(self, response):
        """Trích xuất best practices"""
        sentences = re.split(r'[.!?]', response)
        best_practice_keywords = ["nên", "should", "best", "recommend", "khuyến nghị"]
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in best_practice_keywords):
                if sentence.strip() and sentence.strip() not in self.patterns["best_practices"]:
                    self.patterns["best_practices"].append(sentence.strip())
        
        if len(self.patterns["best_practices"]) > 100:
            self.patterns["best_practices"] = self.patterns["best_practices"][-100:]
    
    def get_learned_knowledge(self, topic=None):
        """Lấy knowledge theo topic"""
        if topic:
            return {
                "topic": topic,
                "frequency": self.patterns["topics"].get(topic, 0),
                "code_patterns": [
                    p for key, patterns in self.patterns["code_patterns"].items()
                    if topic.lower() in key.lower()
                    for p in patterns
                ],
                "related_best_practices": [
                    bp for bp in self.patterns["best_practices"]
                    if topic.lower() in bp.lower()
                ]
            }
        return self.patterns
    
    def get_trending_topics(self, days=7, top_n=10):
        """Topics trending"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_topics = Counter()
        
        for conv_file in self.memory.conversations_dir.glob("*.json"):
            with open(conv_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conv_time = datetime.fromisoformat(data['timestamp'])
                
                if conv_time >= cutoff_date:
                    topics = self._extract_topics(data['user_message'])
                    recent_topics.update(topics)
        
        return recent_topics.most_common(top_n)
    
    def generate_context_for_query(self, query):
        """Tạo context cho query"""
        similar_convs = self.memory.search_similar_conversations(query, n_results=3)
        code_examples = self.memory.search_code_examples(query, n_results=2)
        
        topics = self._extract_topics(query)
        relevant_knowledge = []
        for topic in topics:
            knowledge = self.get_learned_knowledge(topic)
            if knowledge['code_patterns']:
                relevant_knowledge.append(knowledge)
        
        return {
            "similar_conversations": similar_convs,
            "code_examples": code_examples,
            "learned_knowledge": relevant_knowledge
        }
    
    def get_learning_stats(self):
        """Thống kê học tập"""
        return {
            "total_topics": len(self.patterns["topics"]),
            "top_topics": sorted(
                self.patterns["topics"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            "code_patterns_count": sum(
                len(patterns) for patterns in self.patterns["code_patterns"].values()
            ),
            "error_solutions_count": len(self.patterns["error_solutions"]),
            "best_practices_count": len(self.patterns["best_practices"])
        }


# ============================================================================
# MAIN AI CLASS
# ============================================================================

class SelfLearningCodingAI:
    """AI Assistant tự học"""
    
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.memory = MemorySystem()
        self.learning_engine = LearningEngine(self.memory)
        
        self.base_system_prompt = """Bạn là một AI Assistant chuyên về lập trình với khả năng tự học.

Nhiệm vụ của bạn:
1. Giải đáp các câu hỏi về lập trình một cách chi tiết và chính xác
2. Viết code examples rõ ràng, có comment
3. Giải thích concepts phức tạp một cách dễ hiểu
4. Debug và tìm lỗi trong code
5. Đề xuất best practices và optimizations
6. Review code và đưa ra feedback xây dựng

Hỗ trợ ngôn ngữ: Python, JavaScript, Java, C++, Go, Rust, SQL, và nhiều ngôn ngữ khác.

Khi trả lời:
- Luôn giải thích TẠI SAO, không chỉ LÀM SAO
- Đưa ra ví dụ cụ thể
- Highlight các điểm cần lưu ý
- So sánh các approaches khác nhau nếu có
"""
    
    def _build_enhanced_prompt(self, user_message):
        """Build prompt với learned context"""
        context = self.learning_engine.generate_context_for_query(user_message)
        enhanced_prompt = self.base_system_prompt
        
        # Similar conversations
        if context['similar_conversations']['documents']:
            enhanced_prompt += "\n\n--- Kiến thức từ conversations trước: ---\n"
            for doc in context['similar_conversations']['documents'][0][:2]:
                enhanced_prompt += f"\n{doc}\n"
        
        # Code examples
        if context['code_examples']['documents']:
            enhanced_prompt += "\n\n--- Code examples liên quan: ---\n"
            for idx, code in enumerate(context['code_examples']['documents'][0][:2]):
                lang = context['code_examples']['metadatas'][0][idx].get('language', 'code')
                enhanced_prompt += f"\n```{lang}\n{code}\n```\n"
        
        # Best practices
        if context['learned_knowledge']:
            enhanced_prompt += "\n\n--- Best practices đã học: ---\n"
            for knowledge in context['learned_knowledge'][:2]:
                if knowledge['related_best_practices']:
                    for bp in knowledge['related_best_practices'][:3]:
                        enhanced_prompt += f"- {bp}\n"
        
        return enhanced_prompt
    
    def chat(self, user_message, conversation_history=None):
        """Chat và tự học"""
        if conversation_history is None:
            conversation_history = []
        
        system_prompt = self._build_enhanced_prompt(user_message)
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            
            assistant_response = response.choices[0].message.content
            
            # Lưu conversation
            self.memory.save_conversation(
                user_message=user_message,
                assistant_response=assistant_response,
                metadata={
                    "model": "gpt-4",
                    "tokens": response.usage.total_tokens
                }
            )
            
            # Học từ conversation
            self.learning_engine.analyze_conversation(user_message, assistant_response)
            
            return assistant_response
            
        except Exception as e:
            return f"Lỗi: {str(e)}"
    
    def get_stats(self):
        """Thống kê"""
        memory_stats = self.memory.get_conversation_stats()
        learning_stats = self.learning_engine.get_learning_stats()
        trending = self.learning_engine.get_trending_topics()
        
        return {
            "memory": memory_stats,
            "learning": learning_stats,
            "trending_topics": trending
        }
    
    def export_knowledge(self):
        """Export knowledge"""
        return self.memory.export_knowledge_base()


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """Main loop"""
    print("=" * 60)
    print("🚀 SELF-LEARNING CODING AI ASSISTANT")
    print("=" * 60)
    print("\nAI này sẽ tự học và cải thiện qua các câu hỏi của bạn!")
    print("\nCác lệnh đặc biệt:")
    print("  'stats' - Xem thống kê học tập")
    print("  'export' - Export kiến thức đã học")
    print("  'exit' - Thoát")
    print("\n" + "=" * 60 + "\n")
    
    ai = SelfLearningCodingAI()
    conversation_history = []
    
    while True:
        user_input = input("\n💬 Bạn: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'exit':
            print("\n👋 Tạm biệt! AI đã học được nhiều thứ từ bạn.")
            break
        
        if user_input.lower() == 'stats':
            stats = ai.get_stats()
            print("\n📊 THỐNG KÊ HỌC TẬP:")
            print(f"  • Tổng conversations: {stats['memory']['total_conversations']}")
            print(f"  • Code examples: {stats['memory']['code_examples']}")
            print(f"  • Topics đã học: {stats['learning']['total_topics']}")
            print(f"  • Best practices: {stats['learning']['best_practices_count']}")
            print(f"\n🔥 Top topics:")
            for topic, count in stats['trending_topics'][:5]:
                print(f"    {topic}: {count} lần")
            continue
        
        if user_input.lower() == 'export':
            filepath = ai.export_knowledge()
            print(f"\n✅ Đã export kiến thức vào: {filepath}")
            continue
        
        # Chat
        print("\n🤖 AI đang suy nghĩ...", end="", flush=True)
        response = ai.chat(user_input, conversation_history)
        print("\r" + " " * 30 + "\r", end="")
        
        print(f"🤖 AI: {response}")
        
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        
        # Giữ 20 messages gần nhất
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]


if __name__ == "__main__":
    main()
