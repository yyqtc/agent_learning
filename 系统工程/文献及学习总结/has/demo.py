"""
HAS系统最小可行性验证Demo
使用SpaCy NER Pipeline实现提示词脱敏技术
"""

import json
import re
from typing import List, Dict

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("警告：SpaCy未安装，将使用模拟数据进行演示")


class HASSystem:
    """HAS脱敏系统核心类"""
    
    def __init__(self, model_name="en_core_web_sm", use_mock=False):
        """
        初始化HAS系统
        
        Args:
            model_name: SpaCy模型名称
            use_mock: 是否使用模拟数据（如果SpaCy不可用）
        """
        self.use_mock = use_mock
        
        if SPACY_AVAILABLE and not use_mock:
            try:
                print(f"正在加载模型: {model_name}...")
                self.nlp = spacy.load(model_name)
                self.use_mock = False
                print("模型加载完成！")
            except OSError:
                print(f"模型 {model_name} 未找到，使用模拟模式")
                self.use_mock = True
        else:
            self.use_mock = True
            if use_mock:
                print("使用模拟模式进行演示")
    
    def _mock_recognize_entities(self, text: str) -> List[Dict]:
        """
        模拟实体识别（用于演示）
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        # 简单的规则匹配用于演示
        entities = []
        
        # 常见人名模式
        person_pattern = r'\b(John|Alice|Bob|Charlie|David|Emma|Frank|Grace|Henry|Ivy|Jane|Kevin|Laura|Mike|Nancy|Oliver|Peter|Queen|Rachel|Sarah|Tom|Una|Victor|Wendy)\b'
        
        # 地名模式
        location_pattern = r'\b(New York|London|Paris|Beijing|Shanghai|Tokyo|Berlin|Moscow|Sydney|Los Angeles|Washington|California|Texas|Florida)\b'
        
        # 组织名模式
        org_pattern = r'\b(Google|Microsoft|Apple|Amazon|Facebook|IBM|Intel|Oracle|Samsung|Sony)\b'
        
        # 金额模式
        money_pattern = r'\$[\d,]+(?:\.\d{2})?|\d+元'
        
        patterns = [
            (person_pattern, "PERSON"),
            (location_pattern, "GPE"),
            (org_pattern, "ORG"),
            (money_pattern, "MONEY")
        ]
        
        for pattern, label in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end()
                })
        
        return entities
    
    def recognize_entities(self, text: str) -> List[Dict]:
        """
        识别文本中的敏感实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表，包含文本、标签、起始和结束位置
        """
        if self.use_mock:
            return self._mock_recognize_entities(text)
        
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return entities
    
    def anonymize_text(self, text: str, entities: List[Dict]) -> tuple:
        """
        对文本进行脱敏处理
        
        Args:
            text: 原始文本
            entities: 识别的实体列表
            
        Returns:
            元组 (脱敏后文本, 映射表)
        """
        # 按照出现顺序倒序排序，从后往前替换避免偏移问题
        entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        anonymized_text = text
        mapping_table = {}
        counter = {"PERSON": 1, "ORG": 1, "GPE": 1, "LOC": 1, "DATE": 1, "MONEY": 1}  # 可扩展
        
        for entity in entities:
            entity_type = entity["label"]
            if entity_type not in counter:
                entity_type = "OTHER"
                if "OTHER" not in counter:
                    counter["OTHER"] = 1
            
            placeholder = f"[{entity_type}_{counter[entity_type]}]"
            mapping_table[placeholder] = entity["text"]
            
            # 从后往前替换避免偏移问题
            anonymized_text = anonymized_text[:entity['start']] + placeholder + anonymized_text[entity['end']:]
            
            counter[entity_type] += 1
        
        return anonymized_text, mapping_table
    
    def deanonymize_text(self, text: str, mapping_table: Dict) -> str:
        """
        对文本进行去匿名化处理
        
        Args:
            text: 包含占位符的文本
            mapping_table: 映射表
            
        Returns:
            恢复原始实体的文本
        """
        result = text
        for placeholder, original_value in mapping_table.items():
            result = result.replace(placeholder, original_value)
        return result


def demo():
    """主Demo函数"""
    print("=" * 60)
    print("HAS系统最小可行性验证Demo")
    print("=" * 60)
    print()
    
    # 创建HAS系统实例（默认使用模拟模式以确保演示可以运行）
    has_system = HASSystem(use_mock=False)
    print()
    
    # 测试用例1：英文文本
    print("测试用例1：英文文本")
    print("-" * 60)
    input_text_en = "John works in New York at Google with a salary of $100,000 per year."
    print(f"原始文本: {input_text_en}")
    
    # 步骤1：实体识别
    entities_en = has_system.recognize_entities(input_text_en)
    print(f"\n识别出的实体: {entities_en}")
    
    # 步骤2：脱敏处理
    anon_text_en, mapping_en = has_system.anonymize_text(input_text_en, entities_en)
    print(f"\n脱敏后文本: {anon_text_en}")
    print(f"映射表: {json.dumps(mapping_en, ensure_ascii=False, indent=2)}")
    
    # 步骤3：模拟LLM处理
    llm_output_en = anon_text_en.replace("works", "is working")
    print(f"\nLLM输出（模拟）: {llm_output_en}")
    
    # 步骤4：去匿名化
    final_output_en = has_system.deanonymize_text(llm_output_en, mapping_en)
    print(f"\n最终输出: {final_output_en}")
    print()
    
    # 测试用例2：简单文本
    print("测试用例2：简单文本")
    print("-" * 60)
    input_text_simple = "Alice is from London."
    print(f"原始文本: {input_text_simple}")
    
    entities_simple = has_system.recognize_entities(input_text_simple)
    print(f"\n识别出的实体: {entities_simple}")
    
    anon_text_simple, mapping_simple = has_system.anonymize_text(input_text_simple, entities_simple)
    print(f"\n脱敏后文本: {anon_text_simple}")
    print(f"映射表: {json.dumps(mapping_simple, ensure_ascii=False, indent=2)}")
    
    final_output_simple = has_system.deanonymize_text(anon_text_simple, mapping_simple)
    print(f"\n最终输出（反向转换）: {final_output_simple}")
    print()
    
    # 测试用例3：无实体文本
    print("测试用例3：无实体文本")
    print("-" * 60)
    input_text_no_entity = "The weather is beautiful today."
    print(f"原始文本: {input_text_no_entity}")
    
    entities_no_entity = has_system.recognize_entities(input_text_no_entity)
    print(f"\n识别出的实体: {entities_no_entity}")
    
    if entities_no_entity:
        anon_text_no_entity, mapping_no_entity = has_system.anonymize_text(input_text_no_entity, entities_no_entity)
        print(f"\n脱敏后文本: {anon_text_no_entity}")
    else:
        print("\n未识别出实体，文本无需脱敏")
    
    print()
    print("=" * 60)
    print("Demo验证完成！")
    print("=" * 60)


if __name__ == "__main__":
    try:
        demo()
    except OSError as e:
        if "Can't find model" in str(e):
            print("\n错误：找不到SpaCy模型")
            print("请先安装SpaCy模型：")
            print("  python -m spacy download en_core_web_sm")
        else:
            print(f"\n错误：{e}")
    except Exception as e:
        print(f"\n发生错误：{e}")
        import traceback
        traceback.print_exc()

