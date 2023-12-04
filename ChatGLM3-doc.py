import os
from pypdf import PdfReader
import docx
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoConfig, AutoModel, AutoTokenizer
from modelscope import snapshot_download
import json

def combine_pairs(input_list):
    combined_list = []
    for i in range(0, len(input_list), 2):
        if i + 1 < len(input_list):
            pair = [input_list[i], input_list[i + 1]]
            combined_list.append("\n".join(pair))
    return combined_list

def get_data(root_path):
    all_content = []

    files = os.listdir(root_path)
    for file in files:
        path = os.path.join(root_path, file)
        if path.endswith(".pdf"):
            content = ""
            with open(path, 'rb') as f:
                pdf_reader = PdfReader(f)
                pages_info = pdf_reader.pages
                for page_info in pages_info:
                    text = page_info.extract_text()
                    content += text
            all_content.append(content)

        elif path.endswith(".docx"):
            doc = docx.Document(path)
            paragraphs = doc.paragraphs
            content = [i.text for i in paragraphs]

            # 效果优化，长文件做prompt效果不太好，需要进行切割处理
            texts = ""
            for text in content:
                if len(text) <=1:
                    continue
                if len(texts) >= 150:
                    all_content.append(texts)
                    texts = ""
                else:
                    texts += text

        elif path.endswith(".txt"):
            with open(path, "r", encoding='utf-8') as f:
                content = f.read().split('\n')
                # 按每两条数据重新组合，形成向量库
                all_content.extend(combine_pairs(content))

    return all_content

class DFaiss:
    def __init__(self):
        self.sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.index = faiss.IndexFlatL2(384) # 需要告诉模型几维
        self.text_str_list = []

    def add_content(self, text_str_list):
        self.text_str_list = text_str_list
        text_emb = self.get_text_emb(text_str_list)
        self.index.add(text_emb)

    def get_text_emb(self, text_str_list):
        text_emb = self.sentence_model.encode(text_str_list)
        return text_emb

    def search(self, text):
        text_emb = self.get_text_emb([text])
        # D是距离，I是index
        D, I = self.index.search(text_emb, 3)
        if D[0][0]> 15:
            content = ""
        else:
            content = self.text_str_list[I[0][0]]
        return content

class Dprompt:
    def __init__(self):
        model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision="v1.0.0")
        
        # 载入Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).quantize(4).cuda()
        self.myfaiss = DFaiss()
        all_content = get_data(os.path.join("..", "datas"))
        self.myfaiss.add_content(all_content)
        self.maxlen = 5000

    def answer(self, text):
        prompt = self.myfaiss.search(text)
        print(prompt)
        if prompt:
            prompt_content = f"请根据问答对回答问题，问答对是：{prompt}, 问题是：{text}"
        else:
            prompt_content = text
        prompt_content = prompt_content[:self.maxlen]
        response, history = self.model.chat(self.tokenizer, prompt_content, history=[])

        return response


if __name__ == '__main__':
    model = Dprompt()
    while True:
        text = input("请输入：")
        reponse = model.answer(text)
        print(reponse)