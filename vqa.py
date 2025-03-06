from langchain import HuggingFacePipeline
from langchain_community.graphs import Neo4jGraph
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import gradio as gr
from langchain_community.llms import Ollama

# 删除原有的model和tokenizer相关代码
# 替换为Ollama
llm = llm = Ollama(
    model="deepseek-r1:32b",
    base_url="http://127.0.0.1:8000"
) # 或其他模型名称

def generate_response(prompt):
    return llm.invoke(prompt, preserve_format=True)

# Run pipeline


from transformers import AutoModelForTokenClassification, BertTokenizerFast

ner_tokenizer = BertTokenizerFast.from_pretrained('/home/qyj/code/KG/model/bert-base-chinese-medical-ner')
ner_model = AutoModelForTokenClassification.from_pretrained("/home/qyj/code/KG/model/bert-base-chinese-medical-ner")




# LLM模型
def extract_entities(text):
   # 1. 提取实体
   inputs = ner_tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
   outputs = ner_model(**inputs)
   outputs = outputs.logits.argmax(-1) * inputs['attention_mask']
   s = format_outputs([text], outputs)[0]\
   # 2. 提取关键词
   keywords = []
   for token in s:
      if(token['word'] not in keywords and token['word'] != ""):
         keywords.append(token['word'])
   return keywords
# LLM模型
# model_name = "/home/qyj/code/KG/model/DeepSeek-R1-Distill-Qwen-7B"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#    model_name,
#    torch_dtype=torch.float16,
#    device_map="auto",
#    trust_remote_code=True,
# )

# Neo4j连接
graph = Neo4jGraph(
    url = "neo4j+s://9a72e3e0.databases.neo4j.io",  # Neo4j服务器地址
    username = "neo4j",             # 用户名
    password = "wBykwkeh9HdyspnuRAX-1DJ789CKbrDB9W4cjlRJy8U",
)

def format_outputs(sentences, outputs):
   preds = []
   for i, pred_indices in enumerate(outputs):
      words = []
      start_idx = -1
      end_idx = -1
      flag = False
      for idx, pred_idx in enumerate(pred_indices):
            if pred_idx == 1:
               start_idx = idx
               flag = True
               continue

            if flag and pred_idx != 2 and pred_idx != 3:
               # 出现了不应该出现的index
               print("Abnormal prediction results for sentence", sentences[i])
               start_idx = -1
               end_idx = -1
               continue

            if pred_idx == 3:
               end_idx = idx

               words.append({
                  "start": start_idx,
                  "end": end_idx + 1,
                  "word": sentences[i][start_idx:end_idx+1]
               })
               start_idx = -1
               end_idx = -1
               flag = False
               continue

      preds.append(words)

   return preds

def query_knowledge(entities, text=""):
   if not entities:
       query = """
       MATCH (n)-[r]->(m)
       WHERE n.name CONTAINS $text OR m.name CONTAINS $text
       RETURN n.name, type(r), m.name
       """
       return graph.query(query, {"text": text})
   
   query = """
   MATCH (n)-[r]->(m)  
   WHERE any(entity IN $entities WHERE 
       n.name CONTAINS entity OR 
       m.name CONTAINS entity)
   RETURN n.name, type(r), m.name
   """
   return graph.query(query, {"entities": entities})

def format_knowledge(results):
   return "\n".join([
       f"{r['n.name']} {r['type(r)']} {r['m.name']}"
       for r in results
   ])

# def generate_response(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=4096,
#         do_sample=False
#     )
#     response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:])
#     return response

def respond(message, history):
   entities = extract_entities(message)
   print(f"识别出的实体: {entities}")
   
   knowledge = query_knowledge(entities, message)
   context = format_knowledge(knowledge)
   print(f"查询到的知识: {context}")
   
   prompt = f"""基于以下知识：{context}
问题：{message}
请基于上述知识和自己的知识回答这个问题。如果知识库中没有相关信息，请说明无法回答，并尝试通过自己的知识回答！
回答：<think>"""
   
   full_response = generate_response(prompt)
   print(f"生成的回答: {full_response}")
     # 分离思考过程和回答
   if "</think>" in full_response:
      thinking_part = full_response.split("</think>")[0]
      answer_part = full_response.split("</think>")[1]
      # 使用Gradio的Markdown组件显示灰色思考过程
        # 返回字典格式，确保Gradio可以正确处理
      thinking_markdown = f"<div style='background-color: #f0f0f0; padding: 10px;'>{thinking_part}</div>"
      return {
         "text": thinking_markdown + "\n" + answer_part,  # 合并思考和回答部分
      }
   else:
      return {
        "text": full_response, 
      }
   # return full_response

iface = gr.ChatInterface(
   respond,
   title="基于知识图谱的智能问答系统",
   description="本系统使用实体识别和知识图谱增强问答效果",
   examples=[
       "肝脏疾病有哪些?",
       "如何预防肝脏疾病?", 
       "请你介绍肝脏位于哪个部位?"
   ]
)

if __name__ == "__main__":
   iface.launch(share=True)