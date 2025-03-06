from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForTokenClassification, BertTokenizerFast
import torch
import gradio as gr
import warnings

# Import with updated packages to avoid deprecation warnings
from langchain_community.llms import Ollama
    
try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    # Fallback to the deprecated import if the new package isn't installed
    from langchain_community.graphs import Neo4jGraph
    warnings.warn("Using deprecated Neo4jGraph import. Consider installing langchain_neo4j package.")
import base64
import os
from PIL import Image
import io

import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
model_path = "/home/qyj/code/KG/model/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
   model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
# Ollama LLM configuration
llm = Ollama(
    model="deepseek-r1:32b",
    base_url="http://127.0.0.1:8000"
)

# Model B for image analysis - placeholder
def model_b_analyze(image, question):
   # specify the path to the model
      # 创建临时存储目录
   os.makedirs("temp_images", exist_ok=True)
   
   # 保存PIL图像到临时文件
   temp_path = "temp_images/temp_img.jpg"
   image.save(temp_path)
   
   # 准备conversation格式
   conversation = [
      {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [temp_path],  # 使用文件路径
      },
      {"role": "<|Assistant|>", "content": ""},
   ]

   # 使用Janus的函数加载图像
   pil_images = load_pil_images(conversation)
   prepare_inputs = vl_chat_processor(
      conversations=conversation, images=pil_images, force_batchify=True
   ).to(vl_gpt.device)

   # # run image encoder to get the image embeddings
   inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
   # # run the model to get the response
   outputs = vl_gpt.language_model.generate(
      inputs_embeds=inputs_embeds,
      attention_mask=prepare_inputs.attention_mask,
      pad_token_id=tokenizer.eos_token_id,
      bos_token_id=tokenizer.bos_token_id,
      eos_token_id=tokenizer.eos_token_id,
      max_new_tokens=512,
      do_sample=False,
      use_cache=True,
)

   answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
   print(f"{prepare_inputs['sft_format'][0]}", answer)
   return answer

def generate_response(prompt):
    return llm.invoke(prompt, preserve_format=True)

# NER model setup
ner_tokenizer = BertTokenizerFast.from_pretrained('/home/qyj/code/KG/model/bert-base-chinese-medical-ner')
ner_model = AutoModelForTokenClassification.from_pretrained("/home/qyj/code/KG/model/bert-base-chinese-medical-ner")

# Neo4j connection
graph = Neo4jGraph(
    url = "neo4j+s://9a72e3e0.databases.neo4j.io",
    username = "neo4j",
    password = "wBykwkeh9HdyspnuRAX-1DJ789CKbrDB9W4cjlRJy8U",
)

def extract_entities(text):
   # 1. Extract entities
   inputs = ner_tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
   outputs = ner_model(**inputs)
   outputs = outputs.logits.argmax(-1) * inputs['attention_mask']
   s = format_outputs([text], outputs)[0]
   # 2. Extract keywords
   keywords = []
   for token in s:
      if(token['word'] not in keywords and token['word'] != ""):
         keywords.append(token['word'])
   return keywords

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
               # Abnormal index
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


def respond(message, image, history):
   # Process uploaded image if available

   image_analysis = model_b_analyze(image, message)
   
   print(f"图像分析结果: {image_analysis}")
   
   # Regular text processing
   entities = extract_entities(message)
   print(f"识别出的实体: {entities}")
   
   knowledge = query_knowledge(entities, message)
   context = format_knowledge(knowledge)
   print(f"查询到的知识: {context}")
   
   # Modify prompt to include image analysis if available
   if image:
       prompt = f"""基于以下知识：{context}
图像分析结果：{image_analysis}
问题：{message}
请基于上述知识、图像分析结果和自己的知识回答这个问题。如果知识库中没有相关信息，请说明无法回答，并尝试通过自己的知识回答！
回答：<think>"""
   else:
       prompt = f"""基于以下知识：{context}
问题：{message}
请基于上述知识和自己的知识回答这个问题。如果知识库中没有相关信息，请说明无法回答，并尝试通过自己的知识回答！
回答：<think>"""
   
   full_response = generate_response(prompt)
   print(f"生成的回答: {full_response}")
   
   # Separate thinking process and answer
   if "</think>" in full_response:
      thinking_part = full_response.split("</think>")[0]
      answer_part = full_response.split("</think>")[1]
      # Display thinking process in gray using Gradio's Markdown component
      thinking_markdown = f"<div style='background-color: #f0f0f0; padding: 10px;'>{thinking_part}</div>"
      return {
         "text": thinking_markdown + "\n" + answer_part,  # Combine thinking and answer parts
      }
   else:
      return {
        "text": full_response, 
      }

# Updated UI with image upload capability
with gr.Blocks(title="基于知识图谱和图像的智能问答系统") as iface:
   gr.Markdown("# 基于知识图谱和图像的智能问答系统")
   gr.Markdown("本系统使用实体识别、知识图谱和图像分析增强问答效果")
   
   with gr.Row():
      with gr.Column(scale=3):
         chatbot = gr.Chatbot(
               height=500,
               show_label=False,
               render_markdown=True,
               type="messages",  # Use newer message format to avoid deprecation warning
         )
      
      with gr.Column(scale=1):
         image_input = gr.Image(
               label="上传图片（可选）",
               type="pil",
         )
   
   with gr.Row():
      msg = gr.Textbox(
         placeholder="请输入您的问题...",
         show_label=False,
         container=False,
         scale=9,
      )
      submit_btn = gr.Button("发送", scale=1)
   
   with gr.Row():
      gr.Examples(
         [
               ["肝脏疾病有哪些?", None],
               ["如何预防肝脏疾病?", None],
               ["请你介绍肝脏位于哪个部位?", None],
               ["这张图片显示的是什么疾病?", None],  # This example would require an image
         ],
         inputs=[msg, image_input],
         label="示例问题",
      )
   
   def user(user_message, image, history):
      # 将用户消息和图片一起添加到聊天历史
      # 使用额外的metadata字段存储图片
      return "", image, history + [{"role": "user", "content": user_message}]
      
   def bot(image, history):
      # 获取最后一条用户消息及其关联的图片
      user_message = history[-1]["content"]
      
      # 调用respond函数，传入当前消息和关联图片
      response = respond(user_message, image, history)
      
      # 添加机器人回复到聊天历史
      history.append({"role": "assistant", "content": response["text"]})
      return history
    
   submit_btn.click(
      user,
      inputs=[msg, image_input, chatbot],
      outputs=[msg, image_input, chatbot],
      queue=False,
   ).then(
      bot,
      inputs=[image_input, chatbot],
      outputs=[chatbot],
   )
   
   msg.submit(
      user,
      inputs=[msg, image_input, chatbot],
      outputs=[msg, image_input, chatbot],
      queue=False,
   ).then(
      bot,
      inputs=[image_input, chatbot],
      outputs=[chatbot],
   )

if __name__ == "__main__":
   iface.launch(share=True)