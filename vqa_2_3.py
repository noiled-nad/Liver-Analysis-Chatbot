from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForTokenClassification, BertTokenizerFast
import torch
import gradio as gr
import warnings
import re
import os
# 导入所需库，确保使用适当的包
from langchain_community.llms import Ollama

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    # 如果新包未安装，回退到已弃用的导入
    from langchain_community.graphs import Neo4jGraph
    warnings.warn("Using deprecated Neo4jGraph import. Consider installing langchain_neo4j package.")
import base64
import os
from PIL import Image
import io

# Janus模型相关导入
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
os.environ['CUDA_VISIBLE_DEVICES'] = '1 '

# Janus模型加载
model_path = "/home/qyj/code/KG/model/Janus-Pro-7B"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt = AutoModelForCausalLM.from_pretrained(
   model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# Ollama LLM配置
llm = Ollama(
    model="deepseek-r1:32b",
    base_url="http://127.0.0.1:8000"
)

# Janus图像分析函数
def model_b_analyze(image, question):
   # 准备对话格式
   conversation = [
      {
          "role": "<|User|>",
          "content": f"<image_placeholder>\n{question}",
          "images": [image],
      },
      {"role": "<|Assistant|>", "content": ""},
   ]

   # 使用Janus处理器处理输入
   prepare_inputs = vl_chat_processor(
      conversations=conversation, images=[image], force_batchify=True
   ).to(vl_gpt.device)

   # 运行图像编码器获取图像嵌入
   inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
   
   # 运行模型获取响应
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

# 增加一个调试版本的generate_response函数
def generate_response(prompt, requires_tools=False, tools_info=None):
    # 如果需要工具调用，修改提示以包含工具信息
    if requires_tools:
        tools_prompt = f"""
你有以下工具可以使用:
{tools_info}

当你需要使用这些工具时，请明确说明你要使用哪个工具，格式如下:
<tool>工具名称</tool>
<input>工具输入</input>

例如:
<tool>图像分析</tool>
<input>这张图片里有什么?</input>

或者:
<tool>知识图谱</tool>
<input>肝硬化的症状</input>

请不要模拟工具输出，只需指明你要使用的工具。在工具执行后，你将收到工具返回的结果。
你可以根据需要多次使用工具，也可以根据前一个工具的结果决定是否使用下一个工具。
"""
        prompt = tools_prompt + "\n" + prompt
    
    try:
        # 返回流式生成的迭代器
        response = llm.invoke(prompt, preserve_format=True, stream=True)
        
        # 添加一个调试包装器来检查返回的数据类型
        def debug_wrapper(stream):
            for chunk in stream:
                try:
                    if hasattr(chunk, 'content'):
                        print(f"DEBUG: chunk类型: {type(chunk)}, content类型: {type(chunk.content)}")
                    else:
                        print(f"DEBUG: chunk类型: {type(chunk)}")
                        
                    # 如果是列表，打印第一个元素的类型
                    if isinstance(chunk, list) and len(chunk) > 0:
                        print(f"DEBUG: 列表第一个元素类型: {type(chunk[0])}")
                except Exception as e:
                    print(f"DEBUG: 检查chunk时出错: {e}")
                yield chunk
        
        return debug_wrapper(response)
    except Exception as e:
        print(f"调用LLM时发生错误: {e}")
        # 返回一个包含错误消息的流
        return [f"生成响应时发生错误: {str(e)}"]

# NER模型设置
ner_tokenizer = BertTokenizerFast.from_pretrained('/home/qyj/code/KG/model/bert-base-chinese-medical-ner')
ner_model = AutoModelForTokenClassification.from_pretrained("/home/qyj/code/KG/model/bert-base-chinese-medical-ner")

# Neo4j连接
graph = Neo4jGraph(
    url = "neo4j+s://9a72e3e0.databases.neo4j.io",
    username = "neo4j",
    password = "wBykwkeh9HdyspnuRAX-1DJ789CKbrDB9W4cjlRJy8U",
)

# 实体提取函数
def extract_entities(text):
   # 提取实体
   inputs = ner_tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
   outputs = ner_model(**inputs)
   outputs = outputs.logits.argmax(-1) * inputs['attention_mask']
   s = format_outputs([text], outputs)[0]
   
   # 提取关键词
   keywords = []
   for token in s:
      if(token['word'] not in keywords and token['word'] != ""):
         keywords.append(token['word'])
   return keywords

# 格式化输出函数
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
               # 异常索引
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

# 知识查询函数
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

# 格式化知识函数
def format_knowledge(results):
   if not results:
       return "未找到相关知识"
   return "\n".join([
       f"{r['n.name']} {r['type(r)']} {r['m.name']}"
       for r in results
   ])

# 将工具定义为可扩展的字典
AVAILABLE_TOOLS = {
   "图像分析": {
      "keywords": ["图片", "图像", "照片", "看一下", "分析图", "识别图"],
      "description": "用于分析和理解上传的图片内容",
      "function": model_b_analyze,
      "requires_image": True
   },
   "知识图谱": {
      "keywords": ["查询", "知识库", "找一下", "了解", "信息"],
      "description": "用于查询医学知识库获取相关信息",
      "function": lambda query, *args: format_knowledge(query_knowledge(extract_entities(query), query)),
      "requires_image": False
   },
   # 可以轻松添加更多工具
}

# 修改 process_response_stream 函数中的历史格式处理
def process_response_stream(response_stream, message, image, history):
   # Initialize variables
   full_response = ""
   found_think_tag = False
   thinking_part = ""
   answer_part = ""
   
   # Ensure history is a list
   if history is None:
      history = []
   print(4444, history)
   # Initialize bot reply in chat history
   history.append({"role": "assistant", "content": ""})
   
   # Tool processing functions remain the same
   def find_and_execute_tools(text):
      """Find and execute all tool calls in the text"""
      tool_pattern = r"<tool>(.*?)</tool>\s*<input>(.*?)</input>"
      matches = re.findall(tool_pattern, text)
      
      all_tool_results = []
      
      for tool_name, tool_input in matches:
         tool_name = tool_name.strip()
         tool_input = tool_input.strip()
         
         print(f"执行工具调用: {tool_name}, 输入: {tool_input}")
         
         # Execute tool and get result
         tool_result = ""
         if tool_name == "图像分析" and image is not None:
               tool_result = model_b_analyze(image, tool_input)
               print(f"Janus图像分析结果: {tool_result}")
         
         elif tool_name == "知识图谱":
               entities = extract_entities(tool_input)
               print(f"识别出的实体: {entities}")
               knowledge = query_knowledge(entities, tool_input)
               tool_result = format_knowledge(knowledge)
               print(f"知识图谱查询结果: {tool_result}")
         
         else:
               tool_result = f"未知工具: {tool_name}"
         
         tool_display = {
               "name": tool_name,
               "input": tool_input,
               "result": tool_result
         }
         
         all_tool_results.append(tool_display)
      
      return all_tool_results
   
   def replace_tool_calls_with_results(text, results):
      """Replace tool calls in text with results"""
      modified_text = text
      
      tool_results_html = ""
      for result in results:
         tool_html = f"""<div style='background-color: #e6f7ff; padding: 10px; border-left: 4px solid #1890ff; margin: 10px 0;'>
<b>工具调用:</b> {result['name']}<br>
<b>输入:</b> {result['input']}<br>
<b>结果:</b><br>{result['result']}
</div>"""
         tool_results_html += tool_html
      
      tool_pattern = r"<tool>.*?</tool>\s*<input>.*?</input>"
      match = re.search(tool_pattern, modified_text)
      if match:
         start, end = match.span()
         modified_text = modified_text[:end] + "\n\n" + tool_results_html + "\n\n" + modified_text[end:]
      
      return modified_text
   
   # Streaming phase
   accumulated_response = ""
   pending_tool_calls = []
   tool_execution_counter = 0
   max_tool_executions = 5

   try:
      for chunk in response_stream:
         # Handle different chunk formats - THIS IS THE FIX
         chunk_text = ""
         if hasattr(chunk, 'content'):
            # If chunk is an object with content attribute (like StreamingResponse)
            chunk_text = chunk.content
         elif isinstance(chunk, str):
            # If chunk is directly a string
            chunk_text = chunk
         elif isinstance(chunk, list):
            # If chunk is a list, convert to string
            chunk_text = str(chunk[0]) if len(chunk) > 0 else ""
         else:
            # Fallback for other types
            chunk_text = str(chunk)
            
         print(f"Chunk: {chunk_text}")
         
         # Add new chunk to accumulated response
         accumulated_response += chunk_text
         
         # Update global response
         full_response = accumulated_response
         
         # Check for new tool calls
         tool_pattern = r"<tool>(.*?)</tool>\s*<input>(.*?)</input>"
         if re.search(tool_pattern, accumulated_response):
               if tool_execution_counter >= max_tool_executions:
                  print(f"已达到最大工具调用次数 ({max_tool_executions})，停止执行更多工具")
                  limit_notice = f"\n\n<div style='color: red;'>注意：已达到最大工具调用次数 ({max_tool_executions})。</div>\n\n"
                  full_response += limit_notice
                  accumulated_response += limit_notice
               else:
                  tool_results = find_and_execute_tools(accumulated_response)
                  
                  if tool_results:
                     tool_execution_counter += len(tool_results)
                     pending_tool_calls.extend(tool_results)
                     
                     augmented_response = replace_tool_calls_with_results(accumulated_response, tool_results)
                     
                     full_response = augmented_response
                     
                     tools_context = ""
                     for tool in tool_results:
                           tools_context += f"\n工具 \"{tool['name']}\" 返回的结果 (针对输入: {tool['input']}):\n{tool['result']}\n"
                     
                     new_prompt = f"""
                        问题: {message}

                        你之前的回答:
                        {accumulated_response}

                        {tools_context}

                        请继续你的回答，使用工具返回的信息。如果需要更多信息，可以继续使用工具。
                        继续回答:
                        """
                     
                     if len(history) > 0 and history[-1]["role"] == "assistant":
                           history[-1]["content"] = full_response
                     else:
                           history.append({"role": "assistant", "content": full_response})
                     
                     yield history
                     
                     try:
                           new_response_stream = generate_response(new_prompt)
                           
                           accumulated_response = ""
                           
                           response_stream = new_response_stream
                           continue
                     except Exception as e:
                           print(f"调用generate_response时发生错误: {e}")
                           error_message = f"\n\n<div style='color: red;'>调用工具时发生错误: {str(e)}</div>\n\n"
                           full_response += error_message
                           accumulated_response += error_message
         
         # Process think tags
         if found_think_tag:
               answer_part += chunk_text
               thinking_markdown = f"<div style='background-color: #f0f0f0; padding: 10px;'>{thinking_part}</div>"
               
               if len(history) > 0 and history[-1]["role"] == "assistant":
                  history[-1]["content"] = thinking_markdown + "\n" + answer_part
               else:
                  history.append({"role": "assistant", "content": thinking_markdown + "\n" + answer_part})
         else:
               if "</think>" in full_response:
                  found_think_tag = True
                  parts = full_response.split("</think>", 1)
                  thinking_part = parts[0]
                  answer_part = parts[1] if len(parts) > 1 else ""
                  
                  thinking_markdown = f"<div style='background-color: #f0f0f0; padding: 10px;'>{thinking_part}</div>"
                  
                  if len(history) > 0 and history[-1]["role"] == "assistant":
                     history[-1]["content"] = thinking_markdown + "\n" + answer_part
                  else:
                     history.append({"role": "assistant", "content": thinking_markdown + "\n" + answer_part})
               else:
                  thinking_part = full_response
                  
                  if len(history) > 0 and history[-1]["role"] == "assistant":
                     # history[-1]["content"] = f"<div style='background-color: #f0f0f0; padding: 10px;'>{thinking_part}-</div>\n<div>思考中...</div>"
                     history[-1]["content"] = f"{thinking_part}"
                  else:
                     history.append({"role": "assistant", "content": f"<div style='background-color: #f0f0f0; padding: 10px;'>{thinking_part}</div>\n<div>思考中...</div>"})
         yield history
      
   except Exception as e:
      print(f"处理响应流时发生错误: {e}")
      error_message = f"<div style='color: red;'>处理响应时发生错误: {str(e)}</div>"
      
      if len(history) > 0 and history[-1]["role"] == "assistant":
         history[-1]["content"] = error_message
      else:
         history.append({"role": "assistant", "content": error_message})
      
      yield history
   
   return history
# 修改respond函数使用工具字典
def respond(message, image, history):
   # 检测是否需要工具
   print(3333, history)
   requires_tools = False
   
   # 根据关键词检查是否需要工具
   for tool_name, tool_info in AVAILABLE_TOOLS.items():
      if any(keyword in message for keyword in tool_info["keywords"]):
         requires_tools = True
         break
   
   # 如果上传了图片，自动允许图像分析工具
   if image is not None:
      requires_tools = True
   
   # 准备工具信息
   tools_info = ""
   if requires_tools:
      tools_info = "\n".join([
         f"{i+1}. {tool_name}: {tool_info['description']}"
         for i, (tool_name, tool_info) in enumerate(AVAILABLE_TOOLS.items())
      ])
   
   # 构建初始提示
   prompt = f"""
      问题: {message}

      请基于你的知识回答这个问题。如果需要更多信息，可以使用可用工具获取。
      回答: <think>
      """

   # 获取初始响应
   response_stream = generate_response(prompt, requires_tools, tools_info)
   
   # 处理响应流
   return process_response_stream(response_stream, message, image, history)

# 用户发送消息处理函数
# 修改用户和机器人处理函数以确保正确的格式
def user(user_message, image, history):
   print(history)
   # 确保历史是有效的列表
   if history is None:
      history = []
   # 确保添加正确格式的消息
   return "", image, history + [{"role": "user", "content": user_message}]

def bot(user_message, image, history):
   print("2222", history)
   # 确保历史是有效的列表
   if history is None:
      history = []
   # 确保添加正确格式的消息
   history = history + [{"role": "user", "content": user_message}]
   
   # 获取最后一条用户消息
   last_user_message = None
   for msg in reversed(history):
      if msg.get("role") == "user":
         last_user_message = msg.get("content", "")
         break
   
   # 如果没有找到用户消息，返回错误
   if last_user_message is None:
      return history + [{"role": "assistant", "content": "无法找到用户消息。"}]
   
   # 处理响应
   return respond(last_user_message, image, history)
# 创建Gradio界面
with gr.Blocks(title="基于知识图谱和图像的智能问答系统") as iface:
   gr.Markdown("# 基于知识图谱和图像的智能问答系统")
   gr.Markdown("本系统使用实体识别、知识图谱和图像分析增强问答效果，支持多工具调用")
   
   with gr.Row():
      with gr.Column(scale=3):
         chatbot = gr.Chatbot(
               height=500,
               show_label=False,
               render_markdown=True,
               type="messages",  # 使用更新的消息格式避免弃用警告
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
               ["这张图片显示的是什么疾病?", None],  # 这个例子需要一张图片
               ["看一下这张医学图像，并查询相关疾病的治疗方法", None]  # 测试多工具调用
         ],
         inputs=[msg, image_input],
         label="示例问题",
      )
   
   # 设置提交按钮点击事件
   submit_btn.click(
      bot,
      inputs=[msg,image_input, chatbot],
      outputs=[chatbot],
   )
   
   # 设置输入框提交事件
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