from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification, BertTokenizerFast
import torch
import warnings
import re
import os
import sys
import time
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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Janus模型加载
model_path = "/home/lzy123/deepseek/models/Janus-Pro-7B"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt = AutoModelForCausalLM.from_pretrained(
   model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# Ollama LLM配置
llm = Ollama(
    model="qwq:32b",
    # base_url="http://127.0.0.1:8000"
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

# 生成响应函数
def generate_response(prompt, requires_tools=False, tools_info=None):
    # 如果需要工具调用，修改提示以包含工具信息
    try:
        # 返回流式生成的迭代器
        response = llm.invoke(prompt, preserve_format=True, stream=True)
        return response
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

# 清除控制台输出的函数
def clear_console():
    # 根据操作系统使用不同的清屏命令
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Linux, macOS, etc.
        os.system('clear')

# 修改后的主函数，直接处理输入和输出，完全迭代式处理工具调用
# 修改后的主函数，直接处理输入和输出，保留并累积工具调用结果
# 简化版函数，只使用一个持续累积的提示
def process_input(message, image_path=None):
    # 加载图像（如果提供）
    image = None
    if image_path and os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            print(f"已加载图像: {image_path}")
        except Exception as e:
            print(f"加载图像失败: {e}")
    
    # 准备工具信息
    tools_info = "\n".join([
        f"{i+1}. {tool_name}: {tool_info['description']}"
        for i, (tool_name, tool_info) in enumerate(AVAILABLE_TOOLS.items())
    ])
    
    # 构建初始提示 - 这个提示将被持续使用
    initial_prompt = rf"""
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

重要提示: 为了减少回答中的幻觉，请使用知识图谱工具查询相关事实和知识，然后再基于查询结果回答问题。
如果知识图谱中没有找到相关信息，再基于你自己的知识回答。

如果你觉得要使用工具，请你立马使用，不要等到回答的时候再使用。

请你保证在<think>使用工具<\think>

禁止模拟工具输出，你只需指明要使用的工具。在工具执行后，你将收到工具返回的结果。
你可以根据需要多次使用工具，也可以根据前一个工具的结果决定是否使用下一个工具。
问题: {message}

回答:
<think>
"""
    
    
    # 当前累积的提示（只有一个提示，持续追加内容）
    current_prompt = initial_prompt
    
    # 工具调用计数
    tool_execution_counter = 0
    max_tool_executions = 5
    
    # 当前响应的累积文本（用于显示给用户）
    displayed_response = ""
    
    print("\n用户: " + message)
    print("\n系统处理中...\n")
    
    # 主循环
    while True:
        # 获取响应流 - 使用当前累积的提示
        response_stream = generate_response(current_prompt)
        
        # 当前批次的响应
        current_batch = ""
        
        # 处理流式响应
        for chunk in response_stream:
            # 获取文本内容
            chunk_text = ""
            if hasattr(chunk, 'content'):
                chunk_text = chunk.content
            elif isinstance(chunk, str):
                chunk_text = chunk
            elif isinstance(chunk, list):
                chunk_text = str(chunk[0]) if len(chunk) > 0 else ""
            else:
                chunk_text = str(chunk)
            # 添加到当前批次
            current_batch += chunk_text
            
            # 更新显示响应
            displayed_response = displayed_response + chunk_text
            
            # 清除控制台并实时显示
            clear_console()
            print("\n用户: " + message)
            print("\n系统: " + displayed_response)
            sys.stdout.flush()

            if "</input>" in current_batch:
                break
            
        
        # 将当前批次添加到累积提示
        current_prompt += current_batch
        
        # 检查工具调用
        tool_pattern = r"<tool>(.*?)</tool>\s*<input>(.*?)</input>"
        tool_matches = list(re.finditer(tool_pattern, current_batch))
        
        # 如果没有工具调用或已达到最大次数，结束循环
        if not tool_matches:
            break
        
        if tool_execution_counter >= max_tool_executions:
            # 添加最大调用次数提示
            max_limit_notice = f"\n\n注意：已达到最大工具调用次数 ({max_tool_executions})。\n\n"
            current_prompt += max_limit_notice
            displayed_response += max_limit_notice
            
            # 更新显示
            clear_console()
            print("\n用户: " + message)
            print("\n系统: " + displayed_response)
            break
        
        # 处理工具调用（仅处理第一个，其余留给下一轮）
        match = tool_matches[0]
        tool_name_match = re.search(r"<tool>(.*?)</tool>", match.group(0))
        tool_input_match = re.search(r"<input>(.*?)</input>", match.group(0))
        
        if tool_name_match and tool_input_match:
            tool_name = tool_name_match.group(1).strip()
            tool_input = tool_input_match.group(1).strip()
            
            print(f"\n执行工具调用: {tool_name}, 输入: {tool_input}")
            
            # 执行工具调用
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
            
            # 将工具结果直接添加到累积提示
            tool_result_text = f"\n工具结果:\n{tool_result}\n\n"
            current_prompt += tool_result_text + "请你基于上述的内容，继续思考，如果信息不足，请你调用其他的工具。\n"
            displayed_response += tool_result_text
            
            # 更新显示
            clear_console()
            print("\n用户: " + message)
            print("\n系统: " + displayed_response)
            
            tool_execution_counter += 1
        else:
            # 工具调用格式不正确，退出循环
            break
    
    # 完成
    print("\n处理完成")
def main():
    print("基于知识图谱和图像的智能问答系统")
    print("输入 'exit' 或 'quit' 退出系统")
    print("若要使用图像分析功能，请输入 '图片路径' 加<\image>后再输入问题")
    
    while True:
        user_input = input("\n请输入您的问题: ")
        
        # 检查退出命令
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("系统已退出。")
            break
        
        # 检查是否包含图像路径
        image_path = None
        if user_input.startswith('image:'):
            parts = user_input.split('<\\image>', 1)
            if len(parts) > 1:
                image_path = parts[0][6:]  # 去掉 'image:' 前缀
                user_input = parts[1]
                print(f"检测到图像路径: {image_path}")
            else:
                print("请按正确格式输入：'image:图片路径<\\image>问题'")
                continue
        
        # 处理输入
        process_input(user_input, image_path)

if __name__ == "__main__":
    main()