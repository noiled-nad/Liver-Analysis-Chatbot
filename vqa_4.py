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

# 生成响应函数
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

# 替换工具调用为结果
def replace_tool_calls_with_results(text, results):
    """Replace tool calls in text with results"""
    modified_text = text
    
    tool_results_text = ""
    for result in results:
        tool_text = f"""
工具调用: {result['name']}
输入: {result['input']}
结果: 
{result['result']}
"""
        tool_results_text += tool_text
    
    tool_pattern = r"<tool>.*?</tool>\s*<input>.*?</input>"
    match = re.search(tool_pattern, modified_text)
    if match:
        start, end = match.span()
        modified_text = modified_text[:end] + "\n\n" + tool_results_text + "\n\n" + modified_text[end:]
    
    return modified_text

# 清除控制台输出的函数
def clear_console():
    # 根据操作系统使用不同的清屏命令
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Linux, macOS, etc.
        os.system('clear')

# 修改后的主函数，直接处理输入和输出
def process_input(message, image_path=None):
    # 加载图像（如果提供）
    image = None
    if image_path and os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            print(f"已加载图像: {image_path}")
        except Exception as e:
            print(f"加载图像失败: {e}")
    
    # 检测是否需要工具
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
    full_response = ""
    accumulated_response = ""
    pending_tool_calls = []
    tool_execution_counter = 0
    max_tool_executions = 5
    
    # 用于跟踪工具调用状态
    processed_tool_calls = set()
    processing_completed = False
    
    print("\n用户: " + message)
    print("\n系统处理中...\n")
    
    try:
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
            
            # 添加到累积响应
            accumulated_response += chunk_text
            full_response = accumulated_response
            
            # 清除控制台并重新打印
            clear_console()
            print("\n用户: " + message)
            print("\n系统: " + full_response)
            sys.stdout.flush()  # 确保输出立即显示
            
            # 只有在未处理完成时才检查和处理工具调用
            if not processing_completed:
                # 检查新的工具调用
                tool_pattern = r"<tool>(.*?)</tool>\s*<input>(.*?)</input>"
                tool_matches = list(re.finditer(tool_pattern, accumulated_response))
                
                for match in tool_matches:
                    # 提取工具调用信息
                    tool_call = accumulated_response[match.start():match.end()]
                    
                    # 检查是否已处理过此工具调用
                    if tool_call in processed_tool_calls:
                        continue
                    
                    # 标记为已处理
                    processed_tool_calls.add(tool_call)
                    
                    # 检查是否达到最大工具调用次数
                    if tool_execution_counter >= max_tool_executions:
                        limit_notice = f"\n\n注意：已达到最大工具调用次数 ({max_tool_executions})。\n\n"
                        if limit_notice not in full_response:
                            full_response += limit_notice
                            accumulated_response += limit_notice
                            
                            # 清除控制台并重新打印
                            clear_console()
                            print("\n用户: " + message)
                            print("\n系统: " + full_response)
                        break
                    
                    # 提取工具名称和输入
                    tool_name_match = re.search(r"<tool>(.*?)</tool>", tool_call)
                    tool_input_match = re.search(r"<input>(.*?)</input>", tool_call)
                    
                    if tool_name_match and tool_input_match:
                        tool_name = tool_name_match.group(1).strip()
                        tool_input = tool_input_match.group(1).strip()
                        
                        print(f"执行工具调用: {tool_name}, 输入: {tool_input}")
                        
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
                        
                        tool_display = {
                            "name": tool_name,
                            "input": tool_input,
                            "result": tool_result
                        }
                        
                        pending_tool_calls.append(tool_display)
                        tool_execution_counter += 1
                        
                        # 构建工具结果文本
                        tool_result_text = f"""
工具调用: {tool_name}
输入: {tool_input}
结果: 
{tool_result}
"""
                        
                        # 替换工具调用为结果
                        match_start, match_end = match.span()
                        full_response = full_response[:match_end] + "\n\n" + tool_result_text + "\n\n" + full_response[match_end:]
                        
                        # 清除控制台并重新打印
                        clear_console()
                        print("\n用户: " + message)
                        print("\n系统: " + full_response)
                        
                        # 设置处理完成标志，停止当前响应流
                        processing_completed = True
                        break  # 一次只处理一个工具调用，然后重新开始
                
                # 如果已处理完一个工具调用，重新生成响应
                if processing_completed:
                    # 生成新的提示和上下文
                    tools_context = ""
                    for tool in pending_tool_calls:
                        tools_context += f"\n工具 \"{tool['name']}\" 返回的结果 (针对输入: {tool['input']}):\n{tool['result']}\n"
                    
                    new_prompt = f"""
                        问题: {message}

                        你之前的回答:
                        {full_response}

                        {tools_context}

                        请继续你的回答，使用工具返回的信息。如果需要更多信息，可以继续使用工具。
                        继续回答:
                        """
                    
                    try:
                        # 重置处理状态，准备处理新的响应
                        processing_completed = False
                        accumulated_response = full_response  # 保留之前的响应
                        
                        # 获取新的响应流
                        response_stream = generate_response(new_prompt)
                        print("\n继续生成回复中...")
                        break  # 跳出当前循环，开始处理新的响应流
                    except Exception as e:
                        error_message = f"\n\n调用工具时发生错误: {str(e)}\n\n"
                        full_response += error_message
                        accumulated_response += error_message
                        
                        # 清除控制台并重新打印
                        clear_console()
                        print("\n用户: " + message)
                        print("\n系统: " + full_response)
            
            
    except Exception as e:
        error_message = f"处理响应时发生错误: {str(e)}"
        
        # 清除控制台并重新打印
        clear_console()
        print("\n用户: " + message)
        print("\n系统: " + full_response + "\n\n" + error_message)
    
    # 完成后打印最终结果
    clear_console()
    print("\n用户: " + message)
    print("\n系统: " + full_response)
    
    # 检查响应是否看起来不完整
    if full_response.endswith("结果: ") or "工具调用" in full_response and not re.search(r'根据(图像分析|知识图谱)(结果|显示|内容)', full_response):
        print("\n注意：响应可能不完整。系统将尝试继续生成...")
        
        # 生成后续回复的提示
        continue_prompt = f"""
        问题: {message}
        
        你之前的工具调用结果:
        {full_response}
        
        请根据以上工具返回的结果，继续提供完整的分析。注意要详细解释图像或查询结果表明了什么情况，以及可能的医学建议。
        继续回答:
        """
        
        try:
            # 获取新的响应
            continue_response = generate_response(continue_prompt)
            continue_text = ""
            
            # 处理继续的响应
            for chunk in continue_response:
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
                
                continue_text += chunk_text
                
                # 清除控制台并重新打印
                clear_console()
                print("\n用户: " + message)
                print("\n系统: " + full_response + "\n\n继续分析:\n" + continue_text)
                sys.stdout.flush()
                
                # 小延迟，让输出看起来更流畅
            
            print("\n分析完成")
        except Exception as e:
            print(f"\n尝试继续生成时出错: {str(e)}")
    else:
        print("\n处理完成")

# 简单的命令行交互界面
def main():
    print("基于知识图谱和图像的智能问答系统")
    print("输入 'exit' 或 'quit' 退出系统")
    print("若要使用图像分析功能，请输入 'image:图片路径' 加<\\image>后再输入问题")
    
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