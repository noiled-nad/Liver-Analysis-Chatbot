from langchain_neo4j import Neo4jGraph  # 新的导入方式
from langchain_community.llms import Ollama
from transformers import AutoModelForTokenClassification, BertTokenizerFast
import torch
import gradio as gr

# 使用 Ollama 模型（注意新版 Ollama 已弃用旧接口）
llm = Ollama(
    model="deepseek-r1:32b",
    base_url="http://127.0.0.1:8000"
)

def generate_response(prompt):
    # return llm.invoke(prompt, preserve_format=True)
    response_stream = llm.invoke(prompt, preserve_format=True, stream=True)
    for chunk in response_stream:
        # 每次yield返回一个chunk（部分内容）
        yield chunk

# 加载 NER 模型和 tokenizer
ner_tokenizer = BertTokenizerFast.from_pretrained('/home/qyj/code/KG/model/bert-base-chinese-medical-ner')
ner_model = AutoModelForTokenClassification.from_pretrained("/home/qyj/code/KG/model/bert-base-chinese-medical-ner")

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
            if flag and pred_idx not in (2, 3):
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

def extract_entities(text):
    # 1. 提取实体
    inputs = ner_tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)
    outputs = ner_model(**inputs)
    outputs = outputs.logits.argmax(-1) * inputs['attention_mask']
    s = format_outputs([text], outputs)[0]
    # 2. 提取关键词
    keywords = []
    for token in s:
        if token['word'] not in keywords and token['word'] != "":
            keywords.append(token['word'])
    return keywords

# Neo4j 连接
graph = Neo4jGraph(
    url="neo4j+s://9a72e3e0.databases.neo4j.io",
    username="neo4j",
    password="wBykwkeh9HdyspnuRAX-1DJ789CKbrDB9W4cjlRJy8U",
)

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

# def respond(message, history):
#     # history 为对话记录列表，每个元素为 (用户消息, 回复) 的元组
#     history = history or []
#     entities = extract_entities(message)
#     print(f"识别出的实体: {entities}")
    
#     knowledge = query_knowledge(entities, message)
#     context = format_knowledge(knowledge)
#     print(f"查询到的知识: {context}")
    
#     prompt = f"""基于以下知识：{context}
# 问题：{message}
# 请基于上述知识和自己的知识回答这个问题。如果知识库中没有相关信息，请说明无法回答，并尝试通过自己的知识回答！
# 回答：<think>"""
    
#     full_response = generate_response(prompt)
#     print(f"生成的回答: {full_response}")
    
#     # 分离思考过程和回答（如果包含 "</think>" 标签）
#     if "</think>" in full_response:
#         thinking_part = full_response.split("</think>")[0]
#         answer_part = full_response.split("</think>")[1]
#         final_response = f"{thinking_part}\n{answer_part}"
#     else:
#         final_response = full_response

#     history.append((message, final_response))
#     # 返回的两个输出：一个用于展示对话内容，一个用于更新状态（这两个值相同）
#     return history, history


def respond(message, history):
    history = history or []
    # 根据用户输入抽取实体和知识
    entities = extract_entities(message)
    knowledge = query_knowledge(entities, message)
    context = format_knowledge(knowledge)
    print(f"查询到的知识: {context}")
    prompt = f"""基于以下知识：{context}
问题：{message}
请基于上述知识和自己的知识回答这个问题。如果知识库中没有相关信息，请说明无法回答，并尝试通过自己的知识回答！
回答：<think>"""
    
    partial_response = ""
    # 遍历生成器，每获取一个chunk就更新partial_response
    for chunk in generate_response(prompt):
        partial_response += chunk
        if history:
            history[-1] = (message, partial_response)
        else:
            history.append((message, partial_response))
        # yield更新后的对话记录，实现流式输出
        yield history, history



# # 构建 Gradio Blocks 结构（确保 state 仅出现一次）
# with gr.Blocks() as demo:
#     gr.Markdown("# 基于知识图谱的智能问答系统")
#     chatbot = gr.Chatbot(label="对话历史")
#     state = gr.State([])  # 用于保存对话状态
#     with gr.Row():
#         msg = gr.Textbox(label="输入您的问题", placeholder="请输入您的问题……")
#         send_btn = gr.Button("发送")
#     # 设置事件：点击按钮或回车时调用 respond 函数
#     send_btn.click(respond, inputs=[msg, state], outputs=[chatbot, state])
#     msg.submit(respond, inputs=[msg, state], outputs=[chatbot, state])


with gr.Blocks() as demo:
    gr.Markdown("# 基于知识图谱的智能问答系统")
    chatbot = gr.Chatbot(label="对话历史")
    state = gr.State([])  # 用于保存对话状态
    with gr.Row():
        msg = gr.Textbox(label="输入您的问题", placeholder="请输入您的问题……")
        send_btn = gr.Button("发送")
    send_btn.click(respond, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(respond, inputs=[msg, state], outputs=[chatbot, state])

if __name__ == "__main__":
    demo.launch(share=True)