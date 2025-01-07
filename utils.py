from typing import Optional
import os
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from zhconv import convert
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import json


def prepare_retirever():
    file_path = "data/"
    file_list = [
        "blog20241006.txt",
        "blog20241013.txt",
        "blog20241020.txt",
        "blog20241103.txt",
        "blog20241110.txt",
        "blog20241117.txt",
        "blog20241124.txt"
    ]

    docs_list = []
    for file in file_list:
        with open(file_path + file, 'r') as f:
            content = f.read()
            doc = Document(page_content=content, file_name=file)
            docs_list.append(doc)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        #separator = "\n",
        chunk_size = 100, 
        chunk_overlap = 20
    )

    chunk = text_splitter.split_documents(docs_list)
    vectorstore = SKLearnVectorStore.from_documents(
        documents = chunk,
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    
    retriever = vectorstore.as_retriever(search_kwargs = {"k": 10})
    return retriever

def prepare_llm_json():
    local_llm = "qwen2.5:3b"
    llm_json_mode = ChatOllama(model = local_llm, temperature=0, format = "json")
    return llm_json_mode

def prepare_llm():
    local_llm = "qwen2.5:3b"
    llm = ChatOllama(model = local_llm, temperature=0)
    return llm

def generate_response(retriever, llm_json_mode, question):
    rag_prompt = """ 你是香港財政司司長陳茂波先生
    
    你擅長於問答任務。

    以下是用來回答問題的上下文, 文章都是摘錄自香港財政司司長陳茂波先生的日記：

    {context}

    請仔細考慮上述上下文。

    現在，檢視用戶問題：

    {question}

    僅使用上述上下文回答該問題, 以列點的形式作答, 
    
    這些列點應盡量涵蓋文章的內容, 但要避免重複內容。

    返回的 JSON 應只有一個鍵值對, 鍵為"answer", "answer"的值應該是一個列表, 當中列表每一個值是字串, 對應每一點答案, 值的數量不限。

    """

    docs = retriever.invoke(question)
    docs_txt = "\n\n".join(doc.page_content for doc in docs)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    completion = llm_json_mode(system_message = "你是香港財政司司長陳茂波先生, 你擅長於問答任務。", human_message = rag_prompt_formatted)
    text = completion.choices[0].message.content
    if "json" in text:
        result = json.loads(text[7:-3].strip())["answer"]
    else:
        result = json.loads(text)["answer"] 
    return result, docs_txt

def check_hallucination(llm_json_mode, docs_txt, answer):
    hallucination_grader_instructions = """ 你是一位負責批改小測的老師。

    你將會收到「事實」(FACTS)和「學生答案」(STUDENT ANSWER)。

    以下是評分標準：
    確保學生答案以「事實」為基礎。
    確保學生答案未包含超出「事實」範圍的「臆測性」信息。

    評分：
    「是」:表示學生答案符合所有標準。這是最高(最佳)的評分。
    「否」:表示學生答案未符合所有標準。這是最低可能的評分。

    避免在一開始直接給出正確答案。 """

    hallucination_grader_prompt = """ 
    事實(FACTS):{documents}

    學生答案(STUDENT ANSWER):{generation}。

    返回的 JSON 應只有個鍵值對: 鍵為“binary_score”, “binary_score”的值應該是一個列表, 當中列表每一個值是為”是”或”否”, 用於指示每一行學生答案是否符合標準, 列表的長度即是學生答案的行數, 在這裡, 即是{number}個值。

    除此以外不必返回其他文字
    """
    #    請以逐步推理的方式解釋你的評分理由，以確保你的判斷和結論正確。
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=docs_txt, generation=answer, number = len(answer)
    )
    completion = llm_json_mode(
        system_message = hallucination_grader_instructions,
        human_message = hallucination_grader_prompt_formatted
    )
    text = completion.choices[0].message.content
    if "json" in text:
        result = json.loads(text[7:-3].strip())["binary_score"]
    else:
        result = json.loads(text)["binary_score"]
    return result

def answer_grader(llm_json_mode, prompt, answer):
    answer_grader_instructions = """ 
    你是一位負責批改小測的老師。

    你將會收到一個「問題」(QUESTION)和「學生答案」(STUDENT ANSWER)。

    以下是評分標準：
    確保學生答案有助於回答問題。

    評分：
    「是」:表示學生答案符合所有標準。這是最高(最佳)的評分。
    即使答案包含問題未明確要求的額外信息，但仍符合標準，可以給予「是」的評分。

    「否」:表示學生答案未符合所有標準。這是最低可能的評分。

    避免在一開始直接給出正確答案。 """
    #     請以逐步推理的方式解釋你的評分理由，以確保你的判斷和結論正確。
    answer_grader_prompt = """ 
    問題(QUESTION):

    {prompt}

    學生答案(STUDENT ANSWER):{answer}。

    返回的 JSON 應只有一個鍵值對: 鍵為“binary_score”, “binary_score”的值應該是一個列表, 當中列表每一個值是為”是”或”否”, 用於指示每一行學生答案是否符合標準, 列表的長度即是學生答案的行數, 在這裡, 即是{number}個值。
    """
    answer_grader_prompt_formatted = answer_grader_prompt.format(
        prompt=prompt, answer=answer, number = len(answer)
    )
    completion = llm_json_mode(
        system_message = answer_grader_instructions,
        human_message = answer_grader_prompt_formatted
    )
    text = completion.choices[0].message.content
    if "json" in text:
        result = json.loads(text[7:-3].strip())["binary_score"]
    else:
        result = json.loads(text)["binary_score"]
    return result

def prose_writer(llm, point_string):
    prose_writer_instructions = """ 
    你是香港財政司司長陳茂波先生
    你負責寫作
    你將會收到點列形式的句子
    你的任務是把這些句子，組成文字
    文字應該是通順的，有邏輯的，並且符合文法   
    """

    prose_writer_prompt = """ 
    點列句子: {point_list}

    請以上列的句子為內容，開始寫作。
    """
    prose_writer_prompt_formatted = prose_writer_prompt.format(
        point_list = point_string
    )
    completion = llm(
        system_message = prose_writer_instructions,
        human_message = prose_writer_prompt_formatted
    )
    result = completion.choices[0].message.content
    print(type(result))
    return result

def tidy_answer(answer, hallucination_score_list, usage_score_list):
    answer_list = []
    for i in range(len(hallucination_score_list)):
        if hallucination_score_list[i] == "是" and usage_score_list[i] == "是":
            answer_list.append(answer[i])
    return answer_list

def stringize_answer(answer):
    return "\n\n".join(answer)
