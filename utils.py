from pydantic import BaseModel, Field
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
        "blog20241027.txt"
    ]

    docs_list = []
    for file in file_list:
        with open(file_path + file, 'r') as f:
            content = f.read()
            doc = Document(page_content=content, file_name=file)
            docs_list.append(doc)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n",
        chunk_size = 20, 
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

    僅使用上述上下文回答該問題, 以列點的形式作答。

    返回的 JSON 應只有一個鍵值對, 鍵為"answer", "answer"的值應該是一個列表, 當中列表每一個值是字串, 對應每一點答案, 值的數量不限。
    """
    docs = retriever.invoke(question)
    docs_txt = "\n\n".join(doc.page_content for doc in docs)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm_json_mode.invoke([HumanMessage(content=rag_prompt_formatted)])
    return json.loads(generation.content), docs_txt

def check_hallucination(llm_json_mode, docs_txt, answer):
    hallucination_grader_instructions = """ 你是一位負責批改小測的老師。

    你將會收到「事實」(FACTS)和「學生答案」(STUDENT ANSWER)。

    以下是評分標準：
    確保學生答案以「事實」為基礎。
    確保學生答案未包含超出「事實」範圍的「臆測性」信息。

    評分：
    「是」:表示學生答案符合所有標準。這是最高(最佳)的評分。
    「否」:表示學生答案未符合所有標準。這是最低可能的評分。
    請以逐步推理的方式解釋你的評分理由，以確保你的判斷和結論正確。

    避免在一開始直接給出正確答案。 """

    hallucination_grader_prompt = """ 
    事實(FACTS):{documents}

    學生答案(STUDENT ANSWER):{generation}。

    返回的 JSON 應只有一個鍵值對: 鍵為“binary_score”, “binary_score”的值應該是一個列表, 當中列表每一個值是為”是”或”否”, 用於指示每一行學生答案是否符合標準, 列表的長度即是學生答案的行數, 在這裡, 即是{number}個值。
    """
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=docs_txt, generation=answer["answer"], number = len(answer["answer"])
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    return json.loads(result.content)

def answer_grader(prompt, llm_json_mode, answer):
    answer_grader_instructions = """ 
    你是一位負責批改小測的老師。

    你將會收到一個「問題」(QUESTION)和「學生答案」(STUDENT ANSWER)。

    以下是評分標準：
    確保學生答案有助於回答問題。

    評分：
    「是」:表示學生答案符合所有標準。這是最高(最佳)的評分。
    即使答案包含問題未明確要求的額外信息，但仍符合標準，可以給予「是」的評分。

    「否」:表示學生答案未符合所有標準。這是最低可能的評分。

    請以逐步推理的方式解釋你的評分理由，以確保你的判斷和結論正確。

    避免在一開始直接給出正確答案。 """

    answer_grader_prompt = """ 
    問題(QUESTION):

    {prompt}

    學生答案(STUDENT ANSWER):{answer}。

    返回的 JSON 應只有一個鍵值對: 鍵為“binary_score”, “binary_score”的值應該是一個列表, 當中列表每一個值是為”是”或”否”, 用於指示每一行學生答案是否符合標準, 列表的長度即是學生答案的行數, 在這裡, 即是{number}個值。
    """
    answer_grader_prompt_formatted = answer_grader_prompt.format(
        prompt=prompt, answer=answer, number = len(answer)
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=answer_grader_instructions)]
        + [HumanMessage(content=answer_grader_prompt_formatted)]
    )
    return json.loads(result.content)

def prose_writer(llm, point_string):
    prose_writer_instructions = """ 
    你是香港財政司司長陳茂波先生
    你負責寫作
    你將會收到點列形式的句子
    你的任務是把這些句子，組成一段文字
    這段文字應該是通順的，有邏輯的，並且符合文法    
    """

    prose_writer_prompt = """ 
    點列句子: {point_list}

    請以上列的句子為內容，開始寫作。
    """
    prose_writer_prompt_formatted = prose_writer_prompt.format(
        point_list = point_string
    )
    result = llm.invoke(
        [SystemMessage(content=prose_writer_instructions)]
        + [HumanMessage(content=prose_writer_prompt_formatted)]
    )
    return result.content

def tidy_answer(answer, hallucination_score_list):
    answer_list = []
    for i, score in enumerate(hallucination_score_list):
        if score == "是":
            answer_list.append(answer["answer"][i])
    return answer_list

def stringize_answer(answer):
    return "\n\n".join(answer)