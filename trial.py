import streamlit as st
from utils import prepare_retirever, prepare_llm_json, prepare_llm, generate_response, check_hallucination, answer_grader, prose_writer, tidy_answer, stringize_answer
from zhconv import convert
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def init():
    retriever = prepare_retirever()
    MODEL_ID = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    #llm_json_mode = prepare_llm_json()
    #llm = prepare_llm()

    #return retriever, llm_json_mode, llm
    return retriever, model

#retriever, llm_json_mode, llm = init()
retriever, model = init()

st.title('Chat with your documents')

prompt = st.text_area('請把問題寫在下列空格',)

if st.button('詢問'):
    if prompt:
        with st.spinner('Generating...'):
            response, docs_txt = generate_response(retriever, model, prompt)
            with st.expander('相關文件'):
                st.write(docs_txt)
            st.write(response)
            '''
            hallucination_score_list = check_hallucination(llm_json_mode, docs_txt, response)['binary_score']
            point_list = tidy_answer(response, hallucination_score_list)
            #usage_score_list = answer_grader(prompt, llm_json_mode, point_list)
            #graded_point_list = tidy_answer(point_list, usage_score_list)
            if point_list:
                with st.expander('點列形式'):
                    st.write(stringize_answer(point_list))
                prose = prose_writer(llm, point_list)
                st.write(convert(prose,'zh-hk'))
            else:
                st.write('文件似乎沒有相關內容。你可以嘗試以其他方式發問。')
            '''