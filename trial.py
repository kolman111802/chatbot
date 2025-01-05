import streamlit as st
from utils import prepare_retirever, prepare_llm_json, prepare_llm, generate_response, check_hallucination, answer_grader, prose_writer, tidy_answer, stringize_answer
from zhconv import convert

@st.cache_resource
def init():
    retriever = prepare_retirever()
    llm_json_mode = prepare_llm_json()
    llm = prepare_llm()
    return retriever, llm_json_mode, llm

retriever, llm_json_mode, llm = init()

st.title('Chat with your documents')

prompt = st.text_area('請把問題寫在下列空格',)

if st.button('詢問'):
    if prompt:
        with st.spinner('Generating...'):
            response, docs_txt = generate_response(retriever, llm_json_mode, prompt)
            with st.expander('相關文件'):
                st.write(docs_txt)
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
