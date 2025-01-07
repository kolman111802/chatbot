import streamlit as st
from utils import prepare_retirever, prepare_llm_json, prepare_llm, generate_response, check_hallucination, answer_grader, prose_writer, tidy_answer, stringize_answer
from zhconv import convert
from connect import inference

@st.cache_resource
def init():
    retriever = prepare_retirever()
    return retriever

retriever = init()

#st.title('Chat with your documents')

col1, col2 = st.columns(2)

with col1:
    prompt = st.text_area('請把問題寫在下列空格', key='prompt')
    ask_button = st.button('詢問')

if ask_button:
    if prompt:
        with st.spinner('Generating...'):
            response, docs_txt = generate_response(retriever, inference(), prompt)
            with col1:
                with st.expander('相關文件'):
                    st.write(docs_txt)
                with st.expander('回答'):
                    st.write(response)
                hallucination_score_list = check_hallucination(inference(), docs_txt, response)
                with st.expander('是否基於上載文件'):
                    st.write(hallucination_score_list)
                #point_list = tidy_answer(response, hallucination_score_list)
                usage_score_list = answer_grader(inference(), prompt, response)
                with st.expander('是否符合問題要求'):
                    st.write(usage_score_list)
            point_list = tidy_answer(response, hallucination_score_list, usage_score_list)
            if point_list:
                with col1:
                    with st.expander('點列形式'):
                        st.write(convert(stringize_answer(point_list), 'zh-hk'))
                prose = prose_writer(inference(), point_list)
                with col2:
                    st.write(convert(prose,'zh-hk'))
            else:
                with col2:
                    st.write('文件似乎沒有相關內容。你可以嘗試以其他方式發問。')
    else:
        st.write('請輸入問題。')