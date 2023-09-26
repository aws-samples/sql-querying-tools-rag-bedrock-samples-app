# Importing important package 

import streamlit as st 
import text_to_sql_rag_lib as glib 

st.set_page_config(page_title="Sql Querying Tools RAG Bedrock samples App") #HTML title
st.title("Sql Querying Tools RAG Bedrock samples App") #page title

if 'vector_index' not in st.session_state: 
    with st.spinner("Indexing document..."): 
        st.session_state.vector_index = glib.create_get_index() 

input_text = st.text_area("Input text", label_visibility="collapsed") 
go_button = st.button("Go", type="primary") 

if go_button: 
    
    with st.spinner("Evaluating..."): 
        response_content = glib.call_rag_function(index=st.session_state.vector_index, input_text=input_text) #call the model through the supporting library
        
        st.write(response_content) 


