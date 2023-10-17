import streamlit as st
from streamlit_chat import message
from textractcaller.t_call import call_textract, Textract_Features
import boto3
import botocore
import json
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationalRetrievalChain

from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import botocore.exceptions

s3 = boto3.client('s3')
textract_client = boto3.client('textract')

if 'S3_BUCKET_NAME' not in os.environ:
    bucket_name = 'textract-eample-ad-cs'
else:
    bucket_name = os.environ['S3_BUCKET_NAME']

def upload_file(file):
    try:
        foldername = file.name.split('.')[0]
        s3.upload_fileobj(
            Fileobj=file,
            Bucket=bucket_name,
            Key=(foldername + "/" + file.name)
        )
        return True
    except Exception as e:
        print(e)
        return False

def display_pdf_content(file_name):
    # st.markdown('<u>Uploaded File</u>', unsafe_allow_html=True)
    import base64
    # Opening file from file path
    foldername = file_name.split('.')[0]
    obj = s3.get_object(Bucket=bucket_name, Key=(foldername + "/" + file_name))
    base64_pdf = base64.b64encode(obj['Body'].read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="700" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                    if word['BlockType'] == 'SELECTION_ELEMENT':
                        if word['SelectionStatus'] == 'SELECTED':
                            text += 'X '

    return text

def get_rows_columns_map(table_result, blocks_map):
    rows = {}
    for relationship in table_result['Relationships']:
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'CELL':
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    if row_index not in rows:
                        rows[row_index] = {}
                    rows[row_index][col_index] = get_text(cell, blocks_map)
    return rows

def get_tables_fromJSON(response):
    AllTables = []
    if 'Blocks' in response:
        blocks = response['Blocks']
        blocks_map = {}
        table_blocks = []
        for block in blocks:
            blocks_map[block['Id']] = block
            if block['BlockType'] == "TABLE":
                table_blocks.append(block)

        if len(table_blocks) <= 0:
            print("<b> NO Table FOUND </b>")

        for table_result in table_blocks:
            tableMatrix = []
            rows = get_rows_columns_map(table_result, blocks_map)
            for row_index, cols in rows.items():
                thisRow = []
                for col_index, text in cols.items():
                    thisRow.append(text)
                tableMatrix.append(thisRow)
            AllTables.append(tableMatrix)
    return AllTables

def generate_insights(table_info=[]):
    bedrock_client = boto3.client('bedrock-runtime')
    body = json.dumps({
        "prompt": "\n\nHuman:Explain the details in " + str(table_info) + "\n\nAssistant:",
        "max_tokens_to_sample": 300,
        "temperature": 0.1,
        "top_p": 0.9,
    })
    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    return response_body['completion']

def get_rag_chat_response(input_text, memory, index):  # chat client function
    llm = get_llm()
    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, index.vectorstore.as_retriever(),
                                                                        memory=memory)
    chat_response = conversation_with_retrieval(
        {"question": "\n\nHuman:Explain the details in " + input_text + "\n\nAssistant:"})  # pass the user message, history, and knowledge to the model
    # print(chat_response)
    return chat_response['answer']

def display_generated_insights(file_name):
    foldername = file_name.split('.')[0]
    response = call_textract(input_document=('s3://'+bucket_name+'/'+foldername+'/'+file_name),
                            features=[Textract_Features.TABLES],
                            boto3_textract_client=textract_client)

    parseformTables = get_tables_fromJSON(response)
    parseformTables = [[[s.rstrip() for s in row] for row in table] for table in parseformTables]
    table_info = []
    for tables in parseformTables:
        for table in tables:
            info = str(table).replace('[', '').replace(']', '')
            table_info.append(info)
    return get_rag_chat_response(str(table_info), get_memory(), build_index_from_string(table_info))
    # st.write(generate_insights(table_info)

def get_llm():
    model_kwargs = {  # Claude-v2
        "max_tokens_to_sample": 1024,
        "temperature": 0.1,
        "top_p": 0.9
    }

    llm = Bedrock(
        model_id="anthropic.claude-v2",  # set the foundation model
        model_kwargs=model_kwargs)  # configure the properties for Claude
    return llm

def get_memory():  # create memory for this chat session
    memory = ConversationBufferWindowMemory(memory_key="chat_history",
                                            return_messages=True)  # Maintains a history of previous messages
    return memory

def build_index_from_string(table_info=[]): #creates and returns an in-memory vector store to be used in the application
    from datetime import datetime
    filename = 'tmp-' + datetime.today().strftime('%Y%m%d%H%M%S') + '.txt'
    f = open(filename, 'w')
    for tab in table_info:
        f.write(tab + '\n')
    f.flush()
    f.close()

    embeddings = BedrockEmbeddings(
        region_name='us-west-2',
        endpoint_url='https://prod.us-west-2.dataplane.bedrock.aws.dev'
    )  # create a Titan Embeddings client

    text_splitter = RecursiveCharacterTextSplitter(  # create a text splitter
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,  # divide into 1000-character chunks using the separators above
        chunk_overlap=100  # number of characters that can overlap with previous chunk
    )

    index_creator = VectorstoreIndexCreator(  # create a vector store factory
        vectorstore_cls=FAISS,  # use an in-memory vector store for demo purposes
        embedding=embeddings,  # use Titan embeddings
        text_splitter=text_splitter,  # use the recursive text splitter
    )
    from langchain.document_loaders import TextLoader
    loader = TextLoader(filename)
    index_from_loader = index_creator.from_loaders([loader])
    # print(index_from_loader)
    if os.path.exists(filename):
        os.remove(filename)
    return index_from_loader

def generate_embedding(table_info=[]):
    bedrock = boto3.client(service_name='bedrock-runtime',
                           region_name='us-west-2',
                           endpoint_url='https://prod.us-west-2.dataplane.bedrock.aws.dev')
    body = json.dumps({
        "inputText": str(table_info)
    })

    modelId = 'amazon.titan-embed-text-v1'  # change this to use a different version from the model provider
    accept = 'application/json'
    contentType = 'application/json'
    embedding = ''
    try:

        response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read())
        embedding = response_body.get('embedding')
        # print(embedding)

    except botocore.exceptions.ClientError as error:

        if error.response['Error']['Code'] == 'AccessDeniedException':
            embedding=f"\x1b[41m{error.response['Error']['Message']}\
                    \nTo troubeshoot this issue please refer to the following resources.\
                     \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                     \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n"

        else:
            raise error
    return embedding

def extract_table_data_from_pdf(file_name):
    # st.markdown('<u>Embeddings</u>', unsafe_allow_html=True)
    textract = boto3.client('textract')
    foldername = file_name.split('.')[0]
    response = call_textract(input_document=('s3://' + bucket_name + '/' + foldername + '/' + file_name),
                             features=[Textract_Features.TABLES],
                             boto3_textract_client=textract_client)
    parseformTables = get_tables_fromJSON(response)
    parseformTables = [[[s.rstrip() for s in row] for row in table] for table in parseformTables]
    table_info = []
    for tables in parseformTables:
        for table in tables:
            info = str(table).replace('[', '').replace(']', '')
            table_info.append(info)
    return table_info
    # st.write(generate_embedding(table_info))

def display_generated_embeddings(file_name):
    st.markdown('<u>Embeddings</u>', unsafe_allow_html=True)
    st.write(generate_embedding(extract_table_data_from_pdf(file_name)))

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def query(payload):
    response = get_rag_chat_response(payload['inputs']['text'], st.session_state['memory'], st.session_state['index'])
    return response

def get_text_input():
    input_text = st.sidebar.text_input("Human: ", "", key="input")
    return input_text

def clear_message():
    st.sidebar.empty()
    del st.session_state.past[:]
    del st.session_state.generated[:]

st.title('Upload and Display File')

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    if upload_file(uploaded_file):
        st.success("File Uploaded")
        with st.spinner('Displaying Insights.....'):
            if 'insights' not in st.session_state:
                insights = display_generated_insights(uploaded_file.name)
                st.write(insights)
                st.session_state['insights'] = insights
            else:
                st.write(st.session_state['insights'])
        with st.spinner('Indexing the file to start the chat session.....'):
            if 'memory' not in st.session_state:
                st.session_state['memory'] = get_memory()
            if 'index' not in st.session_state:
                st.session_state['index'] = build_index_from_string(extract_table_data_from_pdf(uploaded_file.name))
            display_pdf_content(uploaded_file.name)
            st.sidebar.header('Chat about the uploaded file here')
            with st.sidebar.form(key='widget', clear_on_submit=True):
                user_input = get_text_input()
                if user_input:
                    output = query({
                        "inputs": {
                            "past_user_inputs": st.session_state.past,
                            "generated_responses": st.session_state.generated,
                            "text": user_input,
                        }, "parameters": {"repetition_penalty": 1.33},
                    })

                    st.session_state.past.append('Human: ' + user_input)
                    st.session_state.generated.append('Assistant: ' + output)
                if st.sidebar.button("Clear messages"):
                    clear_message()

            chat_placeholder = st.sidebar.empty()
            with chat_placeholder.container():
                if st.session_state['generated']:
                        for i in range(len(st.session_state['generated']) - 1, -1, -1):
                            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                            message(st.session_state["generated"][i], key=str(i))
    else:
        st.error("Error uploading file")
