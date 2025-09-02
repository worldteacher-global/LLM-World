import os
import sys
# sys.path.append('/home/sagemaker-user/user-default-efs/CLONED_REPOS/LLM-World/src')
sys.path.append('/home/ec2-user/BoW/LLM-World/src') ## EC2
from mytools import MyTools
from utils import Utils
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv
# load_dotenv('/home/sagemaker-user/user-default-efs/CLONED_REPOS/LLM-World/.env')
load_dotenv('/home/ec2-user/BoW/LLM-World/.env')   ## EC2
import streamlit as st

def oneshotagent(input_prompt: str):
    prompt = input_prompt
    
    mcp_client = (
        {
            "mcp_tools":{
                "url":os.getenv("MCP_URL"),
                "transport":"sse"
            }
        }
    )

    system_prompt = ChatPromptTemplate(
        [
            (
                "system","You are a statistics agent. \n\n"
                "INSTRUCTIONS:\n"
                "- Assist with statistics related tasks. \n\n"
                "- Call the apprpriate tools to complete a task. \n\n"
                " List your reasoning when completing a task. \n\n"
                "- Respond with the results of your work, explaning the final result in clear detail."
            ),
            (
                "placeholder", "{messages}"
            )
        ]
    )

    tools = MyTools()
    helper = Utils()

    # mcp_tools = await mcp_client.get_tools()
    # mcp_tool = None
    # for tool_obj in mcp_tools:
    #     if tool_obj.name == "google_search_tool":
    #         mcp_tool = tool_obj 

    tools = [tools.correlation_tool, tools.covariance_tool, tools.anova_one_way, tools.levene_test, tools.normality_tool, tools.calculate_PC, tools.gen_plot, tools.display_base64_image]

    agent = create_react_agent(
        model=helper.openai_llm("gpt-4.1"),
        tools=tools,
        prompt=system_prompt,
        name="statistics_agent" 
    )

    agent_response = agent.invoke(
        {"messages": [
            {"role":"user", "content":[{"type":"text", "text":prompt}]}
            ]
        }
    )
    
    return agent_response

if __name__=='__main__':
   
    # query = 'Generate a line plot with the title "Sales Over Time. The x-axis should be [1, 2, 3, 4, 5] representing months, and the y-axis should be [10, 20, 15, 30, 25] representing sales in thousands.'
    
    # result = oneshotagent(query)

    from langchain_core.messages import ToolMessage 

    # image_path = None

    # for obj in result['messages']:
    #     if isinstance(obj, ToolMessage):     
    #         if obj.name=='gen_plot':
    #             image_path=obj.content

    # print(image_path)

    st.title('I am a Statistics Assistant') 

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    UPLOAD_DIR = "uploaded_files"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    prompt = st.chat_input("Hello! Please submit a statistics related question or request:", accept_file=True, file_type=[".csv"])        
    if prompt:
        
        ### File upload
        user_text = prompt.text if hasattr(prompt, "text") else str(prompt)
      
        file_path = None
        if hasattr(prompt, "files") and prompt.files:
            uploaded_file = prompt.files[0]
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            uploaded_file.seek(0)

            st.session_state["uploaded_file_path"] = file_path
        elif "uploaded_file_path" in st.session_state:
            file_path = st.session_state["uploaded_file_path"]
        else:
            uploaded_file = None
            file_path = None
        ###

        st.chat_message("user").markdown(user_text)    
        st.session_state.messages.append({"role": "user","content":user_text})
        
        with st.chat_message("assistant"):
            
            response = oneshotagent(user_text)            
            
            image_path = None
            if response:                    
                for obj in response['messages']:
                    if isinstance(obj, ToolMessage):     
                        if obj.name=='gen_plot':
                            image_path=obj.content
            else:
                st.session_state.messages.append({"role":"assistant", "content":response['messages'][-1].content})   

            if image_path:
                # st.image(image_path, caption="Created Plot")
                st.write(image_path)
            # else:
                # st.write(response['messages'][-1].content)            
     
            # st.session_state.messages.append({"role":"assistant", "content":response})       
                

    
