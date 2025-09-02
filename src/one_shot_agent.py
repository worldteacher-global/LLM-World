import os
import sys
sys.append('/home/sagemaker-user/user-default-efs/CLONED_REPOS/LLM-World/src')
from mytools import MyTools
from utils import Utils
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import creat_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from dotenv import load_dotenv
load_dotenv('/home/sagemaker-user/user-default-efs/CLONED_REPOS/LLM-World/.env')

