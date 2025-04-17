from typing import Annotated,Optional
from langchain_core.tools import tool
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

############### tools ########################################
def preprocess_query(query: str) -> str:
    """
    Comprehensively clean the query by removing:
    - Prefixes like "Considering document file,"
    - User ID references
    - Unnecessary contextual information
    
    Args:
        query (str): Original query string
    
    Returns:
        str: Cleaned, focused query
    """
    # Preprocessing steps
    def clean_query(text):
        import re
        
        # Common prefixes to remove
        prefixes_to_remove = [
            "considering document file,",
            "considering document,",
            "from document,",
            "in document,",
            "using document,",
            "with document,"
        ]
        
        # Convert to lowercase for consistent matching
        text = text.lower().strip()
        
        # Remove specified prefixes
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text.replace(prefix, '').strip()
        
        # Split the text by "and"
        parts = text.split(" and ")
        
        # Filter out parts containing user ID or identification number
        cleaned_parts = [
            part.strip() for part in parts 
            if not re.search(r'user\s*(?:id|identification\s*number)', part, re.IGNORECASE)
        ]
        
        # Rejoin the remaining parts
        cleaned_text = " ".join(cleaned_parts)
        
        # Remove any remaining numeric sequences
        #cleaned_text = re.sub(r'\d+', '', cleaned_text)
        
        # Remove extra whitespaces
        cleaned_text = ' '.join(cleaned_text.split())
        
        return cleaned_text.capitalize()
    
    # Clean the query
    cleaned_query = clean_query(query)
    
    # Debugging output
    print("\n===== Query Preprocessing =====")
    print(f"Original Query: {query}")
    print(f"Cleaned Query:  {cleaned_query}")
    
    return cleaned_query

@tool
def webSearchTool(query: str) -> str:
    """
    Wrapper function for web search that provides more robust search functionality
    
    Args:
        query (str): The search query
    
    Returns:
        str: Formatted search results
    """
    print("\n===== Web Search Tool Testing =====")
    print(query)
    # query = preprocess_query(query)
    
    try:
        # Initialize the search tool
        search = DuckDuckGoSearchRun()
        
        # Perform web search
        search_results = search.run(query)
        
        # If no results found
        if not search_results or search_results.strip() == "No good search results found":
            return f"No relevant information found for the query: {query}"
        
        # Limit the length of results to prevent overwhelming the model
        return search_results[:3000]
    
    except Exception as e:
        return f"Error during web search: {str(e)}"


@tool
def documentDataRetriverTool(query: str) -> str:
    """Retrieves information from uploaded PDF documents based on the user's query."""

    print("\n===== Document Data Retriver Tool Testing =====")
    print(query)
    query = preprocess_query(query)
    
    
    try:
        local_File_Path = "data/Job Description.pdf"  # You might want to make this configurable
        
        # Check if file exists
        import os
        if not os.path.exists(local_File_Path):
            return "Error: Document not found. Please ensure the document is uploaded correctly."
            
        # Create vector store only if it doesn't exist yet
        global vector_db  # Use a global variable to persist the vector store
        if 'vector_db' not in globals():
            print("Creating vector store from document...")
            loader = UnstructuredPDFLoader(file_path=local_File_Path)
            data = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="local-rag"
            )
            print(f"Created vector store with {len(chunks)} chunks")
        
        # Get relevant documents - limit to top 3 most relevant
        ollamModel = "llama3.2"
        llm = ChatOllama(model=ollamModel,
                         temperature=0.1,
                         max_tokens = 100
                         )
        
        # Improved retriever with more specific instructions
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""
            Generate three different search queries that would help find information in a document to answer the following question:
            Question: {question}
            
            Return only the search queries without explanation or other text.
            """,
        )
        
        # Set up retriever with search_kwargs to limit the number of docs
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(search_kwargs={"k": 3}),
            llm,
            prompt=QUERY_PROMPT
        )
        
        # Improved prompt template for better context handling
        template = """
        Answer the question based ONLY on the following context from the document:
        
        {context}
        
        Question: {question}
        
        If the information is not found in the context, respond with "This information is not present in the document."
        Provide a concise answer based only on what's in the document context.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = chain.invoke(query)
        print(response)
        return response

    except Exception as e:
        return f"Error while processing document: {str(e)}"

import pandas as pd
import numpy as np
from typing import Optional
from langchain.prompts import PromptTemplate    

@tool
def dataAnalyticTool(query: str,analysis_type: str) -> str:
    """
    Wrapper function for analyzing CSV file data with enhanced functionality
    
    Args:
        query (str): The search query
        analysis_type (str): Type of analysis to perform
    
    Returns:
        str: Formatted analysis results
    """
    print("\n===== Document Data Analytic Tool Testing =====")
    print(query,analysis_type)
    # query = preprocess_query(query)
    
    try:

        df = pd.read_csv("data/MC Market Research.csv") 
        # Basic dataset information
        dataset_info = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # Initialize LLM
        ollamModel = "llama3.2"
        llm = ChatOllama(model=ollamModel,
                         temperature=0.1
                         )
        
        # Create a comprehensive prompt for analysis
        analysis_prompt = PromptTemplate(
            input_variables=["query", "dataset_info", "analysis_type"],
            template="""You are an expert data analyst. Perform a comprehensive analysis based on the following context:

            Dataset Information:
            {dataset_info}

            Data Analysis Type:
            {analysis_type}

            User Query: {query}

            Provide a detailed, insightful analysis that addresses the user's specific requirements. Include:
            1. Relevant statistical insights
            2. Key observations
            3. Potential actionable recommendations
            4. Explanation of your analytical approach

            Analysis Response:"""
        )
        
        # Prepare data summary
        numeric_summary = df.describe().to_dict()
        categorical_summary = {col: df[col].value_counts().to_dict() 
                               for col in df.select_dtypes(include=['object']).columns}
        
        data_summary = {
            'numeric_summary': numeric_summary,
            'categorical_summary': categorical_summary
        }
        
        # Generate analysis
        formatted_dataset_info = str(dataset_info)
        formatted_data_summary = str(data_summary)
        
        analysis_input = analysis_prompt.format(
            query=query, 
            dataset_info=formatted_dataset_info, 
            data_summary=formatted_data_summary,
            analysis_type=analysis_type
        )
        
        # Execute LLM analysis
        llm_analysis = llm.invoke(analysis_input)
        
        return llm_analysis
    
    except Exception as e:
        return f"Error during analysis: {str(e)}"
    

##################### Supervisor Node #####################################

from typing import Literal,Optional
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState, END
from langgraph.types import Command

members = ["chatWithDocument","generalChatBot","dataAnlyticBot"]
options = members + ["FINISH"]


system_prompt = f"""
            You are a supervisor responsible for managing a conversation between the following workers: {members}. Your primary objective is to efficiently route user queries to the appropriate worker based on their expertise.

            Task Execution Flow:
            1. Identify the Agent Type:

             - Analyze the user request and determine which worker is best suited to handle it.

             - If the request pertains to document retrieval, assign it to the documentDataRetrieverTool.

             - If the request requires external web data, assign it to the webSearchTool.

             - If the request requires comprehensive data analysis from a CSV file, assign it to the dataAnalyticTool.

            2. Task Execution:

             - Direct the query to the appropriate worker.

             - Wait for the worker to complete the task and return a response.

            3. Completion Handling:

             - Each worker will return their results along with a status update.

             - If the task is successfully completed and no further action is needed, respond with "FINISH".

            Your goal is to ensure smooth coordination, efficient task execution, and accurate query resolution by managing the workflow effectively.
"""

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Annotated[Literal["chatWithDocument","generalChatBot","dataAnlyticBot","FINISH"],...,"worker to route to next, route to FINISH"]
    reasoning: Annotated[str, ..., "Support proper reasoning for routing to the worker"]

class State(MessagesState):
    next: str
    id_number: Optional[str] = None

ollamModel = "llama3.2"
llm = ChatOllama(model=ollamModel,
                         temperature=0.1,
                         max_tokens = 100
                )

def supervisor_node(state: State) -> Command[Literal["chatWithDocument","generalChatBot","dataAnlyticBot", "__end__"]]:

    #Path Tracker
    print("===== Supervisor Node Called =====")
    #print(f"Current State: {state}")
    
    # Determine which agents have been called
    agent_calls = {
        "chatWithDocument": any(msg.name == "chatWithDocument" for msg in state.get("messages", [])),
        "generalChatBot": any(msg.name == "generalChatBot" for msg in state.get("messages", [])),
        "dataAnlyticBot": any(msg.name == "generalChatBot" for msg in state.get("messages", []))
    }
    print(f"Agents Called: {agent_calls}")
    #########################################
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    if any(
        msg.name in ["chatWithDocument", "generalChatBot","dataAnlyticBot"] 
        and msg.content not in ["I do not know.", "Error occurred."]
        for msg in state["messages"]
    ):
        return Command(goto=END)
    
    query = state['messages'][0].content if state['messages'] else ''

    if len(state['messages'])==1:
        query = state['messages'][0].content
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    if query:
        return Command(
            goto=goto, 
            update={"next": goto,'query':query,
            'cur_reasoning':response["reasoning"],
            "messages":[HumanMessage(content=f"user's identification number is {state['id_number']}")]
            }
        )
    return Command(goto=goto, update={"next": goto,'cur_reasoning':response["reasoning"]})

################### Graph ###################################

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

chat_With_Document = create_react_agent(
    llm,
    tools = [documentDataRetriverTool],
    prompt="""
        Intelligent Assistant for Document-Based Query Handling
        You are an advanced assistant responsible for handling user queries using a document-based search tool.

        Critical Instructions:
        1. ALWAYS use the documentDataRetriverTool for document-related queries.
        2. Return the EXACT response from the tool without modification.
        3. Return tool's output EXACTLY as received.
        4. If the tool returns a direct answer, present that answer verbatim.

        - Accuracy and Relevance: Ensure that responses are factual, concise, and directly address the user's query using the most relevant sources.

        Your goal is to provide precise and reliable information derived exclusively from the documentDataRetriverTool tool.
        """
)

def chat_With_Document_node(state: State) -> Command[Literal["supervisor"]]:

    #Path Tracker
    print(">>> Entering chatWithDocument Node <<<")
    # print(f"Current State: {state}")

    ##########################

    result = chat_With_Document.invoke(state)
    #print(result["messages"][-1].content)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="chatWithDocument")
            ]
        },
        goto="supervisor",
    )

general_chat_bot = create_react_agent(
    llm,
    tools = [webSearchTool],
    prompt="""
        Intelligent Assistant for Web-Based Query Handling
        You are an advanced assistant responsible for handling user queries using a web search tool. Your primary objective is to retrieve accurate and up-to-date information efficiently.

        Instructions for Web Search:
        - Primary Source: If the user query requires external data, use the webSearchTool to fetch the latest and most relevant information from the web.

        - Strict Data Source: Generate answers only based on the retrieved web search results. Do not rely on prior knowledge or assumptions.

        - Accuracy and Relevance: Ensure that responses are factual, concise, and directly address the user's query using the most relevant sources.

        Your goal is to provide precise and reliable information derived exclusively from the web search tool.
    """
)

def general_chat_bot_node(state: State) -> Command[Literal["supervisor"]]:

    #Path Tracker
    print(">>> Entering GenaralChatBot Node <<<")
    print(f"Current State: {state}")

    ##########################

    result = general_chat_bot.invoke(state)
    print(result["messages"][-1].content)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="generalChatBot")
            ]
        },
        goto="supervisor",
    )

data_analytic_agent = create_react_agent(
    llm,
    tools = [dataAnalyticTool],
    prompt="""
        Intelligent Data Analysis Agent

            You are an advanced Data Analysis Specialist, responsible for extracting actionable insights from CSV datasets using comprehensive analytical techniques.

            Core Responsibilities:
             - Utilize the dataAnalyticTool for all data-driven queries and analytical tasks.

             - Identify and apply the most suitable analysis type based on the user's request.

             - Ensure precision, depth, and clarity in all analytical responses to support data-driven decision-making.

            Your ultimate goal is to provide accurate, insightful, and valuable data analysis to empower informed decision-making
        """
    )

def data_analytic_agent_node(state: State) -> Command[Literal["supervisor"]]:

    #Path Tracker
    print(">>> Entering dataAnlyticBot Node <<<")
    print(f"Current State: {state}")

    ##########################

    result = data_analytic_agent.invoke(state)
    print(result["messages"][-1].content)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="dataAnlyticBot")
            ]
        },
        goto="supervisor",
    )


#############################GRAPH###############################################################

builder = StateGraph(State)
builder.add_edge(START,"supervisor")
builder.add_node("supervisor",supervisor_node)
builder.add_node("chatWithDocument",chat_With_Document_node)
builder.add_node("generalChatBot",general_chat_bot_node)
builder.add_node("dataAnlyticBot",data_analytic_agent_node)
graph = builder.compile()

def get_final_result(query):
    """
    Helper function to get the final result from the graph
    
    Args:
        query (str): The user's input query
    
    Returns:
        str: The final result from the graph
    """
    # Prepare inputs
    inputs = [HumanMessage(content=query)]
    
    # Configure the state
    config = {"configurable": {"thread_id": "1", "recursion_limit": 10}}  
    state = {'messages': inputs,'id_number': 10232303}
    
    # Invoke the graph
    result = graph.invoke(input=state, config=config)
    
    # Extract the final result (last message's content)
    if result['messages']:
        final_answer = result['messages'][-1].content
        return final_answer
    else:
        return "No result found."
