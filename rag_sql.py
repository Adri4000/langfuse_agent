# cd langfuse/
# run docker compose up -d
# cd ..
# run source .env

# cd langfuse/
# docker compose down


from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

from langchain.agents import initialize_agent, AgentType, Tool
from langfuse import Langfuse, Evaluation
from langfuse.langchain import CallbackHandler

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from jinja2 import Environment, FileSystemLoader
import pandas as pd
from datetime import datetime


#### initialize langfuse
langfuse = Langfuse()
callback_handler = CallbackHandler() # Necessary to trace all tools and llm


# ----------------------------
# Build the tools
# ----------------------------


def RAG_tool(dir="data/chroma_db", k=3):
    """
    Setup RAG document retriever.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=dir, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    tool = Tool(
        name="DocumentRetriever",
        func=lambda q: retriever.get_relevant_documents(q),
        description="Always use this tool to answer any question about books. Do not answer without consulting it. Input a question, output relevant text chunks only.",
        callbacks=[callback_handler]
    )
    return tool



def SQL_tool(dir="sqlite:///data/sale_database.db"):
    """
    Setup SQL tool.
    """
    db = SQLDatabase.from_uri(dir)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
        # Attach callback handler to each SQL tool
    for t in tools:
        t.callbacks = [callback_handler]
    return tools



# ----------------------------
# Build agent with tools
# ----------------------------


class MyAgent():
    """
    Build agent with both SQL & RAG tools.
    """
    def __init__(self, tools, llm):
        self.tools = tools
        self.llm = llm
        self.agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # OPENAI_FUNCTIONS, OPENAI_MULTI_FUNCTIONS, ZERO_SHOT_REACT_DESCRIPTION
            verbose=True,
            callbacks=[callback_handler]  # <-- logs entire agent run
            )

    def answer(self, input: str):
        """
        Takes as input a str
        """
        output = self.agent.run(input)
        return output

    def evaluate(self, *, item, **kwargs):
        """
        Function to use with the evaluation on a Langfuse dataset
        """
        output = self.agent.run(item.input)
        return output



# ----------------------------
# Build evaluator
# ----------------------------


class EvaluatorStructure(BaseModel):
    """
    To help the LLM-as-a-judge to return the following correct format.
    """
    accuracy: int = Field(..., description="Score 1–5 for accuracy", ge=1, le=5)
    clarity: int = Field(..., description="Score 1–5 for clarity", ge=1, le=5)
    exhaustiveness: int = Field(..., description="Score 1–5 for exhaustiveness", ge=1, le=5)

    explanation_accuracy: str = Field(..., description="An explanation for the accuracy score")
    explanation_clarity: str = Field(..., description="An explanation for the clarity score")
    explanation_exhaustiveness: str = Field(..., description="An explanation for the exhaustiveness score")


def evaluator(*, input, output, expected_output, metadata, **kwargs):
    """
    Custom evaluator that scores both faithfulness and relevancy.
    """

    # jinja prompt template for content evaluation
    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("scoring_template.txt.jinja")

    elements = {
        "question": input,
        "model_output": output,
        "expected_answer": expected_output
    }
    prompt = template.render(elements)


    # LLM-as-a-judge
    llm_judge = ChatOpenAI(model_name="gpt-4o-mini", temperature=0) # gpt-4o-mini, o3-mini (no temperature argument)

    structured_llm = llm_judge.with_structured_output(EvaluatorStructure)
    generation = structured_llm.invoke([HumanMessage(content=prompt)])

    response = generation.model_dump()

    scores = [
        Evaluation(name="accuracy", value=int(response["accuracy"]), comment=response["explanation_accuracy"]),
        Evaluation(name="clarity", value=int(response["clarity"]), comment=response["explanation_clarity"]),
        Evaluation(name="exhaustiveness", value=int(response["exhaustiveness"]), comment=response["explanation_exhaustiveness"])
              ]
    
    return scores


def get_eval_data(results, save=True):
    """
    To save locally the evaluation result of the agent.
    """

    records = []

    #Start timestamp of the evaluation on the dataset
    timestamp_str = results.run_name.split(" - ")[1]
    dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    formatted = dt.strftime("%Y-%m-%d_%H-%M-%S")


    for element in results.item_results:

        r = {
            'dataset_run_id': results.dataset_run_id,
            'run_start_time_UTC+0': formatted,
            'input': element.item.input,
            'reference': element.item.expected_output,
            'output': element.output
        }
    
        for ev in element.evaluations:
            r[ev.name] = ev.value
            r[ev.name + "_analysis"] = ev.comment

        records.append(r)    


    df = pd.DataFrame(records)
    if save:
        # take the date from the last "element" variable of the loop
        last_time = formatted
        df.to_csv(f"results/{last_time}.csv", index=False)
    return df



# ----------------------------
# Run with traces in Langfuse (go to http://localhost:3000)
# ----------------------------

if __name__=="__main__":

    #### Choose dataset and if we save the results or not
    dataset_name = "dataset_rag_sql"
    save_result = False


    #### Construct Agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[callback_handler])

    sql_tool = SQL_tool(dir="sqlite:///data/sale_database.db")
    rag_tool = RAG_tool(dir="data/chroma_db", k=3)
    tools = sql_tool + [rag_tool]

    agent = MyAgent(tools=tools, llm=llm)


    #### Evaluation on the dataset
    dataset = langfuse.get_dataset(dataset_name)

    results = dataset.run_experiment(
        name="Production Model Test",
        description="Monthly evaluation of our production model",
        task=agent.evaluate,
        evaluators=[evaluator],
    )

    df_result = get_eval_data(results=results, save=save_result)
