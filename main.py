from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tool import ocr_tool, filter_tool
import pandas as pd
import os


load_dotenv() # load the environment variables from env file; this is loading API key

# pydantic model
class BridgeWork (BaseModel):
    job_number: str
    proposed_work: list[str]
    date: str

llm = ChatOpenAI(model = "gpt-3.5-turbo")
parser = PydanticOutputParser(pydantic_object = BridgeWork)

prompt = ChatPromptTemplate.from_messages(
[
    (
        "system", 
        """
        You are an assistant helping bridge engineers identify proposed bridge work.


        Use the following tools **in sequence**:
        1. `extract_text_from_title_sheet` - extract text from the title sheet (first page) of the bridge plan PDF.
        2. `filter_target_section` - clean and normalize the extracted OCR text.


        Once the text is filtered and normalized, your job:
        1. Extract **all job numbers** (`job_number`) found in the text. 
        - job numbers are always numeric identifiers consisting of exactly six digits, 
          and they may optionally have a single letter at the end (e.g., 123456 or 123456A)
        - If there are multiple job numbers, combine unique values into a comma-separated string (no duplicates).
        - Ignore structure numbers(e.g., B01-21022, V01-21022, C01 of 22222).

        2. Locate the 'Contract for:' section and **all work items associated with any job number**. 
        - For each job number, include all work tasks listed after the job number label (e.g., "JN 201222A: ...").
        - **Do not include structure names or locations** like "Inkster Road Over Rouge River" or "I-96 Over CSX RR".
        - The text immediately following `CONTRACT FOR:` is the scope of work (e.g., "Bridge removal", "Epoxy overlay")
        - Work items text may be a single long squished word or several squished words (e.g., `BRIDGEREPLACEMENTANDAPPROACHRECONSTRUCTION`).

        3. Extract the `date` if present.
        
        4. Proposed work may span multiple lines, be associated with multiple job numbers, or include bullets. 
        - Combine all unique work items into one `proposed_work` list.

        **You must output ONLY valid JSON that matches this format:**
        {format_instructions}

        Do not add any text, bullets, explanations, or Markdown outside the JSON object.
        """
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]
).partial(format_instructions = parser.get_format_instructions())

tools = [ocr_tool, filter_tool]

agent = create_tool_calling_agent(
    llm = llm,
    prompt = prompt,
    tools = tools
)

agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)

# define input and output pathes
folder_path = input("Enter the full path to the folder containing bridge plans (PDFs): ")
folder_path = folder_path + "\\"
output_path = os.path.join(folder_path, "output")
os.makedirs(output_path, exist_ok=True)

results = []

# loop over pdfs in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        print(f"\n Processing: {filename}")
        try:
            query = f"Extract job number, proposed work and date from {file_path}"
            raw_response = agent_executor.invoke({"query": query})

            structured_response = parser.parse(raw_response.get("output"))

            results.append({
                "file_name": filename,
                "job_number": structured_response.job_number,
                "proposed_work": ", ".join(structured_response.proposed_work),
                "date": "".join(structured_response.date),
            })

        except Exception as e:
            print(f" Error processing {filename}: {e}")
            results.append({
                "file_name": filename,
                "job_number": "",
                "proposed_work": "",
                "date": f"Error: {e}"
            })

# Save output
output_file = os.path.join(output_path, "bridge_work_summary.xlsx")
df = pd.DataFrame(results)
df.to_excel(output_file, index=False)
print(f"\n Extraction complete. Results saved to {output_file}")