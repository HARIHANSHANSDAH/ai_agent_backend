
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import Dict
import io
import base64
import matplotlib.pyplot as plt
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)


API_KEY = "MY_API_KEY" 

client = genai.Client(api_key=API_KEY)

uploaded_data = {}  

class QueryRequest(BaseModel):
    session_id: str
    question: str


@app.post("/upload")

async def upload_excel(file: UploadFile, session_id: str = Form(...)):
    content = await file.read()
    filename = file.filename.lower()

    cleaned_sheets = {}

    if filename.endswith((".xlsx", ".xls")):
       
        xls = pd.read_excel(io.BytesIO(content), sheet_name=None)
        for sheet_name, df in xls.items():

            df.columns = [f"Column_{i}" if col is None or str(col).startswith("Unnamed") else str(col) 
                          for i, col in enumerate(df.columns)]
            df.fillna("", inplace=True)
            cleaned_sheets[sheet_name] = df

    elif filename.endswith(".csv"):

        df = pd.read_csv(io.BytesIO(content))
        df.columns = [f"Column_{i}" if col is None or str(col).startswith("Unnamed") else str(col) 
                      for i, col in enumerate(df.columns)]
        df.fillna("", inplace=True)
        cleaned_sheets["Sheet1"] = df

    else:
        return {"error": "Unsupported file format. Please upload .xlsx, .xls, or .csv"}

    uploaded_data[session_id] = cleaned_sheets
    return {"sheets": list(cleaned_sheets.keys())}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    session_id = request.session_id
    question = request.question

    if session_id not in uploaded_data:
        return {"error": "No data uploaded for this session."}

    sheets = uploaded_data[session_id]


    context = {name: df.to_dict(orient="records") for name, df in sheets.items()}


    response = client.models.generate_content(
        model="gemini-2.5-flash",
        
        contents=f"Data: {context}\nQuestion: {question}",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )

    chart_base64 = None
    try:

        first_sheet = next(iter(sheets.values()))
        numeric_cols = first_sheet.select_dtypes(include='number').columns
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(6,4))
            first_sheet.plot(x=numeric_cols[0], y=numeric_cols[1], kind='bar')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
    except Exception as e:
        chart_base64 = None

    return {"answer": response.text, "chart": chart_base64}

@app.get("/sessions")
async def list_sessions():
    return {"sessions": list(uploaded_data.keys())}
