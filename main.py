import tabula
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pdfplumber

# Initialize Sentence Transformer for embeddings
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to extract tables from PDF using tabula
def extract_tables_from_pdf(pdf_path, page_number):
    tables = tabula.read_pdf(pdf_path, pages=page_number, multiple_tables=True, pandas_options={"header": None})
    return tables

# Function to clean and merge multi-line rows in tables
def clean_and_merge_table(df):
    df = df.replace(np.nan, "", regex=True)
    cleaned_data = []
    current_row = []
    for _, row in df.iterrows():
        if current_row and row[0] == "":
            current_row = [f"{a} {b}".strip() for a, b in zip(current_row, row)]
        else:
            if current_row:
                cleaned_data.append(current_row)
            current_row = list(row)
    cleaned_data.append(current_row)
    cleaned_df = pd.DataFrame(cleaned_data)
    headers = cleaned_df.iloc[0]
    cleaned_df = cleaned_df[1:]
    cleaned_df.columns = headers
    return cleaned_df.reset_index(drop=True)

# Function to build FAISS index for retrieved data
def build_faiss_index(documents):
    embeddings = sentence_model.encode(documents)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Build a flat (brute-force) index
    index.add(embeddings)
    return index

# Function to retrieve relevant data using FAISS
def retrieve_relevant_data(query, documents, index):
    query_embedding = sentence_model.encode([query])
    query_vector = np.array(query_embedding).astype('float32')
    _, indices = index.search(query_vector, k=3)  # Retrieve top 3 documents
    relevant_data = []
    for idx in indices[0]:
        relevant_data.append(documents[idx])
    return relevant_data

# Function to ensure that all rows have the same number of columns
def ensure_consistent_columns(relevant_rows, num_columns):
    consistent_rows = []
    for row in relevant_rows:
        if len(row) < num_columns:
            # If there are fewer columns, pad the row with empty strings
            row.extend([''] * (num_columns - len(row)))
        elif len(row) > num_columns:
            # If there are more columns, truncate the row
            row = row[:num_columns]
        consistent_rows.append(row)
    return consistent_rows

# Main function to extract, clean, retrieve, and format output as a table
def run_rag_pipeline(pdf_path, page_number, query):
    # Step 1: Extract tables from PDF
    tables = extract_tables_from_pdf(pdf_path, page_number)
    
    if tables:
        # Step 2: Clean and merge multi-line rows
        raw_df = pd.concat(tables, ignore_index=True)
        cleaned_df = clean_and_merge_table(raw_df)

        # Step 3: Convert cleaned table into a list of documents (each row is a document)
        documents = cleaned_df.apply(lambda x: ' '.join(x.astype(str)), axis=1).tolist()

        # Step 4: Build FAISS index for document retrieval
        index = build_faiss_index(documents)

        # Step 5: Retrieve relevant data using FAISS
        relevant_data = retrieve_relevant_data(query, documents, index)

        # Step 6: Format relevant data as a table
        relevant_rows = []
        for row in relevant_data:
            row_data = row.split()  # Split each row based on spaces (assuming it’s a space-separated table)
            relevant_rows.append(row_data)

        # Ensure all rows have the same number of columns
        relevant_rows = ensure_consistent_columns(relevant_rows, len(cleaned_df.columns))

        # Convert the relevant rows back into a DataFrame for tabular display
        relevant_df = pd.DataFrame(relevant_rows, columns=cleaned_df.columns)
        
        # Return the DataFrame with relevant data
        return relevant_df
    else:
        return "No tables found on the specified page."

# Path to the PDF file
pdf_path = "C:\Users\nagam\Desktop\madhu\Ragpipeline.pdf" 

# Define the query you want to ask the system
query = "Show the data for All Industries, Manufacturing, Finance, Insurance, Real Estate for the years 2010-2015"

# Run the RAG pipeline and get the result as a DataFrame
result_df = run_rag_pipeline(pdf_path, page_number=6, query=query)

# Print the output as a table
if isinstance(result_df, pd.DataFrame):
    # Format the output to match your requested format
    print(result_df.to_string(index=False))
else:
    print(result_df) 