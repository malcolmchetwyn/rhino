

import os
from typing import List, Optional
from fastapi import FastAPI, Request, Form, Depends, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import openai
from langchain.vectorstores import FAISS
from openai.error import OpenAIError
from dotenv import load_dotenv
import tiktoken
from concurrent.futures import ThreadPoolExecutor
from langchain.embeddings import HuggingFaceBgeEmbeddings
import re
import logging
from send_email import send_email 
from fastapi.responses import JSONResponse 
import asyncio


embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5")

loaded_news_indexes = {}

"""Perform all index searches concurrently."""
descriptions = [   
    #"data_indexes/public_news_and_events_index", 
    "data_indexes/annual_reports"
    #"data_indexes/federal_legislativeinstruments_inforce_index",
    #"data_indexes/gov_compliance_law",
    #"data_indexes/salesforce_customer_stories"
]

def load_index(description, embeddings):
    """Load the FAISS index for a given description."""
    if description not in loaded_news_indexes:
        logging.info(f"Loading index for description: {description}")
        #print(f"Loading index for description: {description}")
        loaded_news_indexes[description] = FAISS.load_local(description, embeddings)
    else:
        logging.info(f"Index for {description} already loaded.")
        print(f"Index for {description} already loaded.")
    return loaded_news_indexes[description]


async def optimized_search_index(description, query):
    """Search the loaded FAISS index using caching for optimization."""
    try:
        # Check if index is already loaded, if not, load it
        db = loaded_news_indexes.get(description)
        if db is None:
            db = await asyncio.to_thread(lambda: load_index(description, embeddings))

        law_search = db.similarity_search(query)  # Assuming this is a synchronous operation
        page_contents = [doc.page_content for doc in law_search]
        return '\n'.join(page_contents)

    except Exception as e:
        logging.error(f"Error occurred in optimized_search_index: {e}")
        raise


def concurrent_search_index(args):
    """Function to be called concurrently."""
    return optimized_search_index(*args)

async def search_all_indices(query):
    # Ensure all indices are loaded (this will only actually load missing indices)
    for desc in descriptions:
        if desc not in loaded_news_indexes:
            await asyncio.to_thread(lambda: load_index(desc, embeddings))
            
    # Prepare arguments for each call
    args_list = [(desc, query) for desc in descriptions]

    # Use asyncio to execute the searches concurrently
    results = await asyncio.gather(*(optimized_search_index(desc, query) for desc in descriptions))

    return results

async def get_news_details(essential_query):

    try:
        #Search Vector Stores
        all_search_results = await search_all_indices(essential_query)
        final_index_response = ' '.join(all_search_results)

        # Provided cleaning code
        final_index_response = re.sub(r'\d-\d{2,}', '', final_index_response)  # Remove patterns like '3-30'
        final_index_response = re.sub(r'Part \d+', '', final_index_response)  # Remove patterns like 'Part 7'
        final_index_response = re.sub(r'[ \t]+', ' ', final_index_response)  # Replace any sequence of whitespace with a single space
        final_index_response = re.sub(r'\n\s*\n', '\n', final_index_response)  # Remove blank lines
        final_index_response = re.sub(r'(\d)\s{2,}(\w)', r'\1 \2', final_index_response)  # Collapse spaces between a number and the following word
        #new 
        final_index_response = re.sub(r'\xa0', ' ', final_index_response)  # Replace the non-breaking space character with a regular space
        final_index_response = re.sub(r'\.{3,}', '...', final_index_response)  # Replace any sequence of 3 or more dots with just three dots
        final_index_response = re.sub(r'\s+', ' ', final_index_response)  # Replace any sequence of whitespace (including newlines) with a single space

        return final_index_response

    except Exception as e:
        return "Error"
 
def preload_news_indices():
    #print("Pre-loading news indices...")
    logging.info("Pre-loading news indices...")
    for desc in descriptions:
        load_index(desc, embeddings)  # Adjust 'embeddings' if it's not globally available or needs to change per description
    print("All news indices pre-loaded.")
    logging.info("All news indices pre-loaded.")