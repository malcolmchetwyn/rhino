# Rhino [![N|Solid](https://salesforcerhino.com/img/new_logo.svg)](https://nodesource.com/products/nsolid)
## OpenSource LLM Learning Project

Rhino is a performance optimized Python FastAPI Application to help you understand how Embeddings & Chunking your own data works with OpenAI and other LLM Platforms. I made it as a internal tool to help me so many things such as respond to RFPs, Account Analysis and More.

This is a basic example that does not use tools or agents. I have built account login/signup into it also for you. :) We can use these concepts to extend the Salesforce Platform for any use case for example:
- Enable Sales Reps to Research Annual Reports
- Compare differences between Company A and Company B annual reports
- Determine what the best salesplays are against the accounts annual report
- Upload files and get insights from them
- Extend the Salesforce platform and supercharge it with complimentary capability

You can see Rhino in action here - [Salesforce Rhino](https://salesforcerhino.com)

## Installation

Clone Repo.

Install Homebrew: If you don't already have Homebrew installed, open the Terminal and run:
```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Once Homebrew is installed, you can install Redis by running:
```sh
brew install redis
```

Start Redis Server:
```sh
brew services start redis
```

Verify Redis Installation:
```sh
redis-cli ping
```
Install Python 3.10: Once Homebrew is installed, run the following command to install Python 3.10:
```sh
brew install python@3.10
brew link python@3.10
```

Commands (assumer default directory for home-brew). Run them all, don't worry if deactive has an error.
```sh
deactivate
rm -rf .venv
/opt/homebrew/bin/python3.10 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
```

## Data Indexes
You will require data to be indexes if you want to search locally and with OpenAI. In each file starting with Vector.. you will file code such as the below. Add/Update/Remove the indexes here. Example:
```sh
"""Perform all index searches concurrently."""
descriptions = [   
    #"data_indexes/public_news_and_events_index", 
    "data_indexes/annual_reports"
    #"data_indexes/federal_legislativeinstruments_inforce_index",
    #"data_indexes/gov_compliance_law",
    #"data_indexes/salesforce_customer_stories"
]
```

## Performance
1. Uses FastAPI for low latency and asynchronous request handling, making it suitable for high-performance IO-bound and high-concurrency systems
2. Using word streaming back to use to make it feel modern
3. All indexes are loaded into memory once on script execution to optimize data retrieval. If an index somehow gets dropped it will automatically pick it up again.
4. Used FAISS. Facebooks massively scaleable opensource vector store.
5. Used Langchain where possible. Some of langchain is slow.
6. Only chunks files when over the token limit
7. Implement my own chunking for faster retrieval
8. Uses Redis for Session Management
9. Uvicorn is an ASGI web server. It's known for its speed and efficiency, often used in conjunction with FastAPI and other ASGI frameworks.
10. FAISS lookups use await asyncio.to_thread(lambda: load_index(description, embeddings)) for maximum performance

## Getting Around Token Limitations 
I've written code to tokenize large text inputs, chunk and batch them then combine them if needed. I have not used Langchain but this way is more responsive however it uses an MapReduce pattern.


## Other Info
1. You will need to create your own data indexes to plugin. Ping me directly if you want some examples. I have included a file called "train your data" so you can train you own data. It will process .txt, pdf etc. just put the files in the directory and let is chunk it then either create, merge or delete your DB
2. You will need to add your own OpanAI Key in the .env file
3. I have created a gmail with user name and password you can use
4. If hosted on a domain, you will need to change the forgot password link to your hosted domain
5. Credenitals, all passwords are stored in a file but encrypted (didn't need a DB for this project)
6. Vector Store: it used FAISS. It's facebooks massively scaleable opensource vector store.



## Common Mistakes to look out for

1. Do not install any other lib's other than what's in requirements.txt file. OpenAI is always changing their API's so this version works.

