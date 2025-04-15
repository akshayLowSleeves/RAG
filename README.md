# RAG Project Setup and Usage

This README outlines the steps required to set up and replicate the RAG system. Follow the instructions below to configure your environment, build and run the Docker containers, and make API requests.

---

## 1. Environment Setup

Before starting the system, you need to add a **.env** file with the following 3 environment variables:

- **QDRANT__SERVICE__API_KEY**  
  API key for accessing the database (which will be created in Docker).  
  > *Note:* This key is also used as the authentication key for accessing the API where queries are sent. Typically, two different keys should be used but for this demo, a single key is employed.

- **CUDA_AVAILAIBLE**  
  Specifies whether the RAG system should use the GPU when available.  
  Acceptable values: `"Yes"` or `"No"`  
  > *Note:* The system was tested without a GPU; hence, `"No"` works fine. See [Section 4](#4-cuda-configuration) for more details.

- **OPENAI_API_KEY**  
  Required for accessing the OpenAI API and downloading necessary weights.

> **Important:** This process requires access to the internet (for OpenAI calls and weight downloads).

---

## 2. Build and Run Docker

After setting up the environment variables, proceed with the Docker build and run process:

1. Put your PDF in the `retriver` and `generator` directories. 
2. Execute the bash script **run.sh**.
3. This script will build the containers and start them.  
   Once completed, the API will be available to receive requests at:
    `http://127.0.0.1:8444/retriver/`

---

## 3. Making API Requests

After running the containers, you can send requests to the API. Below is a sample Python script demonstrating how to make an API request:

```
python
import requests
import json

url = "http://127.0.0.1:8444/retriver/"
headers = {
 "Content-Type": "application/json",
 "api-key": "jTM985%#fTRrfs?isn%"
}

Question = 'Which page shows a nail going through the tire?'
data = {"text": Question}

try:
 response = requests.post(url, headers=headers, data=json.dumps(data))
 response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
 jsons = response.json()
 print("Request successful!")
 print("Status Code:", response.status_code)
 print(jsons['response']['text'])
except requests.exceptions.RequestException as e:
 print(f"Request failed: {e}")
except json.JSONDecodeError:
 print("Failed to decode JSON response.")
 ```

---  

## 4. CUDA Configuration

If you are running the system with `CUDA_AVAILAIBLE="No"`, please note the following:

- As explained in the write-up, an open model like `colqwen` is used, so the embedding calculation is done externally (via [Google Colab](https://colab.research.google.com/)).
- To generate the PDF embeddings, follow these steps:
  1. Run `./vlmRAG/retriver/indexer1.py` in the same directory as your PDF file. This script will output an embedding file named `./pdf_emb.pt`.
  2. Place the generated `./pdf_emb.pt` file into the `./vlmRAG/retriver/` folder.
- Finally, run the `./run.sh` file as described in [Section 2](#2-build-and-run-docker) to start the containers.

## 5. Additional Information

- **API Endpoint:**  
  Access the API at [http://127.0.0.1:8444/retriver/](http://127.0.0.1:8444/retriver/).

- **Run Commands:**  
  Ensure you have [Docker](https://www.docker.com/) installed and that your network configuration allows for outbound internet access to fetch dependencies.
- **Evaluations:** 
  Some mini evaluation were done ( as descibed in the write up). The `evaluation.ipynb` has the code for evaluation process described in the write up. The evaluation data is in the `.json` files. Here is the result table -

  | Language | Total | Correct | Wrong | Accuracy (%) |
  |----------|-------|---------|-------|--------------|
  | English  | 27    | 22      | 5     | 81.48        |
  | Spanish  | 27    | 22      | 5     | 81.48        |

