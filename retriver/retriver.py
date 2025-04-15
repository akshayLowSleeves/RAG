from contextlib import asynccontextmanager
import os, httpx, sys
import torch, time
from PIL import Image
from tqdm import tqdm 
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel


app = FastAPI()

if len(sys.argv) < 3:
    print("No argument was passed to the script.")
    sys.exit(1)
db_url = "http://qdrant:6333"
generator_url = "http://generator:8443/query_images/"
is_cuda = sys.argv[1]
api_key = sys.argv[2]
device = "cpu" if is_cuda=="No" else "cuda:0"
client = QdrantClient(
    url=db_url,
    api_key=api_key )



API_KEY_NAME = "api-key"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
VALID_API_KEY = api_key
async def verify_api_key(api_key_header: str = Depends(API_KEY_HEADER)):
    if api_key_header == VALID_API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )


model_name = "vidore/colqwen2.5-v0.2"
model = ColQwen2_5.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,  # "cuda:0" for GPU, "cpu" for CPU, or "mps" for Apple m series
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()
processor = ColQwen2_5_Processor.from_pretrained(model_name)


class Query(BaseModel):
    text: str

async def fetch_data_from_generator(url: str, data: dict = None):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=data, timeout=60.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error calling generator: {str(e)}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Connection error while calling generator: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error while calling generator: {str(e)}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup")
    yield
    print("Application shutdown")

@app.post("/retriver/")
async def retriver(query: Query, api_key: str = Depends(verify_api_key)):
    try:
        query_text = query.text
        with torch.no_grad():
            batch_query = processor.process_queries([query_text]).to(
                model.device
            )
            query_embedding = model(**batch_query)
        multivector_query = query_embedding[0].cpu().float().numpy().tolist()
        search_result = client.query_points(
            collection_name="vw",
            query=multivector_query,
            limit=10,
            timeout=100,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams( ignore=False,  rescore=True, oversampling=2.0 )
            )
        )
        data = {"index":[x.id for x in search_result.points[:7] ], "query": query_text}
        # print(search_result)
        # return {"response": data}

        response = await fetch_data_from_generator(generator_url,data)
        return {"response": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8444)