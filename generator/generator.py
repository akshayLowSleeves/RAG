from typing import List
import os, sys
import io
import base64
from contextlib import asynccontextmanager
from PIL import Image
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from openai import OpenAI
import fitz

app = FastAPI()


if len(sys.argv) < 2:
    print("No argument was passed to the script.")
    sys.exit(1)
openai_api_key = sys.argv[1]
data_path = "./data"
pdf_path = "./vw.pdf"

class InputData(BaseModel):
    query: str
    index: List[int] = Field(..., min_items=4, max_items=8)
class OpenAIResponse(BaseModel):
    text: str


def pil_to_base64(image: Image.Image, format_: str = "PNG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format_)
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format_.lower()};base64,{encoded_image}"
def pdf_to_images(pdf_path, output_folder, dpi=150):
    pdf_document = fitz.open(pdf_path)
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap(matrix=matrix)
        image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    
    return image_paths

async def load_images(folder_path: str) -> List[int]:
    image_list = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_list.append(Image.open(os.path.join(folder_path, filename)))
    return image_list


async def query_openai_with_images(query: str, images: List[str], instruction_prompt: str) -> str:
    try:
        image_payloads = [
            {"type": "image_url", "image_url": {"url": pil_to_base64(img), "detail": "auto"} }
            for img in images
        ]
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[
                { "role": "user", "content": [ {"type": "text", "text": f"{instruction_prompt}\n\nQuery: {query}"},  *image_payloads]}
            ], 
            max_tokens=300 )
        
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong with OpenAI side: {e}") from e


client = OpenAI(api_key=openai_api_key)
pdf_to_images(pdf_path,data_path)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup")
    yield
    print("Application shutdown")


@app.post("/query_images/", response_model=OpenAIResponse)
async def query_images_endpoint(input_data: InputData = Body(...)):
    try:
        image_ids = input_data.index
        query = input_data.query
        images = await load_images(data_path)
        final_images = [images[i] for i in image_ids]
        
        instruction_prompt = """You are a helpful assistant that answers user questions based solely on the content of 
        provided images. The images are pages of a PDF document in Spanish and may contain both text and visual elements 
        such as graphics or diagrams. Your responses must be accurate, concise, and derived exclusively from the information 
        visible in the images, without using external knowledge or making assumptions, and if the answer is not found in 
        the images, you must clearly indicate that the information is not available in the provided content. """

        openai_response = await query_openai_with_images(query, final_images, instruction_prompt)
        return {"text": openai_response}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8443)