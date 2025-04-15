import fitz
import os, time
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available
import torch


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

def load_images_from_folder(folder_path):
    image_list = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_list.append(Image.open(os.path.join(folder_path, filename)))
    return image_list


data_path = "./data"
pdf_path = "./vw.pdf"
pdf_to_images(pdf_path,data_path)

images = load_images_from_folder(data_path)

model_name = "vidore/colqwen2.5-v0.2"
model = ColQwen2_5.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
    ).eval()
processor = ColQwen2_5_Processor.from_pretrained(model_name)


tensors = []
for i in range(0,len(images),3): 
    ini = i
    fin = i+3
    if fin>len(images)-1:
        fin= len(images)
    print(ini,fin)
    st = time.time()
    batch_images = processor.process_images(images[ini:fin]).to(model.device)
    with torch.no_grad():
        image_embeddings = model(**batch_images)
    tensors.append(image_embeddings)
    
    del batch_images
    torch.cuda.empty_cache() 
    print(time.time()-st)
    time.sleep(2)

combined = torch.cat(tensors, dim=0)



torch.save(combined, './pdf_emb.pt')