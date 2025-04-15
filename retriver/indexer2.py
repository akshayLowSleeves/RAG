from qdrant_client import QdrantClient
from qdrant_client.http import models
import torch, time
from dotenv import load_dotenv
import os, sys
from tqdm import tqdm 
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers.utils.import_utils import is_flash_attn_2_available
from PIL import Image
import stamina

time.sleep(10)

if len(sys.argv) < 3:
    print("No argument was passed to the script.")
    sys.exit(1)

api_key = sys.argv[1]
device = sys.argv[2]

client = QdrantClient(
    url="http://qdrant:6333",
    api_key=api_key,
)


filename = './pdf_emb.pt'
if not os.path.exists(filename):
    print(f"Error: File: '{filename}' not found in the current directory.")
    sys.exit(1)  
if device=="No":
    loaded_combined = torch.load(filename, map_location=torch.device('cpu'))
else:
    loaded_combined = torch.load(filename,  map_location=torch.device('cuda:0'))



collection_name = "vw"
if client.collection_exists(collection_name=collection_name):
    # client.delete_collection(collection_name=collection_name)
    pass
else: 
    client.create_collection(
        collection_name=collection_name,
        on_disk_payload=True,  
        vectors_config=models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            on_disk=True, 
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(
                always_ram=True 
                ),
            ),
        ),
    )


    @stamina.retry(on=Exception, attempts=3) 
    def upsert_to_qdrant(points):
        try:
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=False,
            )
        except Exception as e:
            print(f"Error during upsert1: {e}")
            return False
        return True




    with tqdm(total=len(loaded_combined), desc="Indexing Progress") as pbar:
        for j, embedding in enumerate(loaded_combined):
            points = []
            multivector = embedding.cpu().float().numpy().tolist()
            points.append( models.PointStruct( id=j, vector=multivector ) )
            pbar.update(1)

            try:
                upsert_to_qdrant(points)
            except Exception as e:
                print(f"Error during upsert number{j}: {e}")
            time.sleep(1)


    client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10),
    )