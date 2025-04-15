#!/bin/bash

pip install stamina

if [[ "$CUDA_AVAILAIBLE" == "yes" ]]; then
    echo "Turning Pdf to images and extracting embeddings and saving them to the data basea and launching the API."
    python indexer1.py
    python indexer2.py "$QDRANT__SERVICE__API_KEY" "$CUDA_AVAILAIBLE" 
    python retriver.py "$CUDA_AVAILAIBLE" "$QDRANT__SERVICE__API_KEY" 
else
    echo "No GPU so run indexer1.py on GPU resource and generate pdf_emb.pt file. Then upload it to retriver folder.\n Because indexer2.py expects that file."
    python indexer2.py "$QDRANT__SERVICE__API_KEY" "$CUDA_AVAILAIBLE"
    python retriver.py "$CUDA_AVAILAIBLE" "$QDRANT__SERVICE__API_KEY" 
fi


# tail -f /dev/null
