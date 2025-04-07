from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import base64
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction, SentenceTransformerEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from chromadb import Client
from io import BytesIO
import os

app = FastAPI(title="Text-to-Image Search API")

db_path = "imageSearch"
client = chromadb.PersistentClient(path=db_path)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
embedding_func = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

class ImageItem(BaseModel):
    id: str
    base64_image: str
    team: str

class QueryItem(BaseModel):
    query: str
    top_k: int = 3
    team: str

class ImageQuery(BaseModel):
    base64_image: str
    top_k: int = 3
    team: str

def get_image_embedding_from_base64(base64_str):
    clean_b64 = base64_str.replace('\n', '').replace(' ', '')
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224)) 
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings[0].numpy()

def get_text_embedding(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**inputs)
    return text_embedding[0].numpy()


@app.post("/add-image")
def add_image(item: ImageItem):
    try:
        collection = client.get_or_create_collection(
    name=item.team)

        embedding = get_image_embedding_from_base64(item.base64_image)
        collection.add(
            ids=[item.id],
            embeddings=[embedding.tolist()],
            metadatas=[{"team": item.team}],
            documents=["image"]  # required for text search
        )
        return {"message": f"Image '{item.id}' added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/search")
def search_images(query: QueryItem):
    try:
        collection = client.get_or_create_collection(
            name=query.team
            
        )
        query_embedding = get_text_embedding(query.query)
        results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=query.top_k)
        matched = [
            {
                "id": result_id,
                "caption": metadata.get("team")
            }
            for result_id, metadata in zip(results["ids"][0], results["metadatas"][0])
        ]
        return {"results": matched}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/search-by-image")
def search_by_image(query: ImageQuery):
    try:
        # Embed the query image
        embedding = get_image_embedding_from_base64(query.base64_image)
        collection = client.get_or_create_collection(
            name=query.team)

        # Search in Chroma using this image embedding
        results = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=query.top_k
        )

        matched = [
            {
                "id": result_id,
                "metadata": metadata
            }
            for result_id, metadata in zip(results["ids"][0], results["metadatas"][0])
        ]
        return {"results": matched}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))