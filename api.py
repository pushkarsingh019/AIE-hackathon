from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from local_paper_qa.service import LocalPaperQA


app = FastAPI(title="Local Paper QA")
qa = LocalPaperQA()


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/papers")
def papers():
    return {"papers": [asdict(paper) for paper in qa.ensure_index()]}


@app.post("/papers")
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported"}
    safe_name = os.path.basename(file.filename)
    destination = qa.papers_dir / safe_name
    destination.write_bytes(await file.read())
    papers = qa.ensure_index(force=True)
    return {"papers": len(papers), "chunks": sum(len(p.chunks) for p in papers)}


@app.post("/reindex")
def reindex():
    papers = qa.ensure_index(force=True)
    return {"papers": len(papers), "chunks": sum(len(p.chunks) for p in papers)}


@app.post("/ask")
def ask(request: AskRequest):
    return qa.ask(request.question)
