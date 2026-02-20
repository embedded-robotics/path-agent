# Path-AGENT: Mimicking a Clinically Proven Diagnostic Workflow for Open-Ended Pathology Visual Question Answering

1. Aims and Objectives

Open-ended visual question answering (VQA) remains a challenging problem in digital histopathology because it requires identifying diverse tissue types by closely examining changes in cell morphology and organization in different body organs. A pathologist normally makes a diagnosis by firstly examining the whole-slide image (WSI) at different magnification scales to identify the prevalent tissue types and then analyzing how the underlying cell morphology/organization is different from a normal tissue sample within the same body organ. We mimic the same diagnostic workflow by proposing a distributed pathology agent architecture to efficiently handle open-ended VQA. In our proposed architecture, five agent systems are integrated to combine information from the textual medical knowledge base, contextually relevant patches of WSI, and parametric memory alignment of vision-language models to enhance the performance in answering open-ended visual questions about the histopathology.

2. Research Questions

The open-ended questions require a deep comprehension of both images and textual questions, and the relationship between visual objects and textual entities, especially the intricate domain knowledge in pathology. Therefore, we aim to address the following research questions:
•	Identifying key patches in pathology images using nuclei density and contextual relevance of each patch in WSI
•	Retrieving relevant domain knowledge (text) using the pathology image from a medical knowledge base
•	Performance improvement for open-ended visual question answers
•	Performance improvement for finetuned LLaVa-Med [1] after aligning on the objectives efficiently utilizing the knowledge of image, captions and patch summaries

3. Research Methodology

We introduce a distributed architecture consisting of five agent systems with a vision-language model fine-tuned on biomedical data (LLaVA-Med [1]) acting as the foundation model for reasoning. Firstly, Magnifier Agent tessellates the WSI into small patches; then dynamically selects the contextually relevant patches by looking at the density of nuclei (using HistoCartography [2]) and the attention score (using CHEIF [3]) of each patch (Fig. 1). Secondly, AI Pathologist Agent uses image-text contrastive learning based pre-trained model (CLIP [4]) to identify the contextually relevant captions associated with the WSI from a medical knowledge base separated by different body organs (Fig. 1). Third, for each patch, we pass the WSI annotated with the bounding box of the patch along with the open-ended question to ROI Agent which examines if the patch is useful to answer the query (Fig. 1). Fourth, Patch Agent is fed the actual extracted patch, relevant captions of the WSI and open-ended question, and it summarizes the contribution of each patch to answer the query (Fig. 1). The penultimate step is to give all the available information to a Critique Agent which then undergoes an N-round communication with both ROI and Patch agents to refine the summaries extracted from each individual patch (Fig. 1). Finally, these summaries, WSI and its captions, and the open-ended query is passed to a vision-language model (LLaVA-Med [1]) to generate the final answer (Fig. 2). To reduce the hallucinations, LLaVA-Med is fine-tuned using preference pairs via Direct Preference Optimization (DPO) [5] to align with the objective of efficiently utilizing the knowledge of image, captions and patch summaries (Fig. 1)

![Figure 1: Image Tessellation, Domain Knowledge Extraction, and Reasoning between different agents](https://github.com/embedded-robotics/path-agent/blob/master/PathAGENT.png)
<p style="text-align:center;">Figure 1: Image Tessellation, Domain Knowledge Extraction, and Reasoning between different agents.</p>

## 4. Runbook (Integrated Services)

Run components in this order, each in its own environment:

1. **CHIEF API** (`CHIEF_HeatMap_API`, Docker, port `8001`)
```bash
cd CHIEF_HeatMap_API
docker run --name chief_run --gpus all -it \
  -p 8001:8001 \
  -v "$(pwd):/app/CHIEF" \
  -v "/absolute/path/to/path-agent/svs_examples:/data" \
  -w /app/CHIEF \
  chiefcontainer/chief:v1.11 bash

pip install fastapi uvicorn
python3 chief_api.py
```

2. **Combined Patch API** (`Complete_Patch_Extraction_API`, venv, port `8003`)
```bash
cd Complete_Patch_Extraction_API
source .venv390/bin/activate
python complete_patch_extraction_api.py
```

3. **Retriever API** (`PathGenCLIP_Retriever`, optional, port `8000`)
```bash
cd PathGenCLIP_Retriever
python app.py
```

4. **Agentic Orchestrator** (`pathrag-agentic-starter`)
```bash
cd pathrag-agentic-starter
source .venv/bin/activate

export PATHRAG_USE_COMBINED_API=1
export PATHRAG_COMBINED_API_URL=http://localhost:8003/process
export PATHRAG_USE_RETRIEVER_API=1
export PATHRAG_RETRIEVER_API_URL=http://localhost:8000/retrieve_captions

python scripts/run_langgraph.py
```

Health checks:
```bash
curl http://localhost:8001/health
curl http://localhost:8003/health
```

References:
1.	Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 36, 2024.
2.	Guillaume Jaume, Pushpak Pati, Valentin Anklin, Antonio Foncubierta, and Maria Gabrani. Histocartography: A toolkit for graph analytics in digital pathology. Proceedings of the MICCAI Workshop on Computational Pathology, volume 156 of Proceedings of Machine Learning Research, pp. 117–128. PMLR, 27 Sep 2021.
3.	Xiyue Wang Et. al, A pathology foundation model for cancer diagnosis and prognosis prediction. Nature, 634(8035):970–978, September 2024. ISSN: 1476-4687. doi: 10.1038/s41586-024-07894-z.
4.	Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. URL https://arxiv.org/abs/2103.00020.
5.	Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2024. URL https://arxiv.org/abs/2305.18290.
