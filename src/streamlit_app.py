import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import faiss
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModelWithProjection, CLIPTokenizer
import timm

import sys
sys.path.append(os.path.dirname(__file__))
from multimodal_w8a8_smoothquant import SmoothQuantWrapper

# Configuration
MODEL_NAME = 'convnextv2_nano.fcmae_ft_in1k'
LLM_NAME = "Qwen/Qwen1.5-0.5B"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
VISION_CKPT = "./models/hf_w8a8_smoothquant/smoothquant_w8a8.pth"
PROJ_CKPT = "./checkpoints/vlm_projection/projection_head_epoch_1.pth"
FAISS_INDEX_PATH = "./data/faiss_db/general_clip.index"
FAISS_META_PATH = "./data/faiss_db/general_metadata.txt"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set page config
st.set_page_config(page_title="General VLM Explorer", page_icon="VLM", layout="wide")

# Load Models
@st.cache_resource(show_spinner="Loading Models...")
def load_models():
    # 1. Vision Encoder
    vision_encoder = timm.create_model(MODEL_NAME, pretrained=False)
    in_features = vision_encoder.head.fc.in_features
    vision_encoder.head.fc = nn.Linear(in_features, 512)
    
    for name, module in dict(vision_encoder.named_modules()).items():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and "head" not in name:
            dummy_scale = torch.ones(module.in_channels if isinstance(module, nn.Conv2d) else module.in_features)
            sq_layer = SmoothQuantWrapper(module, dummy_scale)
            parts = name.split('.')
            parent = vision_encoder
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], sq_layer)
            
    if os.path.exists(VISION_CKPT):
        ckpt = torch.load(VISION_CKPT, map_location='cpu')
        vision_encoder.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    vision_encoder.head.fc = nn.Identity()
    vision_encoder = vision_encoder.bfloat16().to(DEVICE).eval()
    
    # 2. LLM
    llm = AutoModelForCausalLM.from_pretrained(LLM_NAME, torch_dtype=torch.bfloat16).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Projection Head
    projection_head = nn.Linear(in_features, llm.config.hidden_size).to(DEVICE).bfloat16()
    if os.path.exists(PROJ_CKPT):
        projection_head.load_state_dict(torch.load(PROJ_CKPT, map_location=DEVICE))
    
    # Preprocessing
    _dummy = timm.create_model(MODEL_NAME, pretrained=False)
    data_config = timm.data.resolve_model_data_config(_dummy)
    image_transform = timm.data.create_transform(**data_config, is_training=False)
    
    # CLIP
    clip_tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_ID)
    clip_text_model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL_ID).bfloat16().to(DEVICE).eval()
    
    # FAISS
    index = faiss.read_index(FAISS_INDEX_PATH) if os.path.exists(FAISS_INDEX_PATH) else None
    metadata = []
    if os.path.exists(FAISS_META_PATH):
        with open(FAISS_META_PATH, "r") as f:
            metadata = [line.strip() for line in f.readlines()]
            
    return vision_encoder, llm, tokenizer, projection_head, image_transform, clip_tokenizer, clip_text_model, index, metadata

vision_encoder, llm, tokenizer, projection_head, image_transform, clip_tokenizer, clip_text_model, index, metadata = load_models()

# Streamlit UI
st.title("Omni-Modal VLM Explorer")
st.markdown("W8A8 ConvNeXt-Nano와 Qwen 0.5B를 결합한 경량 VLM입니다. 이미지를 설명하거나 질문에 답변하고, 텍스트로 이미지를 검색할 수 있습니다.")

tab1, tab2, tab3 = st.tabs(["Image-to-Text", "Text-to-Image", "Text-to-Text"])

def generate_with_image(image, prompt, max_new_tokens=100, temperature=0.7):
    pixel_values = image_transform(image).unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
    encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        vision_features = vision_encoder(pixel_values)
        image_embeds = projection_head(vision_features).unsqueeze(1)
        text_embeds = llm.get_input_embeddings()(encoded["input_ids"])
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        attention_mask = encoded["attention_mask"]
        image_attention_mask = torch.ones(attention_mask.shape[0], 1, dtype=attention_mask.dtype, device=DEVICE)
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        outputs = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

with tab1:
    st.subheader("이미지 설명 및 질문 응답")
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="업로드된 이미지", use_container_width=True)

    with col2:
        mode = st.radio("모드 선택", ["캡셔닝", "VQA"], horizontal=True)
        if mode == "캡셔닝":
            prompt = st.text_input("요청", value="이 이미지를 자세히 설명해 주십시오:")
        else:
            prompt = st.text_input("질문", value="이 이미지에서 가장 눈에 띄는 물체는 무엇입니까?")

        if st.button("분석 시작") and uploaded_file is not None:
            with st.spinner("이미지 임베딩을 계산 중입니다."):
                full_prompt = f"<image>\n{prompt}\n"
                response = generate_with_image(image, full_prompt, max_new_tokens=120)
            st.write(response)

with tab2:
    st.subheader("텍스트 기반 이미지 검색")
    query = st.text_input("검색할 이미지의 특징을 영어로 입력하세요")
    if st.button("검색") and query:
        if index is None:
            st.error("FAISS 인덱스가 로드되지 않았습니다.")
        else:
            with st.spinner("텍스트 임베딩을 계산 중입니다."):
                inputs = clip_tokenizer(query, return_tensors="pt", padding=True).to(DEVICE)
                with torch.no_grad():
                    t_feat = F.normalize(clip_text_model(**inputs).text_embeds, p=2, dim=-1).cpu().float().numpy()

                D, I = index.search(t_feat, 4)

                cols = st.columns(4)
                for c_idx, col in enumerate(cols):
                    match_idx = I[0][c_idx]
                    if match_idx < len(metadata):
                        img_path = metadata[match_idx]
                        try:
                            res_img = Image.open(img_path)
                            col.image(res_img, caption=f"유사도: {D[0][c_idx]:.4f}", use_container_width=True)
                        except Exception:
                            col.write("이미지를 찾을 수 없습니다.")

with tab3:
    st.subheader("텍스트 전용 대화")
    user_text = st.text_area("입력", height=120)
    if st.button("응답 생성") and user_text:
        with st.spinner("모델이 응답을 생성 중입니다."):
            encoded = tokenizer(user_text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = llm.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=160,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(response)

st.sidebar.markdown("### Architecture Analytics")
st.sidebar.markdown("""
- **Vision:** ConvNeXt-Nano (W8A8 PTQ)
- **Vision Size:** 14.9 MB
- **LLM:** Qwen1.5-0.5B (BFloat16)
- **FAISS Index:** 1,000 Conceptual Captions
- **Speed:** ~100 FPS (Vision)
""")
