# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1Z93syImskQ2_U1NPPJRfUjBQeJKmBZJe")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
  
     labels[0]: {
       "texts": ["코로나는 2020년대를 혼란에 빠뜨린 병균입니다"],
    #   "images": ["data:image/jpeg;base64,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"],
     "videos": ["https://www.youtube.com/watch?v=U9XJMWezy5k"]
     },
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
