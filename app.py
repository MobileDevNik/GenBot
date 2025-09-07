import io
import os
import base64
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import streamlit as st
from PIL import Image

# Lazy imports for SDKs so the app loads even if some SDKs aren't installed
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore

try:
    from groq import Groq  # type: ignore
except Exception:  # pragma: no cover
    Groq = None  # type: ignore

try:
    import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation  # type: ignore
    from stability_sdk import client as stability_client  # type: ignore
except Exception:  # pragma: no cover
    stability_client = None  # type: ignore
    generation = None  # type: ignore


# -------------------------
# Utilities & UI helpers
# -------------------------

st.set_page_config(page_title="Multi-Modal AI Studio", page_icon="🤖", layout="wide")

PRIMARY_BG = "#0B1220"
CARD_BG = "#121A2B"
ACCENT = "#7C9DFF"
TEXT = "#E7ECFF"
MUTED = "#9AA7C7"

STYLE = f"""
    <style>
    .main {{ background: linear-gradient(180deg, {PRIMARY_BG} 0%, #0E1526 100%); }}
    .stChatFloatingInputContainer {{ background: {CARD_BG}; border: 1px solid #1D2A44; }}
    .bubble-user {{ background: #1C253B; border-radius: 16px; padding: 12px 14px; color: {TEXT}; }}
    .bubble-assistant {{ background: #121C33; border-radius: 16px; padding: 12px 14px; color: {TEXT}; border: 1px solid #1E2A4A; }}
    .pill {{ background: #0F1A33; color: {ACCENT}; border: 1px solid #22325A; padding: 3px 8px; border-radius: 999px; font-size: 12px; }}
    .subtle {{ color: {MUTED}; font-size: 13px; }}
    .section-card {{ background: {CARD_BG}; border: 1px solid #1D2A44; border-radius: 16px; padding: 16px; }}
    .divider {{ height: 1px; background: #203055; margin: 8px 0 16px 0; }}
    .accent-title {{ color: {TEXT}; }}
    a, a:visited {{ color: {ACCENT}; }}
    </style>
"""

st.markdown(STYLE, unsafe_allow_html=True)


def b64_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def section_header(title: str, subtitle: Optional[str] = None):
    st.markdown(f"<h3 class='accent-title'>{title}</h3>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='subtle'>{subtitle}</div>", unsafe_allow_html=True)


# -------------------------
# Model Adapters
# -------------------------

@dataclass
class ModelCapabilities:
    text: bool = True
    image: bool = False
    vision_qa: bool = False
    transcribe: bool = False


class BaseAdapter:
    name: str = "Base"
    capabilities = ModelCapabilities()

    def generate_text(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def generate_image(self, prompt: str, **kwargs) -> Image.Image:
        raise NotImplementedError

    def vision_qa(self, image: Image.Image, question: str, **kwargs) -> str:
        raise NotImplementedError

    def transcribe(self, audio_bytes: bytes, mime: str) -> str:
        raise NotImplementedError


# ---------- OpenAI ----------
class OpenAIAdapter(BaseAdapter):
    name = "OpenAI"
    capabilities = ModelCapabilities(text=True, image=True, vision_qa=True, transcribe=True)

    def __init__(self, api_key: Optional[str]):
        if not OpenAI:
            raise RuntimeError("openai SDK not installed. Run: pip install openai")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        self.client = OpenAI(api_key=api_key)
        # sensible defaults; user can override via sidebar
        self.text_model = st.session_state.get("openai_text_model", "gpt-4o-mini")
        self.image_model = st.session_state.get("openai_image_model", "gpt-image-1")
        self.vision_model = st.session_state.get("openai_vision_model", "gpt-4o-mini")

    def generate_text(self, prompt: str, **kwargs) -> str:
        model = kwargs.get("model", self.text_model)
        try:
            resp = self.client.responses.create(
                model=model,
                input=prompt,
            )
            return resp.output_text
        except Exception as e:
            raise RuntimeError(f"OpenAI text error: {e}")

    def generate_image(self, prompt: str, **kwargs) -> Image.Image:
        model = kwargs.get("model", self.image_model)
        size = kwargs.get("size", "1024x1024")
        try:
            resp = self.client.images.generate(model=model, prompt=prompt, size=size)
            b64 = resp.data[0].b64_json
            return Image.open(io.BytesIO(base64.b64decode(b64)))
        except Exception as e:
            raise RuntimeError(f"OpenAI image error: {e}")

    def vision_qa(self, image: Image.Image, question: str, **kwargs) -> str:
        model = kwargs.get("model", self.vision_model)
        try:
            img_b64 = b64_image(image)
            resp = self.client.responses.create(
                model=model,
                input=[
                    {"role": "user", "content": [
                        {"type": "input_text", "text": question},
                        {"type": "input_image", "image": img_b64},
                    ]}
                ],
            )
            return resp.output_text
        except Exception as e:
            raise RuntimeError(f"OpenAI vision error: {e}")

    def transcribe(self, audio_bytes: bytes, mime: str) -> str:
        try:
            with io.BytesIO(audio_bytes) as buf:
                buf.name = f"audio.{mime.split('/')[-1]}"
                buf.seek(0)
                resp = self.client.audio.transcriptions.create(model="whisper-1", file=buf)
            return resp.text
        except Exception as e:
            raise RuntimeError(f"OpenAI transcription error: {e}")


# ---------- Google Gemini ----------
class GeminiAdapter(BaseAdapter):
    name = "Google Gemini"
    capabilities = ModelCapabilities(text=True, image=True, vision_qa=True, transcribe=False)

    def __init__(self, api_key: Optional[str]):
        if not genai:
            raise RuntimeError("google-generativeai SDK not installed. Run: pip install google-generativeai")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is missing.")
        genai.configure(api_key=api_key)
        self.text_model_name = st.session_state.get("gemini_text_model", "gemini-1.5-pro")
        self.vision_model_name = st.session_state.get("gemini_vision_model", "gemini-1.5-pro")
        self.imagen_model_name = st.session_state.get("gemini_image_model", "imagen-3.0")  # may require access

    def generate_text(self, prompt: str, **kwargs) -> str:
        model = kwargs.get("model", self.text_model_name)
        try:
            m = genai.GenerativeModel(model)
            resp = m.generate_content(prompt)
            return resp.text or ""
        except Exception as e:
            raise RuntimeError(f"Gemini text error: {e}")

    def generate_image(self, prompt: str, **kwargs) -> Image.Image:
        model = kwargs.get("model", self.imagen_model_name)
        try:
            m = genai.GenerativeModel(model)
            resp = m.generate_images(prompt=prompt)
            # take first image
            if not resp.generated_images:
                raise RuntimeError("No image returned.")
            img_data = resp.generated_images[0]
            # google SDK may return PIL image directly or bytes
            if isinstance(img_data, Image.Image):
                return img_data
            # assume bytes
            return Image.open(io.BytesIO(img_data))
        except Exception as e:
            raise RuntimeError("Gemini/Imagen image error: {}".format(e))

    def vision_qa(self, image: Image.Image, question: str, **kwargs) -> str:
        model = kwargs.get("model", self.vision_model_name)
        try:
            m = genai.GenerativeModel(model)
            # Gemini accepts PIL Images directly
            resp = m.generate_content([question, image])
            return getattr(resp, "text", "")
        except Exception as e:
            raise RuntimeError(f"Gemini vision error: {e}")


# ---------- Groq ----------
class GroqAdapter(BaseAdapter):
    name = "Groq"
    capabilities = ModelCapabilities(text=True, image=False, vision_qa=True, transcribe=True)

    def __init__(self, api_key: Optional[str]):
        if not Groq:
            raise RuntimeError("groq SDK not installed. Run: pip install groq")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing.")
        self.client = Groq(api_key=api_key)
        self.text_model = st.session_state.get("groq_text_model", "llama-3.1-70b-versatile")
        self.vision_model = st.session_state.get("groq_vision_model", "llama-3.2-11b-vision-preview")

    def generate_text(self, prompt: str, **kwargs) -> str:
        model = kwargs.get("model", self.text_model)
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.3),
            )
            return resp.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq text error: {e}")

    def vision_qa(self, image: Image.Image, question: str, **kwargs) -> str:
        model = kwargs.get("model", self.vision_model)
        try:
            # Encode image as base64 for Groq vision models
            img_b64 = b64_image(image)
            resp = self.client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                }],
            )
            return resp.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq vision error: {e}")

    def transcribe(self, audio_bytes: bytes, mime: str) -> str:
        try:
            file_obj = io.BytesIO(audio_bytes)
            file_obj.name = f"audio.{mime.split('/')[-1]}"
            file_obj.seek(0)
            resp = self.client.audio.transcriptions.create(
                file=(file_obj.name, file_obj, mime),
                model="whisper-large-v3-turbo",
            )
            return resp.text
        except Exception as e:
            raise RuntimeError(f"Groq transcription error: {e}")


# ---------- Stability (Images only) ----------
class StabilityAdapter(BaseAdapter):
    name = "Stability (SD)"
    capabilities = ModelCapabilities(text=False, image=True, vision_qa=False, transcribe=False)

    def __init__(self, api_key: Optional[str]):
        if not stability_client:
            raise RuntimeError("stability-sdk not installed. Run: pip install stability-sdk")
        if not api_key:
            raise RuntimeError("STABILITY_API_KEY is missing.")
        self.client = stability_client.StabilityInference(
            key=api_key, engine="stable-diffusion-xl-1024-v1-0"
        )

    def generate_image(self, prompt: str, **kwargs) -> Image.Image:
        try:
            answers = self.client.generate(
                prompt=prompt,
                cfg_scale=kwargs.get("cfg_scale", 7.0),
                steps=kwargs.get("steps", 30),
                width=kwargs.get("width", 1024),
                height=kwargs.get("height", 1024),
            )
            for answer in answers:
                for artifact in answer.artifacts:
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        return Image.open(io.BytesIO(artifact.binary))
            raise RuntimeError("No image returned from Stability API.")
        except Exception as e:
            raise RuntimeError(f"Stability image error: {e}")


# -------------------------
# Adapter Factory
# -------------------------

def build_text_adapter(provider: str):
    if provider == "OpenAI":
        return OpenAIAdapter(os.getenv("OPENAI_API_KEY"))
    if provider == "Google Gemini":
        return GeminiAdapter(os.getenv("GOOGLE_API_KEY"))
    if provider == "Groq":
        return GroqAdapter(os.getenv("GROQ_API_KEY"))
    raise RuntimeError("Unsupported text provider selected.")


def build_image_adapter(provider: str):
    if provider == "OpenAI":
        return OpenAIAdapter(os.getenv("OPENAI_API_KEY"))
    if provider == "Google Imagen":
        return GeminiAdapter(os.getenv("GOOGLE_API_KEY"))
    if provider == "Stability (SD)":
        return StabilityAdapter(os.getenv("STABILITY_API_KEY"))
    raise RuntimeError("Unsupported image provider selected.")


# -------------------------
# Sidebar: Keys, Models, Settings
# -------------------------
with st.sidebar:
    st.markdown("## 🔐 API Keys")
    st.caption("Keys are read from environment by default. You can paste here for this session only.")
    openai_key = st.text_input("OPENAI_API_KEY", type="password")
    google_key = st.text_input("GOOGLE_API_KEY", type="password")
    groq_key = st.text_input("GROQ_API_KEY", type="password")
    stability_key = st.text_input("STABILITY_API_KEY (optional)", type="password")

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    if stability_key:
        os.environ["STABILITY_API_KEY"] = stability_key

    st.markdown("---")
    st.markdown("## ⚙️ Model Picks")
    text_provider = st.selectbox("Text Model Provider", ["OpenAI", "Google Gemini", "Groq"], index=0)
    image_provider = st.selectbox("Image Generation Provider", ["OpenAI", "Google Imagen", "Stability (SD)"], index=0)
    vision_provider = st.selectbox("Vision Q&A Provider", ["OpenAI", "Google Gemini", "Groq"], index=0)

    st.markdown("---")
    st.caption("UI tips: Use tabs below for Chat, Image Gen, Visual Q&A, and Voice.")


# -------------------------
# Tabs
# -------------------------
chat_tab, image_tab, vqa_tab, voice_tab = st.tabs(["💬 Chat", "🎨 Image Generation", "🖼️ Visual Q&A", "🎙️ Voice → Text"])


# -------------------------
# Chat Tab (Text Generation)
# -------------------------
with chat_tab:
    section_header("Chat with AI", f"Provider: {text_provider}")

    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts: {role, content}

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"]=="user" else "🤖"):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Type your question…")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                try:
                    adapter = build_text_adapter(text_provider)
                    response = adapter.generate_text(user_prompt)
                except Exception as e:
                    response = f"❌ {e}"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


# -------------------------
# Image Generation Tab
# -------------------------
with image_tab:
    section_header("Text → Image", f"Provider: {image_provider}")
    col1, col2 = st.columns([2,1])
    with col1:
        img_prompt = st.text_area("Describe the image you want", height=120, placeholder="A cozy reading nook with plants and warm light…")
        gen = st.button("Generate Image", use_container_width=True)
    with col2:
        size = st.selectbox("Size", ["1024x1024", "512x512", "2048x2048"], index=0)

    if gen:
        with st.spinner("Painting pixels…"):
            try:
                adapter = build_image_adapter(image_provider)
                if isinstance(adapter, OpenAIAdapter):
                    img = adapter.generate_image(img_prompt, size=size)
                else:
                    img = adapter.generate_image(img_prompt)
                st.image(img, caption="Generated Image", use_column_width=True)
                st.success("Done ✅")
            except Exception as e:
                st.error(str(e))


# -------------------------
# Visual Q&A Tab
# -------------------------
with vqa_tab:
    section_header("Ask about an Image", f"Provider: {vision_provider}")
    up_col, q_col = st.columns([1,2])
    with up_col:
        uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp"])
        if uploaded:
            try:
                img = Image.open(uploaded).convert("RGB")
                st.image(img, caption="Your Image", use_column_width=True)
            except Exception as e:
                st.error(f"Could not open image: {e}")
                img = None
        else:
            img = None
    with q_col:
        question = st.text_area("Your question about this image", height=100, placeholder="What is happening here?")
        ask = st.button("Ask", use_container_width=True)

    if ask:
        if not img:
            st.error("Please upload an image first.")
        elif not question.strip():
            st.error("Please type a question.")
        else:
            with st.spinner("Looking closely…"):
                try:
                    if vision_provider == "OpenAI":
                        adapter = OpenAIAdapter(os.getenv("OPENAI_API_KEY"))
                    elif vision_provider == "Google Gemini":
                        adapter = GeminiAdapter(os.getenv("GOOGLE_API_KEY"))
                    else:
                        adapter = GroqAdapter(os.getenv("GROQ_API_KEY"))
                    answer = adapter.vision_qa(img, question)
                    st.success("Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(str(e))


# -------------------------
# Voice → Text Tab (Speech Input)
# -------------------------
with voice_tab:
    section_header("Speak to the Bot", "Upload or record audio; we'll transcribe it.")

    st.markdown("<div class='subtle'>Tip: Install <code>streamlit-mic-recorder</code> to record directly in the browser, or upload an audio file below.</div>", unsafe_allow_html=True)

    # Optional mic recorder component (if installed). We won't hard-depend on it.
    audio_bytes = None
    try:
        from streamlit_mic_recorder import mic_recorder, speech_to_text
        st.caption("🎤 Mic Recorder (Beta)")
        mic_result = mic_recorder(start_prompt="Start recording", stop_prompt="Stop", just_once=True, use_container_width=True)
        if mic_result and mic_result.get('bytes'):
            audio_bytes = mic_result['bytes']
            st.audio(audio_bytes)
    except Exception:
        pass  # component not installed; ignore

    st.markdown("**Or upload an audio file**")
    up_audio = st.file_uploader("Audio (mp3, wav, m4a)", type=["mp3","wav","m4a","mp4","mpeg","webm"])
    if up_audio:
        audio_bytes = up_audio.read()
        st.audio(audio_bytes)

    colA, colB = st.columns(2)
    with colA:
        st.caption("Transcription Provider")
        asr_provider = st.selectbox("ASR Provider", ["OpenAI", "Groq"], index=0)
    with colB:
        mime = st.selectbox("Audio MIME (hint)", ["audio/mpeg","audio/wav","audio/mp4","audio/x-m4a","audio/webm"], index=0)

    go = st.button("Transcribe", use_container_width=True)

    if go:
        if not audio_bytes:
            st.error("Please record or upload audio first.")
        else:
            with st.spinner("Transcribing…"):
                try:
                    if asr_provider == "OpenAI":
                        adapter = OpenAIAdapter(os.getenv("OPENAI_API_KEY"))
                    else:
                        adapter = GroqAdapter(os.getenv("GROQ_API_KEY"))
                    text = adapter.transcribe(audio_bytes, mime)
                    st.success("Transcript:")
                    st.write(text)
                except Exception as e:
                    st.error(str(e))


# -------------------------
# Footer / Help
# -------------------------
with st.expander("ℹ️ Help & Notes"):
    st.markdown(
        """
        **Why multiple providers?** Different models excel at different tasks. Pick the best for your use case.

        **Privacy**: Inputs are sent to the selected provider's API. Review their data policies.

        **Troubleshooting**
        - If you see *SDK not installed* errors, run `pip install` for the missing library.
        - If you see *API key missing*, add the key in the sidebar or as an environment variable.
        - Google Imagen access may require additional enablement; if you lack access, try OpenAI or Stability for images.
        - For mic recording, install `streamlit-mic-recorder` or simply upload an audio file.
        """
    )