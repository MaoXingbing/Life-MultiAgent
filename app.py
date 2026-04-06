import streamlit as st
from agent.react_agent import ReactAgent
from agent.tools.draw_tools import draw_image_with_display
import time
import re
import os

st.set_page_config(
    page_title="基于AI的图像生成系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stChatMessage[data-testid="assistant-message"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .feature-title {
        font-weight: 600;
        color: #333;
        margin-bottom: 0.3rem;
    }
    .feature-desc {
        color: #666;
        font-size: 0.9rem;
    }
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem 2rem;
        border-top: 1px solid #e9ecef;
        z-index: 100;
    }
    .chat-container {
        padding-bottom: 100px;
    }
    .generated-image {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .draw-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .draw-title {
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🤖 基于AI的图像生成系统</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">基于AI的图像生成系统</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ 功能面板")
    st.markdown("---")
    
    st.markdown("#### 🎯 快捷功能")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌤️ 天气查询", use_container_width=True):
            st.session_state["quick_action"] = "今天天气怎么样？"
        if st.button("📍 位置查询", use_container_width=True):
            st.session_state["quick_action"] = "我在哪个城市？"
    with col2:
        if st.button("📊 使用报告", use_container_width=True):
            st.session_state["quick_action"] = "给我生成使用报告"
        if st.button("❓ 帮助", use_container_width=True):
            st.session_state["quick_action"] = "你能帮我做什么？"
    
    st.markdown("---")
    st.markdown("#### 🎨 AI 绘画")
    draw_prompt = st.text_input("输入绘画描述", placeholder="例如：画一颗树")
    if st.button("🎨 生成图片", use_container_width=True):
        if draw_prompt:
            st.session_state["draw_action"] = draw_prompt
        else:
            st.warning("请输入绘画描述")
    
    st.markdown("---")
    st.markdown("#### 📝 对话管理")
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state["message"] = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.8rem;">
        Powered by LangChain<br>
        Version 1.0.0
    </div>
    """, unsafe_allow_html=True)

if "agent" not in st.session_state:
    with st.spinner("正在初始化智能体..."):
        st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

if "quick_action" in st.session_state and st.session_state["quick_action"]:
    prompt = st.session_state["quick_action"]
    st.session_state["quick_action"] = None
else:
    prompt = None

if "draw_action" in st.session_state and st.session_state["draw_action"]:
    draw_prompt_value = st.session_state["draw_action"]
    st.session_state["draw_action"] = None
    
    with st.spinner(f"正在生成: {draw_prompt_value}"):
        try:
            file_path, base64_str = draw_image_with_display(draw_prompt_value, return_base64=True)
            
            st.image(f"data:image/png;base64,{base64_str}", caption=draw_prompt_value, use_container_width=True)
            
            st.session_state["message"].append({
                "role": "assistant", 
                "content": f"🎨 已生成图片: {draw_prompt_value}",
                "image_base64": base64_str
            })
            
        except Exception as e:
            st.error(f"❌ 生成失败: {str(e)}")

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "image_base64" in message:
            st.markdown("---")
            st.image(f"data:image/png;base64,{message['image_base64']}", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

user_input = st.chat_input("请输入您的问题...")

if prompt or user_input:
    actual_prompt = prompt or user_input
    
    with st.chat_message("user"):
        st.markdown(actual_prompt)
    st.session_state["message"].append({"role": "user", "content": actual_prompt})
    
    response_content = ""
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            res_stream = st.session_state["agent"].execute_stream(actual_prompt)
            
            placeholder = st.empty()
            full_response = ""
            
            for chunk in res_stream:
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            response_content = full_response
            
            image_path_match = re.search(r'图片已生成并保存到:\s*(.+)', full_response)
            if image_path_match:
                image_path = image_path_match.group(1).strip()
                if os.path.exists(image_path):
                    st.markdown("---")
                    st.markdown("### 🖼️ 生成的图片")
                    st.image(image_path, use_container_width=True)
                    
                    import base64
                    with open(image_path, 'rb') as f:
                        img_base64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    st.session_state["message"][-1] = {
                        "role": "assistant", 
                        "content": full_response,
                        "image_base64": img_base64
                    }
    
    if "image_path_match" not in dir() or not image_path_match:
        st.session_state["message"].append({"role": "assistant", "content": response_content})
    
    st.rerun()

if len(st.session_state["message"]) == 0:
    st.markdown("---")
    st.markdown("### 👋 欢迎使用基于AI的图像生成系统！")
    st.markdown("我可以帮您：")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌤️</div>
            <div class="feature-title">天气查询</div>
            <div class="feature-desc">查询各地天气信息</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">使用报告</div>
            <div class="feature-desc">生成使用数据报告</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">智能检索</div>
            <div class="feature-desc">RAG知识库检索</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎨</div>
            <div class="feature-title">AI 绘画</div>
            <div class="feature-desc">文本生成图片</div>
        </div>
        """, unsafe_allow_html=True)
