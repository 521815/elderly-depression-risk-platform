"""Elderly depression risk screening workbench."""

from __future__ import annotations

import io
import base64
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))

import backend

st.set_page_config(
    page_title="抑郁风险筛查工作台",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #18343b;
            --muted: #667b82;
            --line: #dce7e5;
            --panel: #ffffff;
            --canvas: #f3f7f6;
            --accent: #0f766e;
            --accent-strong: #0a5f59;
            --accent-soft: #e6f4f1;
            --amber: #a65f00;
            --amber-soft: #fff5df;
            --green: #11835f;
            --green-soft: #e8f7f1;
            --red: #c53d3d;
            --red-soft: #fff0ef;
            --shadow-sm: 0 1px 2px rgba(16, 24, 40, 0.05), 0 1px 3px rgba(16, 24, 40, 0.06);
            --shadow-md: 0 12px 28px rgba(22, 58, 63, 0.10);
        }
        .stApp {
            background:
                linear-gradient(180deg, #eaf3f1 0, var(--canvas) 250px, var(--canvas) 100%);
            color: var(--ink);
            font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Segoe UI", sans-serif;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stToolbar"] {
            background: transparent;
        }
        [data-testid="stToolbar"] button[data-testid="stBaseButton-header"] {
            display: none;
        }
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 4rem;
            max-width: 1560px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }
        h3 {
            font-size: 1.35rem;
            margin-bottom: 0.15rem;
        }
        p, li, .stMarkdown {
            color: var(--ink);
        }
        [data-testid="stCaptionContainer"] p {
            color: var(--muted);
            line-height: 1.55;
        }
        .st-key-app_header {
            background: linear-gradient(105deg, #123f3b 0%, #0e5d57 58%, #0d6f66 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 0.95rem 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 14px 32px rgba(15, 72, 69, 0.18);
        }
        .st-key-app_header [data-testid="stHorizontalBlock"] {
            align-items: center;
            gap: 0.65rem;
        }
        .app-header-title {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }
        .app-mark {
            width: 44px;
            height: 44px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.18);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 5px;
            box-shadow: none;
        }
        .app-mark img {
            display: block;
            width: 100%;
            height: 100%;
        }
        .app-name {
            color: #ffffff;
            font-size: 1.2rem;
            font-weight: 760;
            line-height: 1.25;
        }
        .app-sub {
            color: rgba(236, 253, 250, 0.76);
            font-size: 0.84rem;
        }
        .app-header-status {
            display: grid;
            grid-template-columns: repeat(3, max-content);
            align-items: center;
            gap: 0.38rem 0.42rem;
            justify-content: flex-end;
            padding-left: 0.65rem;
            border-left: 1px solid rgba(255, 255, 255, 0.14);
        }
        .st-key-app_header .status-pill {
            color: rgba(244, 255, 253, 0.9);
            background: rgba(255, 255, 255, 0.09);
            border-color: rgba(255, 255, 255, 0.14);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
            padding: 0.16rem 0.5rem;
            font-size: 0.76rem;
        }
        .st-key-app_header .status-ok {
            color: #d9fff0;
            background: rgba(44, 202, 149, 0.15);
            border-color: rgba(108, 229, 187, 0.22);
        }
        .st-key-app_header .status-wait {
            color: #fff0c7;
            background: rgba(245, 158, 11, 0.14);
            border-color: rgba(252, 211, 77, 0.22);
        }
        .st-key-app_header div[data-testid="stPopover"] button {
            width: 2.2rem;
            min-width: 2.2rem;
            height: 2.2rem;
            padding: 0;
            border-radius: 9px;
            border: 1px solid rgba(255, 255, 255, 0.20);
            background: rgba(255, 255, 255, 0.10);
            color: #ffffff;
            font-size: 1.18rem;
            line-height: 1;
        }
        .st-key-app_header div[data-testid="stPopover"] button svg {
            display: none;
        }
        .st-key-app_header div[data-testid="stPopover"] button:hover {
            background: rgba(255, 255, 255, 0.18);
            border-color: rgba(255, 255, 255, 0.34);
            color: #ffffff;
        }
        div[data-testid="stPopoverBody"] {
            min-width: 11rem;
            border-radius: 12px;
            padding: 0.65rem;
            box-shadow: 0 16px 34px rgba(18, 52, 56, 0.18);
        }
        div[data-testid="stPopoverBody"] button {
            justify-content: flex-start;
        }
        [data-testid="stSidebar"] {
            background: #fbfdfc;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebarContent"] {
            padding-top: 0.65rem;
            padding-inline: 0.35rem;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] {
            padding: 0.3rem 0 0.55rem;
        }
        [data-testid="stSidebar"] .stCaption {
            color: var(--muted);
        }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-left: 3px solid var(--accent);
            border-radius: 10px;
            padding: 0.7rem 0.85rem;
            min-height: 88px;
            height: 88px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: var(--shadow-sm);
            transition: box-shadow 0.15s ease, transform 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 750;
            line-height: 1.2;
        }
        .stButton button {
            min-height: 2.6rem;
            border-radius: 8px;
            border: 1px solid #d3e1df;
            background: var(--panel);
            color: var(--ink);
            font-weight: 600;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            transition: box-shadow 0.15s ease, transform 0.12s ease, border-color 0.15s ease;
        }
        .stButton button:hover {
            border-color: #9fcac4;
            color: var(--accent-strong);
            box-shadow: 0 8px 18px rgba(15, 118, 110, 0.11);
            transform: translateY(-1px);
        }
        .stButton button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff;
            box-shadow: 0 7px 16px rgba(15, 118, 110, 0.20);
        }
        .stButton button[kind="primary"]:hover {
            background: var(--accent-strong);
            border-color: var(--accent-strong);
            color: #ffffff;
        }
        [data-testid="stTabs"] {
            gap: 0.3rem;
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            display: flex;
            gap: 0.1rem;
            flex-wrap: wrap;
            overflow: visible;
            padding: 0.34rem;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid var(--line);
            border-radius: 10px;
            box-shadow: var(--shadow-sm);
        }
        [data-testid="stTabs"] button[data-baseweb="tab"] {
            height: 2.45rem;
            min-width: max-content;
            border-radius: 8px;
            padding: 0.35rem 0.55rem;
            color: var(--muted);
            font-size: 0.86rem;
            font-weight: 600;
            border: 1px solid transparent;
            background: transparent;
            transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        }
        [data-testid="stTabs"] button[data-baseweb="tab"]:hover {
            background: #edf6f4;
            color: var(--accent-strong);
        }
        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            background: var(--accent);
            color: #ffffff;
            border-color: var(--accent);
            box-shadow: 0 4px 10px rgba(15, 118, 110, 0.18);
        }
        [data-testid="stExpander"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            box-shadow: var(--shadow-sm);
            overflow: hidden;
        }
        [data-testid="stExpander"] summary {
            font-weight: 650;
            background: #f8fbfa;
        }
        [data-testid="stExpander"] summary:hover {
            color: var(--accent-strong);
        }
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            border-radius: 8px;
            border-color: var(--line);
            background: #fbfdfc;
        }
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 1px var(--accent);
        }
        div[data-baseweb="slider"] [role="slider"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12);
        }
        div[data-baseweb="radio"] div[role="radio"][aria-checked="true"] > div:first-child {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
        }
        div[data-baseweb="button-group"] button[aria-pressed="true"],
        div[data-baseweb="button-group"] button[aria-selected="true"] {
            background: var(--accent-soft) !important;
            border-color: var(--accent) !important;
            color: var(--accent-strong) !important;
        }
        .st-key-conversation_mode_selector [role="radiogroup"] {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .st-key-conversation_mode_selector [role="radiogroup"] label {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            min-height: 2.35rem;
            padding: 0.35rem 0.85rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: #f7fbfa;
            cursor: pointer;
            transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
        }
        .st-key-conversation_mode_selector [role="radiogroup"] label:hover {
            border-color: var(--accent);
            background: var(--accent-soft);
        }
        .st-key-conversation_mode_selector [role="radiogroup"] label:has(input:checked),
        .st-key-conversation_mode_selector [aria-checked="true"] {
            border-color: var(--accent);
            background: var(--accent);
        }
        .st-key-conversation_mode_selector [role="radiogroup"] label:has(input:checked) *,
        .st-key-conversation_mode_selector [aria-checked="true"] * {
            color: #ffffff !important;
        }
        .st-key-conversation_mode_selector [role="radiogroup"] label > div:first-child {
            display: none;
        }
        .st-key-main_nav [role="radiogroup"] {
            display: flex;
            flex-wrap: wrap;
            gap: 0.1rem;
            padding: 0.34rem;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid var(--line);
            border-radius: 10px;
            box-shadow: var(--shadow-sm);
        }
        .st-key-main_nav [role="radiogroup"] label {
            display: inline-flex;
            align-items: center;
            min-height: 2.45rem;
            padding: 0.35rem 0.68rem;
            border: 1px solid transparent;
            border-radius: 8px;
            color: var(--muted);
            font-size: 0.86rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        }
        .st-key-main_nav [role="radiogroup"] label:hover {
            background: #edf6f4;
            color: var(--accent-strong);
        }
        .st-key-main_nav [role="radiogroup"] label:has(input:checked),
        .st-key-main_nav [aria-checked="true"] {
            background: var(--accent);
            border-color: var(--accent);
            box-shadow: 0 4px 10px rgba(15, 118, 110, 0.18);
        }
        .st-key-main_nav [role="radiogroup"] label:has(input:checked) *,
        .st-key-main_nav [aria-checked="true"] * {
            color: #ffffff !important;
        }
        .st-key-main_nav [role="radiogroup"] label > div:first-child {
            display: none;
        }
        [data-testid="stChatMessage"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.5rem;
            box-shadow: var(--shadow-sm);
        }
        [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
            background: var(--accent);
        }
        [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
            background: #2f8f83;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }
        [data-testid="stPlotlyChart"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 0.3rem;
            box-shadow: var(--shadow-sm);
        }
        .stDownloadButton button,
        .stFileUploader section {
            border-radius: 8px;
        }
        .stDownloadButton button {
            min-height: 2.6rem;
            border: 1px solid #d3e1df;
            background: #fbfdfc;
            color: var(--accent-strong);
            font-weight: 600;
        }
        .stDownloadButton button:hover {
            border-color: var(--accent);
            background: var(--accent-soft);
            color: var(--accent-strong);
        }
        .risk-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.9rem;
            border: 1px solid currentColor;
            background: var(--panel);
            box-shadow: 0 4px 12px rgba(24, 52, 59, 0.08);
        }
        .status-pill {
            display: inline-block;
            white-space: nowrap;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid transparent;
        }
        .status-ok {
            color: var(--green);
            background: var(--green-soft);
            border-color: rgba(18, 128, 92, 0.18);
        }
        .status-wait {
            color: var(--amber);
            background: var(--amber-soft);
            border-color: rgba(183, 121, 31, 0.2);
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            min-height: 28px;
            margin: 0.42rem 0;
            gap: 0.5rem;
            padding: 0.18rem 0;
        }
        .status-label {
            color: var(--muted);
            font-size: 0.84rem;
        }
        .factor-row {
            display: grid;
            grid-template-columns: minmax(96px, auto) minmax(72px, auto) 1fr;
            gap: 0.5rem 0.7rem;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--line);
            min-height: 34px;
        }
        .factor-row:last-child {
            border-bottom: none;
        }
        .factor-label {
            color: var(--muted);
            font-size: 0.86rem;
        }
        .factor-value {
            color: var(--ink);
            font-weight: 650;
            font-size: 0.94rem;
            white-space: nowrap;
        }
        .factor-delta {
            color: var(--muted);
            font-size: 0.8rem;
            text-align: right;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .user-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 0.8rem;
            margin-bottom: 0.55rem;
            display: flex;
            gap: 0.75rem;
            align-items: center;
            min-height: 96px;
            box-shadow: var(--shadow-sm);
            transition: box-shadow 0.15s ease, transform 0.12s ease, border-color 0.15s ease;
        }
        .user-card:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }
        .user-card.active {
            border-color: var(--accent);
            box-shadow: 0 0 0 1px rgba(26, 111, 181, 0.22), var(--shadow-sm);
        }
        .user-avatar {
            width: 42px;
            height: 42px;
            border-radius: 50%;
            background: var(--accent);
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1rem;
            flex: 0 0 auto;
            box-shadow: 0 3px 8px rgba(15, 118, 110, 0.2);
        }
        .user-card-body {
            flex: 1;
            min-width: 0;
        }
        .user-card-name {
            font-weight: 650;
            font-size: 1rem;
            margin-bottom: 0.2rem;
        }
        .tag-pill {
            display: inline-block;
            padding: 0.12rem 0.55rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.35rem;
            border: 1px solid rgba(15, 118, 110, 0.16);
        }
        .risk-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.3rem;
        }
        .logo-lockup svg {
            width: 100%;
            height: auto;
            display: block;
        }
        .logo-lockup {
            margin-bottom: 0.6rem;
        }
        .doc-block {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 1rem 1.2rem;
            line-height: 1.7;
            box-shadow: var(--shadow-sm);
        }
        hr {
            border-color: var(--line);
        }
        .stAlert {
            border-radius: 10px;
            border: 1px solid var(--line);
            box-shadow: var(--shadow-sm);
        }
        @media (max-width: 720px) {
            .block-container {
                padding-inline: 0.85rem;
                padding-top: 0.9rem;
            }
            [data-testid="stExpandSidebarButton"] {
                position: fixed !important;
                top: 0.45rem !important;
                left: 0.6rem !important;
                z-index: 999999 !important;
                width: 2.45rem !important;
                height: 2.45rem !important;
                border: 1px solid var(--line) !important;
                border-radius: 8px !important;
                background: rgba(255, 255, 255, 0.96) !important;
                box-shadow: var(--shadow-sm) !important;
            }
            .st-key-app_header {
                padding: 0.85rem;
            }
            .app-name {
                font-size: 1.08rem;
            }
            .app-header-status {
                grid-template-columns: 1fr;
                justify-content: flex-start;
                padding-left: 0;
                border-left: 0;
                padding-top: 0.45rem;
                border-top: 1px solid rgba(255, 255, 255, 0.14);
            }
            [data-testid="stTabs"] button[data-baseweb="tab"] {
                padding-inline: 0.7rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_care_css() -> None:
    care_mode = st.session_state.get("care_mode_toggle", False)
    if not care_mode:
        return
    st.markdown(
        """
        <style>
        .stApp {
            font-size: 1.22rem !important;
        }
        .block-container {
            max-width: 1600px;
        }
        .app-name {
            font-size: 1.5rem;
        }
        p, li, .stMarkdown, .stCaption {
            font-size: 1.1rem !important;
            line-height: 1.65;
        }
        .stButton button,
        .stDownloadButton button,
        .stFormSubmitButton button,
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="stBaseButton-primary"] {
            min-height: 56px !important;
            font-size: 1.1rem !important;
            border-radius: 12px !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.9rem !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 1rem !important;
        }
        [data-testid="stChatMessage"] p {
            font-size: 1.12rem !important;
        }
        [data-testid="stSidebar"] {
            font-size: 1.12rem !important;
        }
        [data-testid="stRadio"] label {
            font-size: 1.15rem !important;
            padding: 0.35rem 0.8rem;
            border-radius: 10px;
        }
        .user-card-name {
            font-size: 1.25rem;
        }
        .factor-label, .factor-value {
            font-size: 1.1rem;
        }
        .risk-badge {
            font-size: 1.12rem;
            padding: 0.5rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def speak_text(text: str) -> None:
    payload = json.dumps(str(text), ensure_ascii=False)
    components.html(
        f"""
        <script>
        const utterance = new SpeechSynthesisUtterance({payload});
        utterance.lang = "zh-CN";
        utterance.rate = 0.92;
        utterance.pitch = 1.0;
        speechSynthesis.cancel();
        speechSynthesis.speak(utterance);
        </script>
        """,
        height=0,
        width=0,
    )


def navigate_to(tab_label: str) -> None:
    st.session_state.pending_main_nav = tab_label
    st.session_state.admin_console_page = "工作台"
    st.rerun()


def apply_tab_navigation() -> None:
    target = st.session_state.pop("pending_main_nav", None)
    if not target:
        target = st.session_state.pop("navigate_to", None)
    if not target:
        return
    st.session_state.main_nav = str(target)


def lock_chinese_language() -> None:
    components.html(
        """
        <script>
        var root = window.parent && window.parent.document
          ? window.parent.document
          : document;
        root.documentElement.setAttribute("lang", "zh-CN");
        root.body.setAttribute("translate", "no");
        </script>
        """,
        height=0,
        width=0,
    )


def render_header(
    show_admin_menu: bool = False,
    admin_console_page: str = "工作台",
) -> None:
    status = st.session_state.get("_system_status") or backend.status()
    st.session_state._system_status = status
    logo_path = PROJECT_ROOT / "webapp" / "assets" / "logo-mark.svg"
    logo_mark = ""
    if logo_path.exists():
        logo_data = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        logo_mark = f'<img src="data:image/svg+xml;base64,{logo_data}" alt="">'
    rows = [
        ("CHARLS", status.get("charls_ready", False), "已就绪", "待生成"),
        ("表格基线", status.get("tabular_report", False), "已训练", "未训练"),
        ("MPDD", status.get("mpdd_ready", False), "已就绪", "待构建"),
        ("CMDC", status.get("cmdc_ready", False), "已就绪", "待构建"),
        ("后端 API", status.get("connected", False), "已连接", "本地直连"),
    ]
    pills = "".join(
        f'<span class="status-pill {("status-ok" if ok else "status-wait")}">'
        f"{label} · {ok_text if ok else wait_text}</span>"
        for label, ok, ok_text, wait_text in rows
    )
    with st.container(key="app_header"):
        title_col, controls_col = st.columns(
            [6.3, 3.7],
            vertical_alignment="center",
        )
        with title_col:
            st.markdown(
                f"""
                <div class="app-header-title">
                    <div class="app-mark">{logo_mark}</div>
                    <div>
                        <div class="app-name">老年抑郁风险筛查工作台</div>
                        <div class="app-sub">对话 · 量表 · 多模态 · 个体追踪</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with controls_col:
            if show_admin_menu:
                menu_col, status_col = st.columns(
                    [0.55, 5.45],
                    vertical_alignment="center",
                )
                with menu_col:
                    with st.popover("☰", help="管理菜单"):
                        st.caption("管理菜单")
                        menu_items = [
                            ("工作台", "工作台"),
                            ("账户管理", "账号管理"),
                            ("个人主页", "个人主页"),
                            ("系统说明", "系统说明"),
                        ]
                        for label, page_name in menu_items:
                            if st.button(
                                label,
                                key=f"admin_menu_{page_name}",
                                width="stretch",
                                type=(
                                    "primary"
                                    if admin_console_page == page_name
                                    else "secondary"
                                ),
                            ):
                                st.session_state.admin_console_page = page_name
                                st.rerun()
            else:
                status_col = controls_col
            with status_col:
                st.markdown(
                    f'<div class="app-header-status">{pills}</div>',
                    unsafe_allow_html=True,
                )


INTERVIEW_GREETING = "您好，我先陪您慢慢聊一会儿。不用着急，想到哪说到哪。\n\n最近这一周，您晚上睡得好吗？会不会半夜醒、醒得早，或者白天总没精神？"
COMPANION_GREETING = "您好呀，今天过得怎么样？咱们可以随便聊聊，说说天气、家里的事、吃的饭，或者心里想说的话。"

SYNC_CACHE_KEYS = [
    "tabular_report",
    "imbalance_report",
    "fusion_report",
    "temporal_report",
    "longitudinal_report",
    "multimodal_report",
    "multimodal_subject_report",
    "mpdd_advanced_reports",
    "mpdd_df",
    "mpdd_inference_result",
    "cmdc_report",
    "cmdc_summary",
    "user_list",
    "_system_status",
]


def prime_backend_version() -> None:
    try:
        info = backend.backend_version()
    except Exception:
        return
    version = str(info.get("backend_version") or "")
    data_sig = str(info.get("data_signature") or "")
    if version or data_sig:
        st.session_state._backend_version_seen = version
        st.session_state._backend_data_seen = data_sig
        st.session_state._backend_started_at = info.get("backend_started_at", "")
        st.session_state._last_sync_at = datetime.now().strftime("%H:%M:%S")


@st.fragment(run_every=10)
def auto_sync_backend() -> None:
    try:
        info = backend.backend_version()
    except Exception:
        return
    version = str(info.get("backend_version") or "")
    data_sig = str(info.get("data_signature") or "")
    current = (version, data_sig)
    seen = (
        str(st.session_state.get("_backend_version_seen") or ""),
        str(st.session_state.get("_backend_data_seen") or ""),
    )
    if (not version and not data_sig) or current == seen:
        return
    st.session_state._backend_version_seen = version
    st.session_state._backend_data_seen = data_sig
    st.session_state._backend_started_at = info.get("backend_started_at", "")
    st.session_state._last_sync_at = datetime.now().strftime("%H:%M:%S")
    st.cache_data.clear()
    for key in SYNC_CACHE_KEYS:
        st.session_state.pop(key, None)
    st.toast("检测到后端或数据更新，已自动同步")
    st.rerun()


def init_state() -> None:
    account = st.session_state.get("auth_account") or {}
    is_admin = account.get("role") == "admin"
    default_messages = [{"role": "assistant", "text": INTERVIEW_GREETING}]
    if "messages" not in st.session_state:
        st.session_state.messages = default_messages
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"webapp-{uuid.uuid4().hex}"
    if "agent_cues" not in st.session_state:
        st.session_state.agent_cues = []
    if "safety_alert" not in st.session_state:
        st.session_state.safety_alert = False
    if "agent_engine" not in st.session_state:
        st.session_state.agent_engine = "local"
    if "agent_llm_error" not in st.session_state:
        st.session_state.agent_llm_error = None
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = {}
    if "agent_memory_summary" not in st.session_state:
        st.session_state.agent_memory_summary = ""
    if "agent_latency_ms" not in st.session_state:
        st.session_state.agent_latency_ms = 0
    if "agent_reason" not in st.session_state:
        st.session_state.agent_reason = ""
    if "agent_clinical_alert" not in st.session_state:
        st.session_state.agent_clinical_alert = None
    if "agent_scale_evidence" not in st.session_state:
        st.session_state.agent_scale_evidence = []
    if "agent_covered_topics" not in st.session_state:
        st.session_state.agent_covered_topics = []
    if "agent_focus" not in st.session_state:
        st.session_state.agent_focus = ""
    if "agent_positive_context" not in st.session_state:
        st.session_state.agent_positive_context = ""
    if "agent_protective_factors" not in st.session_state:
        st.session_state.agent_protective_factors = []
    if "agent_alliance_repair_count" not in st.session_state:
        st.session_state.agent_alliance_repair_count = 0
    if "agent_accepted_actions" not in st.session_state:
        st.session_state.agent_accepted_actions = []
    if "agent_guidance_titles" not in st.session_state:
        st.session_state.agent_guidance_titles = []
    if "agent_safety_plan_status" not in st.session_state:
        st.session_state.agent_safety_plan_status = ""
    if "agent_clinical_findings" not in st.session_state:
        st.session_state.agent_clinical_findings = {}
    if "agent_clinical_summary" not in st.session_state:
        st.session_state.agent_clinical_summary = ""
    if "agent_support_preference" not in st.session_state:
        st.session_state.agent_support_preference = ""
    if "agent_general_medical_title" not in st.session_state:
        st.session_state.agent_general_medical_title = ""
    if "agent_medical_sources" not in st.session_state:
        st.session_state.agent_medical_sources = []
    if "conversation_mode" not in st.session_state:
        st.session_state.conversation_mode = "interview"
    if "conversation_mode_selector" not in st.session_state:
        st.session_state.conversation_mode_selector = st.session_state.conversation_mode
    if "companion_messages" not in st.session_state:
        st.session_state.companion_messages = [{"role": "assistant", "text": COMPANION_GREETING}]
    if "companion_session_id" not in st.session_state:
        st.session_state.companion_session_id = f"companion-{uuid.uuid4().hex}"
    if "companion_cues" not in st.session_state:
        st.session_state.companion_cues = []
    if "companion_safety_alert" not in st.session_state:
        st.session_state.companion_safety_alert = False
    if "companion_engine" not in st.session_state:
        st.session_state.companion_engine = "local"
    if "companion_llm_error" not in st.session_state:
        st.session_state.companion_llm_error = None
    if "companion_memory" not in st.session_state:
        st.session_state.companion_memory = {}
    if "companion_memory_summary" not in st.session_state:
        st.session_state.companion_memory_summary = ""
    if "companion_latency_ms" not in st.session_state:
        st.session_state.companion_latency_ms = 0
    if "companion_reason" not in st.session_state:
        st.session_state.companion_reason = ""
    if "companion_clinical_alert" not in st.session_state:
        st.session_state.companion_clinical_alert = None
    if "companion_scale_evidence" not in st.session_state:
        st.session_state.companion_scale_evidence = []
    if "companion_focus" not in st.session_state:
        st.session_state.companion_focus = ""
    if "companion_positive_context" not in st.session_state:
        st.session_state.companion_positive_context = ""
    if "companion_protective_factors" not in st.session_state:
        st.session_state.companion_protective_factors = []
    if "companion_alliance_repair_count" not in st.session_state:
        st.session_state.companion_alliance_repair_count = 0
    if "companion_accepted_actions" not in st.session_state:
        st.session_state.companion_accepted_actions = []
    if "companion_guidance_titles" not in st.session_state:
        st.session_state.companion_guidance_titles = []
    if "companion_safety_plan_status" not in st.session_state:
        st.session_state.companion_safety_plan_status = ""
    if "companion_clinical_findings" not in st.session_state:
        st.session_state.companion_clinical_findings = {}
    if "companion_clinical_summary" not in st.session_state:
        st.session_state.companion_clinical_summary = ""
    if "companion_support_preference" not in st.session_state:
        st.session_state.companion_support_preference = ""
    if "companion_general_medical_title" not in st.session_state:
        st.session_state.companion_general_medical_title = ""
    if "companion_medical_sources" not in st.session_state:
        st.session_state.companion_medical_sources = []
    if is_admin:
        if "tabular_report" not in st.session_state:
            st.session_state.tabular_report = None
        if "imbalance_report" not in st.session_state:
            st.session_state.imbalance_report = None
        if "fusion_report" not in st.session_state:
            st.session_state.fusion_report = None
        if "temporal_report" not in st.session_state:
            st.session_state.temporal_report = None
        if "longitudinal_report" not in st.session_state:
            st.session_state.longitudinal_report = None
        if "multimodal_report" not in st.session_state:
            st.session_state.multimodal_report = None
        if "multimodal_subject_report" not in st.session_state:
            st.session_state.multimodal_subject_report = None
        if "mpdd_advanced_reports" not in st.session_state:
            st.session_state.mpdd_advanced_reports = None
        if "mpdd_inference_result" not in st.session_state:
            st.session_state.mpdd_inference_result = None
        if "mpdd_df" not in st.session_state:
            st.session_state.mpdd_df = None
        if "cmdc_report" not in st.session_state:
            st.session_state.cmdc_report = None
        if "cmdc_summary" not in st.session_state:
            st.session_state.cmdc_summary = None
    if "active_user_id" not in st.session_state:
        st.session_state.active_user_id = None
    if "selected_user_id" not in st.session_state:
        st.session_state.selected_user_id = None
    if "user_list" not in st.session_state:
        st.session_state.user_list = backend.list_users()
    if "age" not in st.session_state:
        st.session_state["age"] = 74
    if "gender" not in st.session_state:
        st.session_state["gender"] = "女"
    if "marital" not in st.session_state:
        st.session_state["marital"] = "有伴侣"
    if "comorbidity" not in st.session_state:
        st.session_state["comorbidity"] = 2
    if "sleep_quality" not in st.session_state:
        st.session_state["sleep_quality"] = 2
    if "self_health" not in st.session_state:
        st.session_state["self_health"] = 3
    if "care_mode" not in st.session_state:
        st.session_state.care_mode = False
    if "cloud_consent" not in st.session_state:
        st.session_state.cloud_consent = False
    if "last_saved_signature" not in st.session_state:
        st.session_state.last_saved_signature = None


HIGH_RISK_DEMO = [
    ("user", "睡不着，经常到半夜醒，早上起来也没精神。"),
    ("user", "以前喜欢下棋，现在不想动，觉得什么都没意思。"),
    ("user", "腰疼、没力气，不想出门，总觉得自己拖累家里人。"),
    ("user", "心里难受，有时候觉得活着没盼头，不过没有别的想法。"),
]

SCALE_LABELS = {
    "cesd10": "流调中心抑郁量表（CES-D-10）",
    "phq9": "患者健康问卷（PHQ-9）",
    "gds15": "老年抑郁量表（GDS-15）",
    "hamd24": "汉密尔顿量表（HAMD-24）",
}

SCALE_ITEMS = {
    "cesd10": [
        "我因一些小事而烦恼",
        "我做事时很难集中注意力",
        "我感到情绪低落",
        "我觉得做任何事都很费劲",
        "我对未来抱有希望",
        "我感到害怕",
        "我睡眠不安稳",
        "我觉得很开心",
        "我感到孤独",
        "我觉得自己无法振作起来",
    ],
    "phq9": [
        "做事时提不起劲或没有兴趣",
        "感到心情低落、沮丧或绝望",
        "入睡困难、睡不安稳或睡眠过多",
        "感觉疲倦或没有活力",
        "食欲不振或吃太多",
        "觉得自己很糟，或觉得自己很失败，或让自己、家人失望",
        "对事物专注有困难，例如看报纸或看电视时",
        "动作或说话速度缓慢到别人已经察觉，或变得比平日更烦躁或坐立不安",
        "有不如死掉或用某种方式伤害自己的念头",
    ],
    "gds15": [
        "你对生活基本上满意吗？",
        "你是否放弃了许多活动和兴趣？",
        "你是否觉得生活空虚？",
        "你是否常感到厌倦？",
        "你对未来是否抱有希望？",
        "你是否因为脑子里想个不停而无法入睡？",
        "你是否大部分时间都感到愉快？",
        "你是否经常感到无助？",
        "你是否宁愿待在家里，而不愿外出做事？",
        "你是否觉得自己的记忆力比大多数人差？",
        "你是否认为现在活着真好？",
        "你是否觉得生活毫无价值？",
        "你是否觉得精力充沛？",
        "你是否觉得自己的处境没有希望？",
        "你是否觉得大多数人比你富有？",
    ],
    "hamd24": [
        "抑郁情绪",
        "罪恶感",
        "自杀",
        "入睡困难",
        "睡眠不深",
        "早醒",
        "工作和兴趣",
        "阻滞",
        "激越",
        "精神性焦虑",
        "躯体性焦虑",
        "胃肠道症状",
        "全身症状",
        "性症状",
        "疑病",
        "体重减轻",
        "自知力",
        "日夜变化",
        "人格解体或现实解体",
        "偏执症状",
        "强迫症状",
        "能力减退感",
        "绝望感",
        "自卑感",
    ],
}

SCALE_LEVEL_LABELS = {
    "normal": "正常",
    "depression_risk": "抑郁风险",
    "mild": "轻度",
    "moderate": "中度",
    "moderately_severe": "中重度",
    "severe": "重度",
}


def reset_conversation() -> None:
    backend.agent_reset(st.session_state.get("session_id"), "interview")
    st.session_state.messages = [
        {
            "role": "assistant",
            "text": INTERVIEW_GREETING,
        }
    ]
    st.session_state.session_id = f"webapp-{uuid.uuid4().hex}"
    st.session_state.agent_cues = []
    st.session_state.safety_alert = False
    st.session_state.agent_engine = "local"
    st.session_state.agent_llm_error = None
    st.session_state.agent_memory = {}
    st.session_state.agent_covered_topics = []
    st.session_state.agent_memory_summary = ""
    st.session_state.agent_latency_ms = 0
    st.session_state.agent_reason = ""
    st.session_state.agent_clinical_alert = None
    st.session_state.agent_scale_evidence = []
    st.session_state.agent_focus = ""
    st.session_state.agent_positive_context = ""
    st.session_state.agent_protective_factors = []
    st.session_state.agent_alliance_repair_count = 0
    st.session_state.agent_accepted_actions = []
    st.session_state.agent_guidance_titles = []
    st.session_state.agent_safety_plan_status = ""
    st.session_state.agent_clinical_findings = {}
    st.session_state.agent_clinical_summary = ""
    st.session_state.agent_support_preference = ""
    st.session_state.agent_general_medical_title = ""
    st.session_state.agent_medical_sources = []


def reset_companion_conversation() -> None:
    backend.agent_reset(st.session_state.get("companion_session_id"), "companion")
    st.session_state.companion_messages = [{"role": "assistant", "text": COMPANION_GREETING}]
    st.session_state.companion_session_id = f"companion-{uuid.uuid4().hex}"
    st.session_state.companion_cues = []
    st.session_state.companion_safety_alert = False
    st.session_state.companion_engine = "local"
    st.session_state.companion_llm_error = None
    st.session_state.companion_memory = {}
    st.session_state.companion_memory_summary = ""
    st.session_state.companion_latency_ms = 0
    st.session_state.companion_reason = ""
    st.session_state.companion_clinical_alert = None
    st.session_state.companion_scale_evidence = []
    st.session_state.companion_focus = ""
    st.session_state.companion_positive_context = ""
    st.session_state.companion_protective_factors = []
    st.session_state.companion_alliance_repair_count = 0
    st.session_state.companion_accepted_actions = []
    st.session_state.companion_guidance_titles = []
    st.session_state.companion_safety_plan_status = ""
    st.session_state.companion_clinical_findings = {}
    st.session_state.companion_clinical_summary = ""
    st.session_state.companion_support_preference = ""
    st.session_state.companion_general_medical_title = ""
    st.session_state.companion_medical_sources = []


def sync_conversation_mode() -> None:
    selected = st.session_state.get("conversation_mode_selector")
    if selected in {"interview", "companion"}:
        st.session_state.conversation_mode = selected


def profile_to_state(user: dict) -> None:
    st.session_state["age"] = int(user.get("age") or 74)
    st.session_state["gender"] = "女" if user.get("gender") != "male" else "男"
    st.session_state["marital"] = (
        "有伴侣" if user.get("marital_status") != "single" else "独居 / 丧偶 / 离异"
    )
    st.session_state["comorbidity"] = int(user.get("comorbidity_count") or 2)
    st.session_state["self_health"] = int(user.get("self_rated_health") or 3)


def fill_high_risk_demo() -> None:
    messages = [
        {
            "role": "assistant",
            "text": INTERVIEW_GREETING,
        }
    ]
    for role, text in HIGH_RISK_DEMO:
        messages.append({"role": role, "text": text})
    st.session_state.messages = messages
    st.session_state.agent_cues = []
    st.session_state.safety_alert = False


def inject_auth_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6faf9;
        }
        .auth-hero {
            padding: 1.6rem 1.7rem 1.35rem;
            border: 1px solid #d5e6e2;
            border-radius: 20px;
            background: linear-gradient(145deg, #ffffff 0%, #eefaf6 58%, #eaf1fb 100%);
            box-shadow: 0 18px 42px rgba(30, 74, 80, .10);
            margin-bottom: 1rem;
        }
        .auth-hero h1 {
            margin: 0 0 .35rem 0;
            color: #18343b;
            font-size: 1.78rem;
            letter-spacing: 0;
        }
        .auth-hero p {
            margin: 0;
            color: #5c7379;
            line-height: 1.65;
        }
        .auth-rule {
            margin: .75rem 0 0;
            color: #38575f;
            font-size: .86rem;
        }
        div[data-testid="stForm"] {
            border: 1px solid #dbe8e5;
            border-radius: 16px;
            background: rgba(255, 255, 255, .92);
            padding: 1.05rem 1.1rem .9rem;
            box-shadow: 0 12px 30px rgba(34, 72, 78, .08);
        }
        div[data-baseweb="tab-list"] {
            justify-content: center;
            gap: 1.1rem;
        }
        div[data-baseweb="tab"] {
            font-weight: 650;
        }
        div[data-testid="stTextInput"] input {
            min-height: 2.7rem;
            border-radius: 10px;
        }
        div[data-testid="stFormSubmitButton"] button {
            min-height: 2.65rem;
            border-radius: 10px;
            font-weight: 650;
        }
        .auth-account {
            padding: .55rem .7rem;
            border-radius: 10px;
            background: #eef6f4;
            border: 1px solid #d7e7e3;
            color: #24474f;
            font-size: .88rem;
            margin: .35rem 0 .7rem;
        }
        .role-pill {
            display: inline-block;
            padding: .1rem .45rem;
            border-radius: 999px;
            background: #d9efe9;
            color: #17635e;
            font-size: .78rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _auth_challenge(key: str) -> dict | None:
    challenge = st.session_state.get(key)
    if not isinstance(challenge, dict) or not challenge.get("challenge_id"):
        challenge = backend.auth_challenge()
        if challenge:
            st.session_state[key] = challenge
    return challenge if isinstance(challenge, dict) else None


def _clear_auth_state() -> None:
    backend.set_auth_token(None)
    for key in list(st.session_state.keys()):
        st.session_state.pop(key, None)


def _finish_login(result: dict) -> None:
    backend.set_auth_token(result.get("token"))
    account = result.get("account") or {}
    st.session_state.auth_token = result.get("token")
    st.session_state.auth_account = account
    st.session_state.login_welcome = (
        account.get("display_name") or account.get("username") or ""
    )
    for key in ["login_challenge", "register_challenge", "auth_flash"]:
        st.session_state.pop(key, None)


def ensure_auth_session() -> bool:
    token = st.session_state.get("auth_token")
    if not token:
        return False
    backend.set_auth_token(str(token))
    account = backend.auth_me()
    if account:
        st.session_state.auth_account = account
        return True
    _clear_auth_state()
    return False


def render_auth_gate() -> None:
    inject_auth_css()
    _, center, _ = st.columns([0.6, 1.55, 0.6])
    with center:
        logo_png = PROJECT_ROOT / "webapp" / "assets" / "previews" / "logo-zh.png"
        if logo_png.exists():
            logo_left, logo_mid, logo_right = st.columns([1, 1.4, 1])
            with logo_mid:
                st.image(str(logo_png), width=240)
        st.markdown(
            """
            <div class="auth-hero">
                <h1>老年抑郁风险筛查工作台</h1>
                <p>请先登录。管理员可使用完整工作台和账号管理；普通用户只进入评估与个人记录功能。</p>
                <div class="auth-rule">用户名允许 @ _ - &amp; ¥ $ # 和数字、大小写字母；密码至少 8 位，并同时包含大写和小写字母。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "本系统仅用于科研风险筛查和辅助沟通，不替代医生诊断。"
            "对话默认在本地处理；启用第三方大模型前需要单独同意，并自动隐藏姓名、住址、电话、证件号和邮箱。"
        )
        flash = st.session_state.pop("auth_flash", None)
        if flash:
            st.error(flash)
        login_tab, register_tab, recovery_tab = st.tabs(["登录", "注册", "找回密码"])

        with login_tab:
            with st.form("auth_login_form"):
                username = st.text_input("用户名", key="login_username")
                password = st.text_input("密码", type="password", key="login_password")
                challenge = _auth_challenge("login_challenge")
                if challenge:
                    st.info(f"人机验证：{challenge['question']}")
                else:
                    st.error("无法连接后端获取人机验证，请确认桌面版后端已启动。")
                answer = st.text_input("请输入计算结果", key="login_answer")
                submitted = st.form_submit_button("登录", width="stretch")
            if submitted:
                if challenge is None:
                    st.session_state.auth_flash = "无法获取人机验证，请稍后重试"
                else:
                    result = backend.auth_login(
                        username.strip(),
                        password,
                        challenge["challenge_id"],
                        answer.strip(),
                    )
                    if result.get("ok"):
                        _finish_login(result)
                        st.rerun()
                    st.session_state.auth_flash = result.get("error") or "登录失败"
                    st.session_state.pop("login_challenge", None)
                    st.rerun()

        with register_tab:
            question_options = backend.auth_security_options() or [
                "您最喜欢的颜色是？",
                "您小时候居住的城市是？",
                "您最喜欢的一道菜是？",
                "您第一只宠物的名字是？",
            ]
            with st.form("auth_register_form"):
                reg_username = st.text_input("用户名", key="register_username")
                display_name = st.text_input("显示名称（可选）", key="register_display_name")
                reg_password = st.text_input(
                    "密码", type="password", key="register_password"
                )
                confirm_password = st.text_input(
                    "确认密码", type="password", key="register_confirm_password"
                )
                reg_challenge = _auth_challenge("register_challenge")
                if reg_challenge:
                    st.info(f"人机验证：{reg_challenge['question']}")
                else:
                    st.error("无法连接后端获取人机验证，请确认桌面版后端已启动。")
                reg_answer = st.text_input("请输入计算结果", key="register_answer")
                st.markdown("**密保问题**")
                security_questions = []
                security_answers = []
                for index in range(3):
                    q_col, a_col = st.columns([1.25, 1])
                    with q_col:
                        question = st.selectbox(
                            f"问题 {index + 1}",
                            question_options,
                            index=min(index, len(question_options) - 1),
                            key=f"register_security_question_{index}",
                        )
                    with a_col:
                        answer = st.text_input(
                            f"答案 {index + 1}",
                            key=f"register_security_answer_{index}",
                        )
                    security_questions.append(question)
                    security_answers.append(answer)
                register_submitted = st.form_submit_button("注册并登录", width="stretch")
            if register_submitted:
                if reg_challenge is None:
                    st.session_state.auth_flash = "无法获取人机验证，请稍后重试"
                elif reg_password != confirm_password:
                    st.session_state.auth_flash = "两次输入的密码不一致"
                elif len(set(security_questions)) != 3:
                    st.session_state.auth_flash = "三个密保问题不能重复"
                else:
                    result = backend.auth_register(
                        reg_username.strip(),
                        reg_password,
                        display_name.strip(),
                        reg_challenge["challenge_id"],
                        reg_answer.strip(),
                        security_questions,
                        security_answers,
                    )
                    if result.get("ok"):
                        _finish_login(result)
                        st.rerun()
                    st.session_state.auth_flash = result.get("error") or "注册失败"
                st.session_state.pop("register_challenge", None)
                st.rerun()

        with recovery_tab:
            st.caption("先填写完整用户名并获取密保问题，再回答全部问题并设置新密码。")
            recovery_username = st.text_input(
                "用户名", key="recovery_username_input"
            )
            if st.button("获取密保问题", width="stretch", key="recovery_questions_button"):
                result = backend.auth_recovery_questions(recovery_username.strip())
                if result.get("ok"):
                    st.session_state.recovery_username = recovery_username.strip()
                    st.session_state.recovery_questions = result.get("questions") or []
                    st.session_state.auth_flash = "密保问题已加载，请完整作答"
                else:
                    st.session_state.auth_flash = result.get("error") or "无法获取密保问题"
                    st.session_state.pop("recovery_questions", None)
                st.rerun()

            recovery_questions = st.session_state.get("recovery_questions") or []
            if recovery_questions:
                with st.form("auth_recovery_form"):
                    recovery_answers = []
                    for index, question in enumerate(recovery_questions):
                        st.markdown(f"**{index + 1}. {question}**")
                        recovery_answers.append(
                            st.text_input(
                                f"答案 {index + 1}",
                                key=f"recovery_answer_{index}",
                            )
                        )
                    new_password = st.text_input(
                        "新密码", type="password", key="recovery_new_password"
                    )
                    confirm_new_password = st.text_input(
                        "确认新密码",
                        type="password",
                        key="recovery_confirm_password",
                    )
                    reset_submitted = st.form_submit_button(
                        "重置密码", width="stretch"
                    )
                if reset_submitted:
                    if new_password != confirm_new_password:
                        st.session_state.auth_flash = "两次输入的新密码不一致"
                    else:
                        result = backend.auth_recovery_reset(
                            st.session_state.get("recovery_username", ""),
                            recovery_answers,
                            new_password,
                        )
                        if result.get("ok"):
                            st.session_state.auth_flash = "密码已重置，请返回登录"
                            st.session_state.pop("recovery_questions", None)
                            st.session_state.pop("recovery_username", None)
                        else:
                            st.session_state.auth_flash = result.get("error") or "密码重置失败"
                    st.rerun()


def risk_gauge(probability: float) -> go.Figure:
    value = round(probability * 100, 1)
    color = "#16a34a" if value < 40 else "#f59e0b" if value < 70 else "#dc2626"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 34, "color": color}},
            title={"text": "抑郁风险概率", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.24},
                "steps": [
                    {"range": [0, 40], "color": "#e7f6ec"},
                    {"range": [40, 70], "color": "#fdf0d8"},
                    {"range": [70, 100], "color": "#fde5e5"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": value},
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def assessment_report_markdown(
    user: dict | None,
    demographics: dict,
    result: dict,
    scale_name: str,
    transcript: list[dict],
    safety_alert: bool,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    level_cn = {"low": "低风险", "medium": "中风险", "high": "高风险"}.get(
        result.get("risk_level"), "未知"
    )
    scale_level_cn = SCALE_LEVEL_LABELS.get(
        result.get("scale_level"), result.get("scale_level", "")
    )
    lines = [
        "# 老年抑郁风险筛查报告",
        "",
        f"生成时间：{now}",
        "",
        "## 用户信息",
        "",
    ]
    if user:
        gender = "女" if user.get("gender") == "female" else "男"
        lines += [
            f"- 姓名 / 编号：{user.get('name', '-')}",
            f"- 年龄：{user.get('age', '-')} 岁",
            f"- 性别：{gender}",
            f"- 慢性病数量：{user.get('comorbidity_count', '-')} 项",
        ]
    else:
        lines.append("- 未选择用户档案")
    lines += [
        "",
        "## 本次评估输入",
        "",
        f"- 量表：{SCALE_LABELS.get(scale_name, scale_name)}",
        f"- 年龄：{demographics.get('age', '-')} 岁",
        f"- 性别：{'女' if demographics.get('gender') == 'female' else '男'}",
        f"- 婚姻状况：{'有伴侣' if demographics.get('married') else '独居 / 丧偶 / 离异'}",
        f"- 慢性病数量：{demographics.get('comorbidity_count', '-')} 项",
        f"- 睡眠质量：{demographics.get('sleep_quality', '-')}（1 很好 → 4 较差）",
        f"- 自评健康：{demographics.get('self_rated_health', '-')}（1 很好 → 5 很差）",
        "",
        "## 评估结果",
        "",
        f"- 抑郁风险概率：**{float(result.get('depression_probability') or 0):.0%}**",
        f"- 风险等级：**{level_cn}**",
        f"- 量表分：{result.get('scale_score', '-')}（等级：{scale_level_cn}）",
        "",
        "### 风险构成",
        "",
        "| 信号 | 风险 |",
        "| --- | ---: |",
        f"| 语言线索 | {float(result.get('text_risk', 0)):.0%} |",
        f"| 量表风险 | {float(result.get('scale_risk', 0)):.0%} |",
        f"| 背景 / 共病 | {float(result.get('context_risk', 0)):.0%} |",
    ]
    if result.get("expression_risk") is not None:
        lines.append(f"| 表情 / 行为 | {float(result['expression_risk']):.0%} |")
    if result.get("mpdd_risk") is not None:
        lines.append(f"| MPDD 多模态 | {float(result['mpdd_risk']):.0%} |")
    fusion_cn = "已学习融合权重" if result.get("fusion_mode") == "learned" else "文献先验权重"
    lines += [
        "",
        f"融合方式：{fusion_cn}",
        "",
    ]
    language_cues = result.get("language_cues") or []
    if language_cues:
        cue_text = "、".join(
            f"{c['cue']}（{c['confidence']:.0%}）" if isinstance(c, dict) else str(c)
            for c in language_cues
        )
        lines += ["### 语言线索", "", cue_text, ""]
    if result.get("llm_summary"):
        lines += ["### 大模型摘要", "", str(result["llm_summary"]), ""]
    if transcript:
        lines += ["### 对话记录", ""]
        for msg in transcript:
            role = "用户" if msg.get("role") == "user" else "系统"
            lines.append(f"> {role}：{msg.get('text', '')}")
        lines.append("")
    if safety_alert or result.get("risk_level") == "high":
        lines += [
            "### 安全提示",
            "",
            "本次筛查出现高风险或安全相关表达，请优先联系家人或医生。",
            "全国心理援助热线：12356；紧急情况请拨打 120 / 110。",
            "",
        ]
    lines += [
        "---",
        "",
        "本报告仅用于科研筛查与风险提示，不替代临床诊断。",
        "",
    ]
    return "\n".join(lines)


def conversation_export_markdown(
    messages: list[dict],
    mode: str,
    cues: list,
    safety_alert: bool,
    memory: dict | None = None,
    guidance_titles: list | None = None,
    accepted_actions: list | None = None,
    safety_plan_status: str = "",
    clinical_summary: str = "",
    support_preference: str = "",
    general_medical_title: str = "",
    medical_sources: list | None = None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode_cn = "温情问询" if mode == "interview" else "日常陪伴式聊天"
    lines = [
        "# 对话记录",
        "",
        f"模式：{mode_cn}",
        f"导出时间：{now}",
        "",
    ]
    for msg in messages:
        role = "用户" if msg.get("role") == "user" else "系统"
        lines.append(f"> {role}：{msg.get('text', '')}")
    if cues:
        lines += ["", "## 风险线索", ""]
        lines.append("、".join(str(c) for c in cues))
    if memory:
        lines += ["", "## 对话记忆", ""]
        for key, value in memory.items():
            lines.append(f"- {key}：{value}")
    if guidance_titles:
        lines += ["", "## 本轮支持策略", ""]
        lines.append("、".join(str(item) for item in guidance_titles))
    if accepted_actions:
        lines += ["", "## 已确认行动", ""]
        for action in accepted_actions:
            lines.append(f"- {action}")
    if clinical_summary:
        lines += ["", "## 临床访谈摘要", "", clinical_summary]
    if support_preference:
        preference_label = {
            "listen": "先倾听",
            "action": "一起想办法",
            "both": "倾听并一起想办法",
        }.get(support_preference, support_preference)
        lines += ["", "## 支持偏好", "", preference_label]
    if general_medical_title:
        lines += ["", "## 医学知识线索", "", general_medical_title]
    if medical_sources:
        lines += ["", "## MedlinePlus 参考", ""]
        for source in medical_sources:
            title = source.get("title_zh") or source.get("title") or "医学科普"
            url = source.get("url") or ""
            lines.append(f"- [{title}]({url})" if url else f"- {title}")
    if safety_alert:
        lines += [
            "",
            "## 安全事件",
            "",
            "对话中出现需要关注的安全表达。请优先联系家人或医生；",
            "全国心理援助热线 12356，紧急情况请拨打 120 / 110。",
        ]
    safety_labels = {
        "checking": "正在确认当前安全与可联系的人",
        "support_confirmed": "已确认有人陪伴，建议今天继续联系专业人员",
        "emergency_referral": "已进入紧急求助提示，请立即联系真人帮助",
    }
    if safety_plan_status:
        lines += [
            "",
            "## 安全计划",
            "",
            safety_labels.get(safety_plan_status, safety_plan_status),
        ]
    lines += ["", "---", "", "本记录仅用于科研筛查与风险提示，不替代临床诊断。"]
    return "\n".join(lines)


def persist_assessment(active_user: dict, record: dict, signature: str) -> bool:
    if not st.session_state.get("assessment_storage_consent", False):
        return False
    if signature == st.session_state.get("last_saved_signature"):
        return False
    backend.save_assessment(active_user["id"], record)
    st.session_state.last_saved_signature = signature
    return True


def status_pill(ok: bool, ok_text: str, wait_text: str) -> str:
    cls = "status-ok" if ok else "status-wait"
    label = ok_text if ok else wait_text
    return f'<span class="status-pill {cls}">{label}</span>'


def dialogue_engine_caption(
    engine: str,
    reason: str,
    latency_ms,
    prompt_version: str = "",
) -> str:
    labels = {
        "llm": "大模型增强",
        "fastpath": "预案快速响应",
        "cache": "缓存复用",
        "local": "本地规则",
    }
    reason_labels = {
        "safety_rule": "安全规则",
        "response_cache": "缓存命中",
        "intent_model": "意图模型",
        "llm": "云端模型",
        "llm_fallback": "云端失败回退",
        "keyword_or_rule": "关键词规则",
        "local_fallback": "本地回退",
        "greeting": "问候预案",
        "slot_followup": "澄清与跟进",
        "clinical_red_flag": "临床安全分流",
        "medical_boundary": "医学边界保护",
        "help_request": "主动求助分流",
    }
    text = f"对话引擎：{labels.get(engine, engine or '本地规则')}"
    if reason and reason in reason_labels:
        text += f" · {reason_labels[reason]}"
    if prompt_version:
        text += f" · Prompt {prompt_version}"
    try:
        latency = float(latency_ms or 0)
        if latency > 0:
            text += f" · {latency:.0f} ms"
    except (TypeError, ValueError):
        pass
    return text


def human_bytes(size: int) -> str:
    value = float(size or 0)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def render_sidebar() -> str:
    account = st.session_state.get("auth_account") or {}
    is_admin = account.get("role") == "admin"
    with st.sidebar:
        logo_path = PROJECT_ROOT / "webapp" / "assets" / "logo.svg"
        logo_png = PROJECT_ROOT / "webapp" / "assets" / "previews" / "logo-zh.png"
        if logo_png.exists():
            st.image(str(logo_png), width=260)
        elif logo_path.exists():
            logo_svg = logo_path.read_text(encoding="utf-8")
            st.markdown(f'<div class="logo-lockup">{logo_svg}</div>', unsafe_allow_html=True)
        else:
            st.title("筛查工作台")
        st.caption("老年抑郁风险多模态评估")
        role_text = "管理员" if is_admin else "普通用户"
        st.markdown(
            f'<div class="auth-account"><b>{account.get("display_name") or account.get("username")}</b>'
            f'<br><span class="role-pill">{role_text}</span> · {account.get("username", "")}</div>',
            unsafe_allow_html=True,
        )
        if st.button("退出登录", width="stretch", key="logout_button"):
            backend.auth_logout()
            _clear_auth_state()
            st.rerun()
        care_mode = st.toggle(
            "关怀模式（大字 / 高对比）",
            value=st.session_state.get("care_mode", False),
            key="care_mode_toggle",
        )
        st.session_state.care_mode = care_mode
        if care_mode:
            st.success("关怀模式已开启：大字 / 高对比")
        else:
            st.caption("当前为标准模式")
        if is_admin:
            st.markdown("---")
            st.markdown("#### 数据状态")
            system_status = st.session_state.get("_system_status") or backend.status()
            st.session_state._system_status = system_status
            rows = [
                ("CHARLS 标准表", system_status.get("charls_ready", False), "已就绪", "待生成"),
                (
                    "表格基线",
                    system_status.get("tabular_report", False),
                    "已训练",
                    "未训练",
                ),
                ("MPDD 特征", system_status.get("mpdd_ready", False), "已就绪", "待构建"),
                ("CMDC 外部验证", system_status.get("cmdc_ready", False), "已就绪", "待构建"),
                ("CMDC 稳健性", system_status.get("cmdc_advanced_ready", False), "已完成", "待分析"),
                ("CMDC 锁定留出", system_status.get("cmdc_holdout_ready", False), "已完成", "待验证"),
                (
                    "对话意图模型",
                    system_status.get("dialogue_intent_model", False),
                    "已加载",
                    "未训练",
                ),
                (
                    "对话响应策略模型",
                    system_status.get("dialogue_response_policy_model", False),
                    "已加载",
                    "未训练",
                ),
                (
                    "伦理安全审计",
                    system_status.get("ethics_audit_passed", False),
                    "已通过",
                    "待检查",
                ),
                (
                    "MedlinePlus 医学知识库",
                    system_status.get("medical_kb_ready", False),
                    f"{system_status.get('medical_kb_topics', 0)} 个主题",
                    "待导入",
                ),
                ("后端 API", system_status.get("connected", False), "已连接", "本地直连"),
            ]
            for label, ok, ok_text, wait_text in rows:
                st.markdown(
                    f'<div class="status-row"><span class="status-label">{label}</span>'
                    f"{status_pill(ok, ok_text, wait_text)}</div>",
                    unsafe_allow_html=True,
                )
            prompt_versions = system_status.get("prompt_versions") or {}
            if prompt_versions:
                st.caption(
                    "Prompt 版本：问询 "
                    f"{prompt_versions.get('interview', '-')} · 陪伴 "
                    f"{prompt_versions.get('companion', '-')}"
                )
            if system_status.get("tabular_models"):
                st.caption(f"已保存表格模型：{system_status['tabular_models']} 个")
            last_sync = st.session_state.get("_last_sync_at")
            sync_text = "自动同步：每 10 秒检查后端更新"
            if last_sync:
                sync_text += f" · 上次 {last_sync}"
            st.caption(sync_text)
            st.markdown("---")
            api_key = st.text_input("大模型 API Key（可选）", type="password")
            if api_key:
                st.caption("已填写，用于对话 Agent 与风险摘要。")
            else:
                st.caption("留空时使用本地 .env 配置。")
            return api_key
        st.markdown("---")
        st.caption("普通用户仅可使用风险评估与个人记录功能。")
        return ""


def store_companion_response(agent_response: dict) -> None:
    st.session_state.companion_cues = agent_response.get("risk_cues", [])
    st.session_state.companion_safety_alert = agent_response.get(
        "safety_alert", False
    )
    st.session_state.companion_engine = agent_response.get("engine", "local")
    st.session_state.companion_llm_error = agent_response.get("llm_error")
    st.session_state.companion_memory = agent_response.get("memory", {})
    st.session_state.companion_memory_summary = agent_response.get(
        "memory_summary", ""
    )
    st.session_state.companion_latency_ms = agent_response.get("latency_ms", 0)
    st.session_state.companion_reason = agent_response.get("fastpath_reason", "")
    st.session_state.companion_prompt_version = agent_response.get(
        "last_prompt_version", ""
    )
    st.session_state.companion_clinical_alert = agent_response.get("clinical_alert")
    st.session_state.companion_scale_evidence = agent_response.get(
        "scale_evidence", []
    )
    st.session_state.companion_focus = agent_response.get("focus", "")
    st.session_state.companion_positive_context = agent_response.get(
        "positive_context", ""
    )
    st.session_state.companion_protective_factors = agent_response.get(
        "protective_factors", []
    )
    st.session_state.companion_alliance_repair_count = agent_response.get(
        "alliance_repair_count", 0
    )
    st.session_state.companion_accepted_actions = agent_response.get(
        "accepted_actions", []
    )
    st.session_state.companion_guidance_titles = agent_response.get(
        "last_guidance_titles", []
    )
    st.session_state.companion_safety_plan_status = agent_response.get(
        "safety_plan_status", ""
    )
    st.session_state.companion_clinical_findings = agent_response.get(
        "clinical_findings", {}
    )
    st.session_state.companion_clinical_summary = agent_response.get(
        "clinical_summary", ""
    )
    st.session_state.companion_support_preference = agent_response.get(
        "support_preference", ""
    )
    st.session_state.companion_general_medical_title = agent_response.get(
        "general_medical_title", ""
    )
    st.session_state.companion_medical_sources = agent_response.get(
        "medical_sources", []
    )


def store_interview_response(agent_response: dict) -> None:
    st.session_state.agent_cues = agent_response.get("risk_cues", [])
    st.session_state.safety_alert = agent_response.get("safety_alert", False)
    st.session_state.agent_engine = agent_response.get("engine", "local")
    st.session_state.agent_llm_error = agent_response.get("llm_error")
    st.session_state.agent_memory = agent_response.get("memory", {})
    st.session_state.agent_memory_summary = agent_response.get(
        "memory_summary", ""
    )
    st.session_state.agent_latency_ms = agent_response.get("latency_ms", 0)
    st.session_state.agent_reason = agent_response.get("fastpath_reason", "")
    st.session_state.agent_prompt_version = agent_response.get(
        "last_prompt_version", ""
    )
    st.session_state.agent_clinical_alert = agent_response.get("clinical_alert")
    st.session_state.agent_scale_evidence = agent_response.get("scale_evidence", [])
    st.session_state.agent_covered_topics = agent_response.get("covered_topics", [])
    st.session_state.agent_focus = agent_response.get("focus", "")
    st.session_state.agent_positive_context = agent_response.get(
        "positive_context", ""
    )
    st.session_state.agent_protective_factors = agent_response.get(
        "protective_factors", []
    )
    st.session_state.agent_alliance_repair_count = agent_response.get(
        "alliance_repair_count", 0
    )
    st.session_state.agent_accepted_actions = agent_response.get(
        "accepted_actions", []
    )
    st.session_state.agent_guidance_titles = agent_response.get(
        "last_guidance_titles", []
    )
    st.session_state.agent_safety_plan_status = agent_response.get(
        "safety_plan_status", ""
    )
    st.session_state.agent_clinical_findings = agent_response.get(
        "clinical_findings", {}
    )
    st.session_state.agent_clinical_summary = agent_response.get(
        "clinical_summary", ""
    )
    st.session_state.agent_support_preference = agent_response.get(
        "support_preference", ""
    )
    st.session_state.agent_general_medical_title = agent_response.get(
        "general_medical_title", ""
    )
    st.session_state.agent_medical_sources = agent_response.get(
        "medical_sources", []
    )


def render_companion_chat(api_key: str) -> None:
    st.markdown("#### 日常陪伴式聊天")
    st.caption("这里不进入风险评分，就是陪您慢慢聊聊天。")

    button_col, hint_col = st.columns([0.28, 0.72])
    with button_col:
        if st.button("清空陪伴对话", width="stretch"):
            reset_companion_conversation()
            st.rerun()
    with hint_col:
        st.caption("切回“温情问询”可继续做风险筛查。")

    chat_box = st.container(border=True, height=420)
    with chat_box:
        for msg in st.session_state.companion_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["text"])

    companion_md = conversation_export_markdown(
        st.session_state.companion_messages,
        "companion",
        st.session_state.companion_cues,
        st.session_state.companion_safety_alert,
        memory=st.session_state.get("companion_memory") or {},
        guidance_titles=st.session_state.get("companion_guidance_titles") or [],
        accepted_actions=st.session_state.get("companion_accepted_actions") or [],
        safety_plan_status=st.session_state.get(
            "companion_safety_plan_status", ""
        ),
        clinical_summary=st.session_state.get("companion_clinical_summary", ""),
        support_preference=st.session_state.get(
            "companion_support_preference", ""
        ),
        general_medical_title=st.session_state.get(
            "companion_general_medical_title", ""
        ),
        medical_sources=st.session_state.get("companion_medical_sources") or [],
    )
    st.download_button(
        "导出对话记录",
        data=companion_md.encode("utf-8"),
        file_name=f"companion_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        key="export_companion_chat",
    )

    user_input = st.chat_input("随便聊聊，说说今天的天气、家常，或者心里想说的话")
    if user_input:
        st.session_state.companion_messages.append({"role": "user", "text": user_input})
        try:
            with st.spinner("正在接上下一句"):
                agent_response = backend.agent_chat(
                    messages=st.session_state.companion_messages,
                    context={
                        "mode": "companion",
                        "api_key": api_key,
                        "cloud_consent": st.session_state.get(
                            "cloud_consent", False
                        ),
                    },
                    session_id=st.session_state.companion_session_id,
                )
            store_companion_response(agent_response)
            reply = agent_response.get("reply") or "我在听，您继续说。"
        except Exception:
            reply = "抱歉，刚才没有成功回复。请您再说一次，我会继续听着。"
            st.session_state.companion_engine = "local"
            st.session_state.companion_llm_error = "请求失败"
        st.session_state.companion_messages.append({"role": "assistant", "text": reply})
        st.rerun()

    st.caption(
        dialogue_engine_caption(
            st.session_state.companion_engine,
            st.session_state.get("companion_reason", ""),
            st.session_state.get("companion_latency_ms", 0),
            st.session_state.get("companion_prompt_version", ""),
        )
    )
    companion_alert = st.session_state.get("companion_clinical_alert")
    if companion_alert:
        st.error(
            f"安全提醒：{companion_alert.get('message', '请尽快联系家人或医生评估。')}"
        )
    with st.expander("本轮对话记忆", expanded=False):
        memory = st.session_state.get("companion_memory") or {}
        if memory:
            for key, value in memory.items():
                st.markdown(f"- **{key}**：{value}")
        else:
            st.caption("还没有记录到明确的个人信息。")
        if st.session_state.get("companion_memory_summary"):
            st.caption(f"早前摘要：{st.session_state.companion_memory_summary}")
        if st.session_state.get("companion_focus"):
            st.caption(f"本轮重点：{st.session_state.companion_focus}")
        if st.session_state.get("companion_positive_context"):
            st.caption(
                f"同时确认：{st.session_state.companion_positive_context}"
            )
        if st.session_state.get("companion_protective_factors"):
            st.caption(
                "保护因素："
                + "、".join(st.session_state.companion_protective_factors)
            )
        if st.session_state.get("companion_alliance_repair_count"):
            st.caption(
                "已根据您的反馈调整对话方式 "
                f"{st.session_state.companion_alliance_repair_count} 次"
            )
        if st.session_state.get("companion_accepted_actions"):
            st.caption(
                "已确认行动："
                + "、".join(st.session_state.companion_accepted_actions)
            )
        if st.session_state.get("companion_guidance_titles"):
            st.caption(
                "本轮支持策略："
                + "、".join(st.session_state.companion_guidance_titles)
            )
        if st.session_state.get("companion_clinical_summary"):
            st.caption(
                "临床访谈摘要："
                + st.session_state.companion_clinical_summary
            )
        if st.session_state.get("companion_support_preference"):
            preference_label = {
                "listen": "先倾听",
                "action": "一起想办法",
                "both": "倾听并一起想办法",
            }.get(st.session_state.companion_support_preference, "")
            if preference_label:
                st.caption(f"支持偏好：{preference_label}")
        if st.session_state.get("companion_general_medical_title"):
            st.caption(
                "医学知识线索："
                + st.session_state.companion_general_medical_title
            )
        for source in st.session_state.get("companion_medical_sources") or []:
            title = source.get("title_zh") or source.get("title") or "医学科普"
            url = source.get("url") or ""
            st.caption(
                "MedlinePlus 参考："
                + (f"[{title}]({url})" if url else title)
            )
        evidence = st.session_state.get("companion_scale_evidence") or []
        if evidence:
            st.markdown("**量表核对提示**")
            for item in evidence:
                st.caption(
                    f"PHQ-9 第 {item['item']} 项（{item['label']}）："
                    f"{'、'.join(item['cues'])}，需量表确认"
                )
    if st.session_state.companion_safety_alert:
        st.error(
            "对话中出现了需要关注的安全表达。请优先联系家人或医生；"
            "全国心理援助热线 12356，紧急情况请拨打 120 / 110。"
        )
        safety_status = {
            "checking": "正在确认当前安全与可联系的人",
            "support_confirmed": "已确认有人陪伴，建议今天继续联系专业人员",
            "emergency_referral": "已进入紧急求助提示，请立即联系真人帮助",
        }.get(st.session_state.get("companion_safety_plan_status"), "")
        if safety_status:
            st.caption(safety_status)


def render_assessment(api_key: str) -> None:
    st.subheader("抑郁风险筛查")
    st.caption("填写背景与量表，再进行一轮简短访谈；每次回答都会实时更新风险概率。")
    cloud_consent = st.toggle(
        "允许将脱敏后的对话文本发送给第三方大模型服务",
        value=st.session_state.get("cloud_consent", False),
        key="cloud_consent_toggle",
        help="默认关闭。开启后仍不会发送姓名、住址、电话、证件号或邮箱。",
    )
    st.session_state.cloud_consent = cloud_consent
    if not cloud_consent:
        st.caption("当前仅使用本地规则与本地模型，不向第三方发送对话。")

    if "conversation_mode_selector" not in st.session_state:
        st.session_state.conversation_mode_selector = st.session_state.get(
            "conversation_mode", "interview"
        )
    mode = st.radio(
        "对话模式",
        options=["interview", "companion"],
        key="conversation_mode_selector",
        on_change=sync_conversation_mode,
        format_func=lambda x: "温情问询" if x == "interview" else "日常陪伴式聊天",
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode in {"interview", "companion"}:
        st.session_state.conversation_mode = mode
    mode = st.session_state.get("conversation_mode", "interview")
    st.caption(
        "当前模式：温情问询 · 进入抑郁风险筛查"
        if mode == "interview"
        else "当前模式：日常陪伴式聊天 · 不进入风险评分"
    )
    if mode == "companion":
        render_companion_chat(api_key)
        return

    storage_consent = st.toggle(
        "我已知晓并同意将本次评估数据保存到当前设备",
        value=st.session_state.get("assessment_storage_consent", False),
        key="assessment_storage_consent_toggle",
        help="不勾选仍可查看和导出本次结果，但不会写入用户评估记录。",
    )
    st.session_state.assessment_storage_consent = storage_consent

    users = backend.list_users()
    user_by_id = {u["id"]: u for u in users}
    if st.session_state.active_user_id not in user_by_id:
        st.session_state.active_user_id = None
    loaded_id = st.session_state.get("profile_loaded_user_id")
    active_uid = st.session_state.active_user_id
    if active_uid:
        active_user_cache = user_by_id[active_uid]
        if (
            active_uid != loaded_id
            or active_user_cache.get("updated_at")
            != st.session_state.get("profile_loaded_updated_at")
        ):
            profile_to_state(active_user_cache)
            st.session_state.profile_loaded_user_id = active_uid
            st.session_state.profile_loaded_updated_at = active_user_cache.get("updated_at")

    with st.container(border=True):
        st.markdown("#### 当前用户")
        if users:
            options = [u["id"] for u in users]
            default_index = (
                options.index(st.session_state.active_user_id)
                if st.session_state.active_user_id in options
                else 0
            )
            selected = st.selectbox(
                "切换用户",
                options,
                index=default_index,
                format_func=lambda uid: user_by_id[uid]["name"],
                key="assessment_user_select",
            )
            if selected != st.session_state.active_user_id:
                st.session_state.active_user_id = selected
                st.session_state.profile_loaded_user_id = None
                st.rerun()
            active_user = user_by_id[selected]
            profile_col, act_col = st.columns(
                [0.72, 0.28], gap="medium", vertical_alignment="center"
            )
            with profile_col:
                name = active_user.get("name", "未命名")
                gender_cn = "女" if active_user.get("gender") != "male" else "男"
                st.markdown(
                    f'<div class="user-card" style="min-height:64px;padding:0.6rem 0.8rem;">'
                    f'<div class="user-avatar">{name[:1]}</div>'
                    f'<div class="user-card-body">'
                    f'<div class="user-card-name">{name}</div>'
                    f'<div style="color:var(--muted);font-size:0.85rem;">'
                    f"{active_user.get('age', '-')} 岁 · {gender_cn} · "
                    f"共病 {active_user.get('comorbidity_count', '-')} 项</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            with act_col:
                if st.button("更新档案", key="update_profile_from_form", width="stretch"):
                    updated = backend.update_user(
                        active_user["id"],
                        age=int(st.session_state.get("age", 74)),
                        gender="female" if st.session_state.get("gender", "女") == "女" else "male",
                        marital_status=(
                            "partner"
                            if st.session_state.get("marital", "有伴侣") == "有伴侣"
                            else "single"
                        ),
                        comorbidity_count=int(st.session_state.get("comorbidity", 2)),
                        self_rated_health=int(st.session_state.get("self_health", 3)),
                    )
                    st.session_state.profile_loaded_updated_at = (
                        updated.get("updated_at") if updated else None
                    )
                    st.toast("档案已更新")
                    st.rerun()
        else:
            active_user = None
            st.caption("还没有用户。填写下方姓名并保存，会自动同步到用户记录页。")

        st.caption("新用户")
        name_col, btn_col = st.columns([0.62, 0.38], vertical_alignment="center")
        with name_col:
            new_name = st.text_input(
                "新用户姓名 / 编号",
                key="new_user_name",
                placeholder="输入新用户姓名后创建",
                label_visibility="collapsed",
            )
        with btn_col:
            if st.button("创建用户并载入", type="primary", width="stretch"):
                new_name = (new_name or "").strip()
                if not new_name:
                    st.warning("请先填写姓名 / 编号")
                else:
                    created = backend.create_user(
                        name=new_name,
                        age=int(st.session_state.get("age", 74)),
                        gender=(
                            "female" if st.session_state.get("gender", "女") == "女" else "male"
                        ),
                        marital_status=(
                            "partner"
                            if st.session_state.get("marital", "有伴侣") == "有伴侣"
                            else "single"
                        ),
                        comorbidity_count=int(st.session_state.get("comorbidity", 2)),
                        self_rated_health=int(st.session_state.get("self_health", 3)),
                    )
                    st.session_state.active_user_id = created["id"]
                    st.session_state.selected_user_id = created["id"]
                    st.session_state.profile_loaded_user_id = created["id"]
                    st.session_state.profile_loaded_updated_at = created.get("updated_at")
                    st.rerun()

    left, right = st.columns([0.40, 0.60], gap="large")

    with left:
        with st.expander("基本信息与共病", expanded=True):
            age = st.slider("年龄", 60, 95, key="age")
            gender_label = st.selectbox("性别", ["女", "男"], key="gender")
            gender = "female" if gender_label == "女" else "male"
            married_label = st.selectbox(
                "婚姻状况", ["有伴侣", "独居 / 丧偶 / 离异"], key="marital"
            )
            married = married_label == "有伴侣"
            comorbidity_count = st.slider("慢性病数量", 0, 8, key="comorbidity")
            sleep_quality = st.select_slider(
                "睡眠质量",
                options=[1, 2, 3, 4],
                format_func=lambda x: {1: "很好", 2: "较好", 3: "一般", 4: "较差"}[x],
                key="sleep_quality",
            )
            self_health = st.select_slider(
                "自评健康",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: {1: "很好", 2: "好", 3: "一般", 4: "较差", 5: "很差"}[x],
                key="self_health",
            )

        with st.expander("量表作答", expanded=True):
            scale_options = ["cesd10", "phq9", "gds15", "hamd24"]
            scale_key = st.selectbox(
                "量表",
                scale_options,
                format_func=lambda x: SCALE_LABELS[x],
            )
            item_config = {
                "cesd10": (10, 3),
                "phq9": (9, 3),
                "gds15": (15, 1),
                "hamd24": (24, 4),
            }
            item_n, item_max = item_config[scale_key]
            answer_mode = st.segmented_control(
                "作答方式",
                options=["slider", "buttons"],
                default="slider",
                format_func=lambda x: "滑动作答" if x == "slider" else "大按钮作答",
            )
            answers = []
            cols = st.columns(2)
            for i in range(item_n):
                with cols[i % 2]:
                    if answer_mode == "buttons":
                        answers.append(
                            st.radio(
                                f"{i + 1}. {SCALE_ITEMS[scale_key][i]}",
                                options=list(range(item_max + 1)),
                                index=0,
                                horizontal=True,
                                key=f"scale_btn_{scale_key}_{i}",
                            )
                        )
                    else:
                        answers.append(
                            st.slider(
                                f"{i + 1}. {SCALE_ITEMS[scale_key][i]}",
                                min_value=0,
                                max_value=item_max,
                                value=0,
                                key=f"scale_{scale_key}_{i}",
                            )
                        )
            if scale_key == "cesd10":
                st.caption("选项：0=没有或很少（<1天），1=有时（1-2天），2=经常（3-4天），3=大部分/所有时间（5-7天）。")
            elif scale_key == "phq9":
                st.caption("选项：0=没有，1=几天，2=一半以上的天数，3=几乎每天。")
            elif scale_key == "gds15":
                st.caption("选项：0=否，1=是。部分题目会反向计分，系统按量表规则自动处理。")
            elif scale_key == "hamd24":
                st.caption("部分题目会反向计分，系统按量表规则自动处理。")
            if scale_key == "hamd24":
                st.caption("选项：0=无症状，1=轻度，2=中度，3=重度，4=极重度。")

        with st.expander("表情 / 行为线索", expanded=False):
            face_file = st.file_uploader("上传面部特征数据表（可选）", type=["csv"])
            expr_risk = 0.5
            if face_file is not None:
                try:
                    face_df = pd.read_csv(io.BytesIO(face_file.getvalue()))
                    expr = backend.expression_risk_from_openface(face_df)
                    expr_risk = expr["risk"]
                    st.metric("表情风险", f"{expr_risk:.0%}")
                    if expr.get("emotion_summary"):
                        st.json(expr["emotion_summary"])
                except Exception as exc:
                    st.warning(f"面部特征文件解析失败：{exc}")
            else:
                expr_risk = st.slider("表情风险（人工/视频摘要）", 0.0, 1.0, 0.5, 0.05)

    with right:
        if active_user:
            history = backend.list_assessments(active_user["id"])
            if history:
                latest = history[0]
                risk_text = {
                    "low": "低风险",
                    "medium": "中风险",
                    "high": "高风险",
                }.get(latest.get("risk_level"), "")
                st.caption(
                    f"{active_user['name']} · 最近评估 "
                    f"{float(latest.get('depression_probability') or 0):.0%} {risk_text}"
                )
            else:
                st.caption(f"{active_user['name']} · 暂无历史评估，保存后可在用户记录页查看趋势")
        else:
            st.caption("当前未选择用户，可在上方填写姓名创建。")
        st.markdown("#### 对话访谈")
        action_cols = st.columns([0.34, 0.30, 0.36])
        with action_cols[0]:
            if st.button("填入高风险示例回答", width="stretch"):
                fill_high_risk_demo()
                st.rerun()
        with action_cols[1]:
            if st.button("清空对话", width="stretch"):
                reset_conversation()
                st.rerun()
        with action_cols[2]:
            st.caption("问题会随回答动态调整")

        chat_box = st.container(border=True, height=300)
        with chat_box:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["text"])

        conversation_md = conversation_export_markdown(
            st.session_state.messages,
            "interview",
            st.session_state.agent_cues,
            st.session_state.safety_alert,
            memory=st.session_state.get("agent_memory") or {},
            guidance_titles=st.session_state.get("agent_guidance_titles") or [],
            accepted_actions=st.session_state.get("agent_accepted_actions") or [],
            safety_plan_status=st.session_state.get(
                "agent_safety_plan_status", ""
            ),
            clinical_summary=st.session_state.get("agent_clinical_summary", ""),
            support_preference=st.session_state.get(
                "agent_support_preference", ""
            ),
            general_medical_title=st.session_state.get(
                "agent_general_medical_title", ""
            ),
            medical_sources=st.session_state.get("agent_medical_sources") or [],
        )
        st.download_button(
            "导出对话记录",
            data=conversation_md.encode("utf-8"),
            file_name=f"interview_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            key="export_interview_chat",
        )

        user_input = st.chat_input("回答访谈问题，或自由描述近况")
        if user_input:
            st.session_state.messages.append({"role": "user", "text": user_input})
            agent_context = {
                "age": age,
                "gender": gender,
                "married": married,
                "comorbidity_count": comorbidity_count,
                "sleep_quality": sleep_quality,
                "self_rated_health": self_health,
                "scale_name": scale_key,
                "api_key": api_key,
                "cloud_consent": st.session_state.get("cloud_consent", False),
                "mode": "interview",
            }
            try:
                with st.spinner("正在接上下一句"):
                    agent_response = backend.agent_chat(
                        messages=st.session_state.messages,
                        context=agent_context,
                        session_id=st.session_state.session_id,
                    )
                store_interview_response(agent_response)
                reply = agent_response.get("reply") or "我还在听，您继续说。"
            except Exception:
                reply = "抱歉，刚才没有成功回复。请您再说一次，我会继续接着听。"
                st.session_state.agent_engine = "local"
                st.session_state.agent_llm_error = "请求失败"
            st.session_state.messages.append({"role": "assistant", "text": reply})
            st.rerun()

        st.caption(
            dialogue_engine_caption(
                st.session_state.agent_engine,
                st.session_state.get("agent_reason", ""),
                st.session_state.get("agent_latency_ms", 0),
                st.session_state.get("agent_prompt_version", ""),
            )
        )
        clinical_alert = st.session_state.get("agent_clinical_alert")
        if clinical_alert:
            st.error(
                f"安全提醒：{clinical_alert.get('message', '请尽快联系家人或医生评估。')}"
            )
        with st.expander("本轮对话记忆", expanded=False):
            memory = st.session_state.get("agent_memory") or {}
            covered = st.session_state.get("agent_covered_topics") or []
            if memory:
                for key, value in memory.items():
                    st.markdown(f"- **{key}**：{value}")
            else:
                st.caption("还没有记录到姓名、年龄、居住地或爱好等信息。")
            if covered:
                st.caption("已覆盖主题：" + "、".join(str(item) for item in covered))
            if st.session_state.get("agent_memory_summary"):
                st.caption(f"早前摘要：{st.session_state.agent_memory_summary}")
            if st.session_state.get("agent_focus"):
                st.caption(f"本轮重点：{st.session_state.agent_focus}")
            if st.session_state.get("agent_positive_context"):
                st.caption(
                    f"同时确认：{st.session_state.agent_positive_context}"
                )
            if st.session_state.get("agent_protective_factors"):
                st.caption(
                    "保护因素："
                    + "、".join(st.session_state.agent_protective_factors)
                )
            if st.session_state.get("agent_alliance_repair_count"):
                st.caption(
                    "已根据您的反馈调整对话方式 "
                    f"{st.session_state.agent_alliance_repair_count} 次"
                )
            if st.session_state.get("agent_accepted_actions"):
                st.caption(
                    "已确认行动："
                    + "、".join(st.session_state.agent_accepted_actions)
                )
            if st.session_state.get("agent_guidance_titles"):
                st.caption(
                    "本轮支持策略："
                    + "、".join(st.session_state.agent_guidance_titles)
                )
            if st.session_state.get("agent_clinical_summary"):
                st.caption(
                    "临床访谈摘要："
                    + st.session_state.agent_clinical_summary
                )
            if st.session_state.get("agent_support_preference"):
                preference_label = {
                    "listen": "先倾听",
                    "action": "一起想办法",
                    "both": "倾听并一起想办法",
                }.get(st.session_state.agent_support_preference, "")
                if preference_label:
                    st.caption(f"支持偏好：{preference_label}")
            if st.session_state.get("agent_general_medical_title"):
                st.caption(
                    "医学知识线索："
                    + st.session_state.agent_general_medical_title
                )
            for source in st.session_state.get("agent_medical_sources") or []:
                title = source.get("title_zh") or source.get("title") or "医学科普"
                url = source.get("url") or ""
                st.caption(
                    "MedlinePlus 参考："
                    + (f"[{title}]({url})" if url else title)
                )
            evidence = st.session_state.get("agent_scale_evidence") or []
            if evidence:
                st.markdown("**量表核对提示**")
                for item in evidence:
                    st.caption(
                        f"PHQ-9 第 {item['item']} 项（{item['label']}）："
                        f"{'、'.join(item['cues'])}，需量表确认"
                    )
        if st.session_state.safety_alert:
            st.error(
                "对话中出现了需要关注的安全表达。请优先联系家人或医生；"
                "全国心理援助热线 12356，紧急情况请拨打 120 / 110。"
            )
            safety_status = {
                "checking": "正在确认当前安全与可联系的人",
                "support_confirmed": "已确认有人陪伴，建议今天继续联系专业人员",
                "emergency_referral": "已进入紧急求助提示，请立即联系真人帮助",
            }.get(st.session_state.get("agent_safety_plan_status"), "")
            if safety_status:
                st.caption(safety_status)
        if st.session_state.agent_cues:
            st.markdown(
                "**已识别风险线索：** "
                + " ".join(
                    f'<span class="status-pill status-wait">{cue}</span>'
                    for cue in st.session_state.agent_cues
                ),
                unsafe_allow_html=True,
            )

        user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
        demographics = {
            "age": age,
            "gender": gender,
            "married": married,
            "comorbidity_count": comorbidity_count,
            "sleep_quality": sleep_quality,
            "self_rated_health": self_health,
        }
        scale_answers = {scale_key: answers}

        completer = None
        if (
            (api_key or os.getenv("OPENAI_API_KEY"))
            and st.session_state.get("cloud_consent", False)
        ):
            try:
                from elderly_depression.models.llm_assessor import openai_completer

                completer = openai_completer(api_key=api_key or None)
            except Exception as exc:
                st.warning(f"大模型未启用：{exc}")

        mpdd_risk = st.session_state.get("mpdd_inference_risk")
        result = backend.run_assessment(
            transcript=user_messages,
            scale_answers=scale_answers,
            demographics=demographics,
            expression_summary={"risk": expr_risk},
            scale_name=scale_key,
            mpdd_risk=mpdd_risk,
            language_cues=st.session_state.agent_cues,
            llm_complete=completer,
        )

        prob = result["depression_probability"]
        level = result["risk_level"]
        level_map = {"low": ("低风险", "#16a34a"), "medium": ("中风险", "#f59e0b"), "high": ("高风险", "#dc2626")}
        level_text, level_color = level_map[level]
        st.markdown(
            f'<span class="risk-badge" style="color:{level_color};">{level_text}</span> '
            f"<span style='color:var(--muted);font-size:0.85rem;'>依据对话、量表、共病与表情线索</span>",
            unsafe_allow_html=True,
        )

        gauge_col, factor_col = st.columns([0.46, 0.54], gap="large")
        with gauge_col:
            st.plotly_chart(risk_gauge(prob), width="stretch")
        with factor_col:
            st.markdown("#### 风险构成")
            scale_level_cn = SCALE_LEVEL_LABELS.get(result.get("scale_level"), result.get("scale_level"))
            factor_items = [
                ("语言线索", f"{result['text_risk']:.0%}", ""),
                ("量表风险", f"{result['scale_risk']:.0%}", ""),
                ("背景 / 共病", f"{result['context_risk']:.0%}", ""),
                ("量表分", f"{result['scale_score']}", f"等级: {scale_level_cn}"),
            ]
            if mpdd_risk is not None:
                factor_items.append(("MPDD 多模态", f"{mpdd_risk:.0%}", "来自被试级推理管线"))
            if result.get("expression_risk") is not None:
                factor_items.append(("表情/行为", f"{result['expression_risk']:.0%}", ""))
            for label, value, delta in factor_items:
                st.markdown(
                    f'<div class="factor-row"><span class="factor-label">{label}</span>'
                    f'<span class="factor-value">{value}</span>'
                    f'<span class="factor-delta">{delta}</span></div>',
                    unsafe_allow_html=True,
                )
            fusion_mode_cn = "已学习融合权重" if result.get("fusion_mode") == "learned" else "文献先验权重"
            st.caption(f"当前融合方式：{fusion_mode_cn}")
            text_method_cn = (
                "规则 + LLM 线索" if result.get("text_method") == "rule_llm" else "规则词典"
            )
            st.caption(f"语言评分方式：{text_method_cn}")
            if result.get("fusion_weights"):
                weight_text = "，".join(
                    f"{k} {v:+.3f}" for k, v in result["fusion_weights"].items()
                )
                st.caption(f"权重/系数：{weight_text}")
            if result.get("language_cues"):
                cue_text = "、".join(
                    f"{c['cue']}（{c['confidence']:.0%}）" for c in result["language_cues"]
                )
                st.markdown(f"**语言线索：** {cue_text}")
            if result.get("llm_summary"):
                st.markdown("**大模型摘要**")
                st.markdown(result["llm_summary"])
            else:
                st.markdown("**当前判断依据**")
                st.markdown(
                    "- 对话中的负面表达、睡眠/兴趣/躯体线索\n"
                    "- 量表作答对应的风险分\n"
                    "- 年龄、性别、婚姻与慢性病共病\n"
                    "- 表情/行为风险摘要"
                )
            report_md = assessment_report_markdown(
                active_user,
                demographics,
                result,
                scale_key,
                user_messages,
                st.session_state.safety_alert,
            )
            record = {
                "scale_name": scale_key,
                "scale_score": result["scale_score"],
                "scale_level": result.get("scale_level"),
                "depression_probability": result["depression_probability"],
                "risk_level": result["risk_level"],
                "text_risk": result["text_risk"],
                "scale_risk": result["scale_risk"],
                "context_risk": result["context_risk"],
                "expression_risk": result.get("expression_risk"),
                "mpdd_risk": result.get("mpdd_risk"),
                "fusion_mode": result.get("fusion_mode"),
                "language_cues": result.get("language_cues"),
                "demographics": demographics,
                "transcript": user_messages,
                "storage_consent": True,
                "consent_recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            signature = json.dumps(
                {
                    "scale": scale_key,
                    "answers": answers,
                    "demographics": demographics,
                    "transcript": user_messages,
                    "expr_risk": expr_risk,
                    "mpdd_risk": mpdd_risk,
                    "cues": st.session_state.agent_cues,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            action_cols = st.columns(3 if active_user else 2)
            with action_cols[0]:
                if active_user:
                    if st.button(
                        "保存本次评估到用户记录",
                        key="save_assessment",
                        width="stretch",
                        disabled=not storage_consent,
                    ):
                        if persist_assessment(active_user, record, signature):
                            st.toast("已保存到用户记录")
                            st.rerun()
                        else:
                            st.toast("本次结果已保存，无需重复保存")
            with action_cols[1 if active_user else 0]:
                st.download_button(
                    "导出评估报告",
                    data=report_md.encode("utf-8"),
                    file_name=f"depression_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    width="stretch",
                    key="export_risk_report",
                )
            speech_summary = (
                f"本次评估的抑郁风险概率为{prob:.0%}，风险等级为{level_text}。"
                "该结果仅用于科研筛查与风险提示，不替代临床诊断。"
            )
            with action_cols[2 if active_user else 1]:
                if st.button("朗读结果", key="speak_result", width="stretch"):
                    speak_text(speech_summary)
            if active_user:
                if st.button(
                    "保存并查看用户记录",
                    key="save_and_view_records",
                    width="stretch",
                    disabled=not storage_consent,
                ):
                    persist_assessment(active_user, record, signature)
                    st.session_state.selected_user_id = active_user["id"]
                    st.session_state.active_user_id = active_user["id"]
                    navigate_to("用户记录")
        st.caption(result["disclaimer"])


MODEL_LABELS = {
    "logistic": "逻辑回归",
    "random_forest": "随机森林",
    "xgboost": "梯度提升树",
}

METHOD_LABELS = {
    "none": "不处理",
    "smote": "合成少数类过采样",
    "smote_enn": "清理后采样",
    "smote_tomek": "清理后采样（Tomek）",
    "adasyn": "自适应合成采样",
    "oversample": "随机过采样",
    "undersample": "随机欠采样",
}

MODALITY_LABELS = {
    "face": "面部",
    "audio": "语音",
    "gait": "步态",
    "personality": "人格",
    "fusion": "融合",
}

METRIC_LABELS = {
    "accuracy": "准确率",
    "precision": "精确率",
    "recall": "召回率",
    "f1": "F1 分数",
    "auc": "曲线下面积",
    "balanced_accuracy": "平衡准确率",
    "sensitivity": "灵敏度",
    "specificity": "特异度",
    "pr_auc": "PR-AUC",
    "brier": "Brier 分数",
    "brier_uncalibrated": "校准前 Brier",
    "brier_calibrated": "校准后 Brier",
    "ece_uncalibrated": "校准前 ECE",
    "ece_calibrated": "校准后 ECE",
    "threshold": "阈值",
    "positive_rate": "预测阳性率",
    "f1_macro": "F1 宏平均",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "phq_binary_accuracy": "PHQ 二分类准确率",
    "model": "模型",
    "resample_method": "采样方法",
    "modality": "模态",
}


def chinese_tabular_report(report: pd.DataFrame) -> pd.DataFrame:
    report = report.copy()
    if "model" in report.columns:
        report["model"] = report["model"].map(MODEL_LABELS).fillna(report["model"])
    return report.rename(columns=METRIC_LABELS)


def chinese_imbalance_report(report: pd.DataFrame) -> pd.DataFrame:
    report = report.copy()
    if "resample_method" in report.columns:
        report["resample_method"] = (
            report["resample_method"].map(METHOD_LABELS).fillna(report["resample_method"])
        )
    return report.rename(columns=METRIC_LABELS)


def chinese_multimodal_report(report: pd.DataFrame) -> pd.DataFrame:
    report = report.copy()
    if "modality" in report.columns:
        report["modality"] = report["modality"].map(MODALITY_LABELS).fillna(report["modality"])
    return report.rename(columns=METRIC_LABELS)


def chinese_subject_report(report: pd.DataFrame) -> pd.DataFrame:
    report = report.copy()
    if "modality" in report.columns:
        report["modality"] = report["modality"].map(MODALITY_LABELS).fillna(report["modality"])
    rename = {}
    for col in report.columns:
        suffix = ""
        base = col
        if col.endswith("_mean"):
            suffix = "均值"
            base = col[:-5]
        elif col.endswith("_std"):
            suffix = "±"
            base = col[:-4]
        label = METRIC_LABELS.get(base)
        if label:
            rename[col] = f"{label}{suffix}"
    return report.rename(columns=rename)


def chinese_advanced_report(report: pd.DataFrame) -> pd.DataFrame:
    report = report.copy()
    rename = {col: METRIC_LABELS[col] for col in report.columns if col in METRIC_LABELS}
    return report.rename(columns=rename)


def render_charls() -> None:
    st.subheader("养老调查数据与表格模型")
    if not backend.charls_exists():
        st.info("尚未生成养老调查标准表。点击下面按钮读取本机中国健康与养老追踪调查数据。")
        if st.button("生成养老调查标准表", type="primary"):
            with st.spinner("正在读取中国健康与养老追踪调查数据并构建标准表..."):
                backend.build_charls_dataset()
                st.cache_data.clear()
            st.rerun()
        return

    df = load_charls_cached()
    pos_rate = float(df["depression"].mean())
    st.markdown("#### 数据概览")
    cols = st.columns(5)
    cols[0].metric("样本量", f"{len(df):,}")
    cols[1].metric("特征数", len(df.columns) - 4)
    cols[2].metric("抑郁量表高风险率", f"{pos_rate:.1%}")
    cols[3].metric("平均年龄", f"{df['age'].mean():.1f} 岁")
    cols[4].metric("平均共病数", f"{df['comorbidity_count'].mean():.1f}")

    plot_col, prev_col = st.columns(2, gap="large")
    with plot_col:
        fig_age = px.histogram(
            df,
            x="age",
            color="depression",
            barmode="overlay",
            nbins=20,
            opacity=0.72,
            color_discrete_map={0: "#64748b", 1: "#ef4444"},
            labels={"age": "年龄", "count": "样本数", "depression": "抑郁高风险"},
        )
        fig_age.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_age, width="stretch")

    with prev_col:
        chronic_cols = [c for c in df.columns if c.startswith("chronic_")]
        prev = df[chronic_cols].mean().sort_values(ascending=False) * 100
        fig_prev = px.bar(
            prev,
            orientation="h",
            color=prev.values,
            color_continuous_scale=["#dbeafe", "#3b82f6", "#7c3aed"],
            labels={"index": "慢性病", "value": "患病率 (%)"},
        )
        fig_prev.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
        st.plotly_chart(fig_prev, width="stretch")

    st.markdown("#### 表格模型结果")
    if st.session_state.get("tabular_report") is None:
        st.session_state.tabular_report = backend.load_csv(backend.MODEL_REPORT)
    report = st.session_state.tabular_report
    if report is not None and not report.empty:
        display_report = chinese_tabular_report(report)
        st.dataframe(display_report, width="stretch", hide_index=True)
        fig_model = px.bar(
            display_report,
            x="模型",
            y=["曲线下面积", "F1 分数"],
            barmode="group",
            color_discrete_map={"曲线下面积": "#2563eb", "F1 分数": "#f59e0b"},
            labels={"value": "分数", "variable": "指标", "模型": "模型"},
        )
        fig_model.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_model, width="stretch")
    else:
        st.info("还没有表格基线结果。")

    st.markdown("#### 操作")
    op_cols = st.columns(2)
    with op_cols[0]:
        if st.button("重新生成养老调查表", width="stretch"):
            with st.spinner("正在重建标准表..."):
                backend.build_charls_dataset()
                st.cache_data.clear()
            st.rerun()
    with op_cols[1]:
        if st.button("重新训练基线（SMOTE-Tomek）", type="primary", width="stretch"):
            with st.spinner("正在训练逻辑回归、随机森林、梯度提升树..."):
                st.session_state.tabular_report = backend.train_tabular_baseline("smote_tomek")
            st.rerun()

    st.markdown("---")
    st.markdown("#### 时间外验证")
    st.caption("只用早期波次训练、用之后波次做验证，更接近“预测未来抑郁风险”而不是同一时期分类。")
    temporal_cols = st.columns(2)
    with temporal_cols[0]:
        if st.button("运行时间外验证（前波次 → 后波次）", type="primary", width="stretch"):
            with st.spinner("正在按波次划分训练集和测试集并训练三类模型..."):
                st.session_state.temporal_report = backend.train_charls_temporal("smote_tomek")
            st.rerun()
    with temporal_cols[1]:
        st.caption("例如：Wave 1-3 训练 → Wave 4 验证；系统会对每个后续波次都生成一条时间外结果。")

    if st.session_state.get("temporal_report") is None:
        st.session_state.temporal_report = backend.charls_temporal_report()
    temporal_report = st.session_state.temporal_report
    if temporal_report is not None and not temporal_report.empty:
        display_temporal = chinese_tabular_report(temporal_report)
        if "test_wave" in display_temporal.columns:
            display_temporal = display_temporal.rename(columns={"test_wave": "验证波次"})
            display_temporal["验证波次"] = display_temporal["验证波次"].astype(int)
        if "train_rows" in display_temporal.columns:
            display_temporal = display_temporal.rename(columns={"train_rows": "训练样本", "test_rows": "验证样本"})
        st.dataframe(display_temporal, width="stretch", hide_index=True)
        fig_temporal = px.bar(
            display_temporal,
            x="验证波次",
            y=["曲线下面积", "F1 分数"],
            color="模型",
            barmode="group",
            labels={"value": "分数", "variable": "指标"},
        )
        fig_temporal.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_temporal, width="stretch")
    else:
        st.info("还没有时间外验证结果。")

    st.markdown("---")
    st.markdown("#### 个体纵向轨迹")
    st.caption("用同一人的上一波次特征预测当前波次抑郁，并按人分组切分，避免多波次信息泄漏。")
    long_cols = st.columns(2)
    with long_cols[0]:
        if st.button("运行个体纵向轨迹（上一波次 → 当前）", type="primary", width="stretch"):
            with st.spinner("正在构造滞后特征并按人切分训练..."):
                st.session_state.longitudinal_report = backend.train_charls_longitudinal(
                    "smote_tomek"
                )
            st.rerun()
    with long_cols[1]:
        st.caption("特征包含上一波次年龄、共病、睡眠、自评健康、认知和 CESD 分。")

    if st.session_state.get("longitudinal_report") is None:
        st.session_state.longitudinal_report = backend.charls_longitudinal_report()
    long_report = st.session_state.longitudinal_report
    if long_report is not None and not long_report.empty:
        display_long = chinese_tabular_report(long_report)
        rename_map = {
            "split": "切分方式",
            "train_ids": "训练人数",
            "test_ids": "验证人数",
            "train_rows": "训练样本",
            "test_rows": "验证样本",
        }
        display_long = display_long.rename(columns=rename_map)
        st.dataframe(display_long, width="stretch", hide_index=True)
        fig_long = px.bar(
            display_long,
            x="模型",
            y=["曲线下面积", "F1 分数"],
            barmode="group",
            color_discrete_map={"曲线下面积": "#2563eb", "F1 分数": "#f59e0b"},
            labels={"value": "分数", "variable": "指标"},
        )
        fig_long.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_long, width="stretch")
    else:
        st.info("还没有个体纵向轨迹结果。")

    st.markdown("---")
    st.markdown("#### 可解释性、校准与分层")
    research = backend.research_reports()
    feature_imp = research.get("charls_feature_importance")
    if feature_imp:
        st.markdown("**特征重要性（置换 + 逻辑系数）**")
        st.dataframe(pd.DataFrame(feature_imp).head(15), width="stretch", hide_index=True)
    cal_curve = research.get("charls_calibration_curve")
    if cal_curve:
        st.markdown("**校准可靠性曲线**")
        cal_df = pd.DataFrame(cal_curve)
        fig_cal = px.line(
            cal_df,
            x="predicted_freq",
            y="observed_freq",
            markers=True,
            labels={"predicted_freq": "预测频率", "observed_freq": "实际频率"},
        )
        fig_cal.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#dc2626", dash="dash"),
                name="理想校准线",
            )
        )
        fig_cal.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_cal, width="stretch")
    decision_curve = research.get("charls_decision_curve")
    if decision_curve:
        st.markdown("**决策曲线（净获益）**")
        dca_df = pd.DataFrame(decision_curve)
        fig_dca = px.line(
            dca_df,
            x="threshold",
            y=["net_benefit", "treat_all_benefit"],
            labels={"value": "净获益", "threshold": "阈值", "variable": "策略"},
            color_discrete_map={"net_benefit": "#2563eb", "treat_all_benefit": "#64748b"},
        )
        fig_dca.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_dca, width="stretch")
    bootstrap_ci = research.get("charls_bootstrap_ci")
    if bootstrap_ci:
        st.caption(
            f"AUC {bootstrap_ci['auc_point']:.3f} (95% CI "
            f"{bootstrap_ci['auc_ci'][0]:.3f}-{bootstrap_ci['auc_ci'][1]:.3f}) · "
            f"Brier {bootstrap_ci['brier_point']:.3f} (95% CI "
            f"{bootstrap_ci['brier_ci'][0]:.3f}-{bootstrap_ci['brier_ci'][1]:.3f})"
        )
    subgroup = research.get("charls_subgroup")
    if subgroup:
        st.markdown("**年龄/性别/共病分层报告**")
        st.dataframe(pd.DataFrame(subgroup), width="stretch", hide_index=True)


@st.cache_data(show_spinner=False)
def load_charls_cached() -> pd.DataFrame:
    return backend.load_charls()


def render_imbalance() -> None:
    st.subheader("类别不平衡方法对比")
    st.caption("同一组梯度提升树参数下，比较不处理、合成少数类过采样、清理后采样、自适应合成采样、过采样和欠采样。")
    if st.session_state.get("imbalance_report") is None:
        st.session_state.imbalance_report = backend.load_csv(backend.IMBALANCE_REPORT)
    report = st.session_state.imbalance_report
    if report is not None and not report.empty:
        display_report = chinese_imbalance_report(report)
        st.dataframe(display_report, width="stretch", hide_index=True)
        fig = px.bar(
            display_report,
            x="采样方法",
            y=["曲线下面积", "F1 分数"],
            barmode="group",
            color_discrete_map={"曲线下面积": "#2563eb", "F1 分数": "#f59e0b"},
            labels={"value": "分数", "variable": "指标", "采样方法": "方法"},
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("还没有不平衡对比结果。")

    if st.button("运行不平衡方法对比", type="primary"):
        with st.spinner("正在训练 6 组梯度提升树对比实验..."):
            st.session_state.imbalance_report = backend.run_imbalance_comparison()
        st.rerun()


def render_fusion() -> None:
    st.subheader("融合权重校准")
    st.caption(
        "上传一份信号概率数据表（text_risk / scale_risk / context_risk / expression_risk，"
        "可选 mpdd_risk）和标签列 depression，训练逻辑回归融合模型；"
        "训练后风险评估页会自动优先使用学习到的权重，未训练时仍用文献先验权重。"
    )

    status = backend.fusion_status()
    if status.get("ready"):
        calibrated_text = "，概率已做 Isotonic 校准" if status.get("calibrated") else ""
        st.success(f"已训练融合模型{calibrated_text}，风险评估将优先使用学习权重。")
        variants = status.get("variants", {})
        for name, variant in variants.items():
            label = "含 MPDD" if name == "with_mpdd" else "无 MPDD"
            st.markdown(f"**{label}**")
            st.dataframe(
                pd.DataFrame([variant.get("coefficients", {})]),
                width="stretch",
                hide_index=True,
            )
            st.caption(f"阈值：{variant.get('threshold', 0.5):.3f}")
    else:
        st.info("当前使用文献先验权重，训练后可自动切换为数据学习权重。")

    components = backend.fusion_components()
    if components:
        st.markdown("#### 真实组件学习")
        context = components.get("context")
        if context:
            st.markdown("**背景/共病组件（CHARLS 真实校准）**")
            st.caption(
                f"截距 {context['intercept']:.3f} · 系数 {context['coefficient']:.3f} · "
                f"AUC {context['auc']:.3f} · F1 {context['f1']:.3f} · Brier {context['brier']:.3f}"
            )
        mpdd_text = components.get("mpdd_text")
        if mpdd_text:
            st.markdown("**MPDD 描述文本信号**")
            st.caption(
                f"AUC {mpdd_text['mean_auc']:.3f} ± {mpdd_text['auc_std']:.3f} · "
                f"F1 {mpdd_text['mean_f1']:.3f}"
            )
        if not context and not mpdd_text:
            st.caption("尚未生成真实组件学习结果。")

    research = backend.research_reports()
    mpdd_struct = research.get("mpdd_structured_text")
    if mpdd_struct:
        st.markdown("#### MPDD 结构化文本特征")
        st.json(mpdd_struct)
    mpdd_cal = research.get("mpdd_calibration_dca")
    if mpdd_cal:
        st.markdown("#### MPDD 校准与决策曲线（holdout）")
        for name, pipeline in mpdd_cal.get("pipelines", {}).items():
            st.caption(
                f"{name}：AUC {pipeline['auc']:.3f} · Brier {pipeline['brier']:.3f}"
            )
        st.caption(
            f"固定 holdout：训练 {mpdd_cal.get('tune_subjects')} 人 / 测试 "
            f"{mpdd_cal.get('test_subjects')} 人"
        )
    fusion_rows = research.get("fusion_training_rows")
    if fusion_rows is not None:
        st.caption(f"已积累融合训练数据行数：{fusion_rows}")

    uploaded = st.file_uploader("上传融合训练数据 CSV", type=["csv"], key="fusion_csv")
    if uploaded is not None:
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        st.dataframe(df.head(20), width="stretch")
        label_col = "depression" if "depression" in df.columns else "label"
        if st.button("训练融合模型", type="primary"):
            with st.spinner("正在训练逻辑回归融合模型..."):
                resp = backend.train_fusion(df.to_dict(orient="records"), label_col=label_col)
                st.session_state.fusion_report = pd.DataFrame(resp.get("report", []))
            st.rerun()

    if st.session_state.get("fusion_report") is None:
        st.session_state.fusion_report = backend.fusion_report()
    report = st.session_state.fusion_report
    if report is not None and not report.empty:
        st.markdown("#### 融合模型报告")
        display = report.copy()
        if "variant" in display.columns:
            display["variant"] = (
                display["variant"]
                .map({"no_mpdd": "无 MPDD", "with_mpdd": "含 MPDD"})
                .fillna(display["variant"])
            )
        st.dataframe(chinese_imbalance_report(display), width="stretch", hide_index=True)

        curve = backend.fusion_calibration_curve()
        if curve is not None and not curve.empty:
            st.markdown("#### 校准可靠性曲线")
            curve["阶段"] = curve["stage"].map({"raw": "校准前", "calibrated": "校准后"})
            fig_cal = px.line(
                curve,
                x="bin_mid",
                y="observed_freq",
                color="阶段",
                markers=True,
                labels={
                    "bin_mid": "预测概率分箱",
                    "observed_freq": "实际阳性率",
                    "阶段": "阶段",
                },
                color_discrete_map={"校准前": "#64748b", "校准后": "#2563eb"},
            )
            fig_cal.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color="#dc2626", dash="dash"),
                    name="理想校准线",
                )
            )
            fig_cal.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_cal, width="stretch")


def render_data_cache() -> None:
    st.subheader("数据存储优化")
    st.caption(
        "把宽表特征转成 Parquet（float32 + zstd）并按模态分文件缓存，"
        "读取更快、占用更小；训练逻辑、交叉验证和 PCA 切分完全不变。"
    )
    status = backend.mpdd_cache_status()

    cols = st.columns(4)
    cols[0].metric("原始 .npy", human_bytes(status.get("raw_npy_bytes", 0)))
    cols[1].metric("特征 CSV", human_bytes(status.get("utterance_csv_bytes", 0)))
    cols[2].metric("Parquet 缓存", human_bytes(status.get("parquet_bytes", 0)))
    raw = float(status.get("raw_npy_bytes") or 0)
    parquet = float(status.get("parquet_bytes") or 0)
    ratio = (1 - parquet / raw) if raw > 0 else 0
    cols[3].metric("相对原始节省", f"{ratio:.0%}")

    if status.get("parquet_ready"):
        st.success("紧凑缓存已生成，多模态训练会自动优先读取 Parquet；未生成时自动回退 CSV。")
    else:
        st.info("还没有紧凑缓存，点击下面按钮生成。")

    if st.button("生成紧凑特征缓存", type="primary"):
        with st.spinner("正在转换特征表、写入分模态 Parquet 和 manifest..."):
            backend.build_mpdd_cache()
        st.rerun()

    if status.get("manifest_fresh"):
        st.caption("manifest 已是最新，重复构建会跳过未变化的原始文件。")
    else:
        st.caption("manifest 尚未生成或原始文件有变化，首次构建会扫描一次 .npy 目录。")


def render_jobs() -> None:
    st.subheader("后台任务")
    st.caption("长任务异步执行，提交后可刷新查看状态，不再阻塞页面。")
    task_labels = {
        "tabular_train": "表格基线训练",
        "tabular_temporal": "表格时间外验证",
        "tabular_longitudinal": "表格纵向轨迹",
        "tabular_synthetic": "表格模拟组训练",
        "imbalance": "不平衡方法对比",
        "mpdd_build": "MPDD 特征构建",
        "mpdd_augmentation": "MPDD 片段增强",
        "synthetic_generate": "生成模拟数据",
        "real_profile": "真实数据画像",
        "fusion_optimize": "融合优化",
    }
    task = st.selectbox(
        "任务",
        list(task_labels.keys()),
        format_func=lambda t: task_labels[t],
    )
    if st.button("提交任务", type="primary"):
        resp = backend.submit_job(task)
        if "job_id" in resp:
            st.toast("任务已提交", icon="✅")
            st.success(f"已提交：{resp['job_id']}")
            st.rerun()
        else:
            st.toast("提交失败", icon="⚠️")
            st.error(resp.get("error", "提交失败"))
    if st.button("刷新任务状态"):
        st.rerun()

    @st.fragment(run_every=5)
    def jobs_panel():
        jobs = backend.list_jobs()
        if jobs:
            st.markdown("#### 最近任务（每 5 秒自动刷新）")
            for job in jobs:
                row_cols = st.columns([0.18, 0.14, 0.12, 0.30, 0.26])
                row_cols[0].write(job.get("task", ""))
                row_cols[1].write(job.get("status", ""))
                row_cols[2].write(f"{float(job.get('progress', 0)) * 100:.0f}%")
                row_cols[3].write(job.get("stage", ""))
                if job.get("status") == "running":
                    if row_cols[4].button(
                        "取消",
                        key=f"cancel_{job['id']}",
                        width="stretch",
                    ):
                        backend.cancel_job(job["id"])
                        st.rerun()
                else:
                    row_cols[4].write(job.get("error") or "")
            st.divider()
        else:
            st.info("还没有后台任务。")

    jobs_panel()


def render_synthetic() -> None:
    st.subheader("模拟数据")
    st.caption(
        "模拟组由真实数据中的主题/特征元素重组生成，只作为独立的训练参数；"
        "不与真实数据混合，评估只用真实数据。"
    )
    st.markdown("#### 真实数据特征分析")
    profile = backend.real_profile_status()
    if profile.get("ready"):
        tab = profile.get("tabular", {})
        mm = profile.get("multimodal", {})
        dia = profile.get("dialogue", {})
        cols = st.columns(4)
        cols[0].metric("表格样本", tab.get("rows", 0))
        cols[1].metric("高风险率", f"{float(tab.get('positive_rate', 0)):.1%}")
        cols[2].metric("缺失率", f"{float(tab.get('missing_rate', 0)):.1%}")
        cols[3].metric("对话语句", dia.get("sentences", 0))
        st.caption(
            "多模态 PCA 前 10 维累计方差："
            + "；".join(
                f"{name} {float(profile['multimodal'][name]['pca_cumulative_variance'][-1] * 100):.0f}%"
                for name in ["face", "audio", "gait", "personality"]
                if profile.get("multimodal", {}).get(name, {}).get("pca_cumulative_variance")
            )
        )
    else:
        st.info("还没有真实数据特征分析，点击下方按钮生成。")
    if st.button("生成真实数据特征分析"):
        with st.spinner("正在分析表格分布、多模态 PCA 和对话主题..."):
            backend.generate_real_profile()
        st.rerun()

    st.markdown("#### 多模态模拟组 v3（层次化 + 联合潜空间）")
    v3 = backend.synthetic_v3_status()
    if v3.get("ready") and v3.get("validation"):
        val = v3["validation"]
        if val.get("feasible"):
            st.success("通过分布验证门槛，可用于训练侧实验。")
        else:
            st.warning("未通过分布验证门槛，不建议作为训练参数。")
        st.json(val.get("gates", {}))
        if v3.get("grid") is not None:
            st.markdown("**网格搜索（边际校准 + 验证门槛）**")
            st.dataframe(pd.DataFrame(v3["grid"]), width="stretch", hide_index=True)
        if v3.get("best_params"):
            st.caption(f"最优参数：{v3['best_params']}")
        v3_report = PROJECT_ROOT / "output" / "synthetic_v3_subject_report.csv"
        if v3_report.exists():
            st.dataframe(pd.read_csv(v3_report), width="stretch", hide_index=True)
    else:
        st.info("尚未运行 v3，点击下方按钮生成并对比。")
    if st.button("运行 v3 层次化模拟组并对比", type="primary"):
        with st.spinner("正在生成合成被试、验证分布并跑 subject-aware 对比..."):
            backend.run_synthetic_v3()
        st.rerun()
    if st.button("运行 v3 网格搜索（边际校准 + 最优参数）", type="primary"):
        with st.spinner("正在网格搜索参数并验证分布，完成后跑最优 subject-aware 对比..."):
            backend.run_synthetic_v3_grid()
        st.rerun()

    st.markdown("#### 片段级增强（新方向）")
    st.caption(
        "不再生成新被试，而是在真实片段上做被试内小噪声复制，保留真实分布与被试结构。"
    )
    if v3.get("augmentation_ready"):
        st.success("片段级增强数据已生成")
        aug_report = v3.get("augmentation_report")
        if aug_report:
            st.dataframe(pd.DataFrame(aug_report), width="stretch", hide_index=True)
        gait_grid = v3.get("gait_augmentation_grid")
        if gait_grid:
            st.markdown("**gait 分模态噪声网格**")
            st.dataframe(pd.DataFrame(gait_grid), width="stretch", hide_index=True)
        tuning_grid = v3.get("augmentation_tuning_grid")
        if tuning_grid:
            st.markdown("**科学调参：调参集 subject-aware 网格**")
            st.dataframe(pd.DataFrame(tuning_grid), width="stretch", hide_index=True)
        if v3.get("augmentation_best_params"):
            st.caption(f"最优参数（仅在调参集选择）：{v3['augmentation_best_params']}")
        final_report = v3.get("augmentation_final_report")
        if final_report:
            st.markdown("**最终测试集报告（未见过的受试者）**")
            st.dataframe(pd.DataFrame(final_report), width="stretch", hide_index=True)
        repeated_summary = v3.get("repeated_holdout_summary")
        if repeated_summary:
            st.markdown("**重复 holdout 汇总（5 次 80/20）**")
            st.dataframe(pd.DataFrame(repeated_summary), width="stretch", hide_index=True)
    else:
        st.info("尚未运行片段级增强实验。")
    if st.button("运行片段级增强实验", type="primary"):
        with st.spinner("正在生成增强片段、验证分布并跑 subject-aware 对比..."):
            backend.run_segment_augmentation()
        st.rerun()
    if st.button("运行科学调参 + 最终测试集验证", type="primary"):
        with st.spinner("正在调参集搜索、最终测试集评估，避免参数泄漏..."):
            backend.run_augmentation_optimization()
        st.rerun()
    if st.button("运行重复 holdout（5 次 80/20）", type="primary"):
        with st.spinner("正在用固定参数重复 5 次受试者切分评估..."):
            backend.run_repeated_holdout()
        st.rerun()

    st.markdown("#### 训练协议：模拟组训练 + 真实组验证")
    tabular_syn_report = backend.load_csv(
        PROJECT_ROOT / "output" / "tabular_synthetic_report.csv"
    )
    if tabular_syn_report is not None and not tabular_syn_report.empty:
        st.dataframe(chinese_tabular_report(tabular_syn_report), width="stretch", hide_index=True)
    if st.button("表格：模拟组训练 + 真实组验证", type="primary"):
        with st.spinner("正在用表格模拟组训练、真实受试者 holdout 评估..."):
            backend.train_tabular_synthetic()
        st.rerun()
    st.caption(
        "多模态片段级增强实验即多模态版协议；真实组只用于验证和评估，不参与训练。"
    )
    ab_history = backend.research_reports().get("ab_history")
    if ab_history:
        st.markdown("#### A/B 回归检查历史")
        st.dataframe(pd.DataFrame(ab_history), width="stretch", hide_index=True)

    status = backend.synthetic_status()
    if status.get("ready"):
        st.success("模拟组已生成")
        dialogue = status.get("dialogue", {})
        tabular = status.get("tabular", {})
        multimodal = status.get("multimodal", {})
        cols = st.columns(3)
        cols[0].metric("对话样本", dialogue.get("rows", 0))
        cols[1].metric("表格样本", tabular.get("rows", 0))
        cols[2].metric("多模态样本", multimodal.get("rows", 0))
        st.markdown("#### 样本分布")
        st.json(dialogue.get("risk_counts", {}))
        st.caption(
            f"表格模拟组阳性率 {float(tabular.get('positive_rate', 0)):.1%} · "
            f"多模态模拟组阳性率 {float(multimodal.get('positive_rate', 0)):.1%}"
        )
        dialogue_csv = Path(status.get("dialogue", {}).get("path", ""))
        if dialogue_csv.exists():
            st.markdown("#### 对话样本预览")
            st.dataframe(pd.read_csv(dialogue_csv).head(5), width="stretch", hide_index=True)
        dialogue_val = status.get("dialogue_validation")
        if dialogue_val and dialogue_val.get("ready"):
            st.markdown("#### 对话模拟组验证")
            if dialogue_val.get("feasible"):
                st.success("通过对话分布与风险标签一致性验证。")
            else:
                st.warning("未通过对话验证门槛。")
            st.json(dialogue_val.get("metrics", {}))
        if st.button("验证对话模拟组"):
            with st.spinner("正在计算对话分布一致性与风险标签一致性..."):
                backend.validate_dialogue_samples()
            st.rerun()

        comparison_v2 = PROJECT_ROOT / "output" / "synthetic_comparison_v2_report.csv"
        comparison_path = (
            comparison_v2
            if comparison_v2.exists()
            else PROJECT_ROOT / "output" / "synthetic_comparison_report.csv"
        )
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path)
            display = comparison.copy()
            display["训练数据"] = display["source"].map(
                {"real": "真实组", "synthetic": "模拟组"}
            )
            rename = {
                "model": "模型",
                "auc": "曲线下面积",
                "f1": "F1 分数",
                "threshold": "阈值",
                "positive_rate": "预测阳性率",
            }
            display = display.rename(columns=rename)
            st.markdown("#### 历史：真实组 vs 旧版聚合式模拟组")
            st.caption(
                "这里展示的是旧版聚合式多模态模拟组（v1/v2）回溯对比；"
                "两种训练方式都在同一批真实保留数据上评估。当前正式推荐方向为下方片段级增强。"
            )
            st.dataframe(display, width="stretch", hide_index=True)
            st.markdown(
                "**历史结论：** 表格模拟组与真实组差距很小（AUC 约 0.82 vs 0.84），"
                "可以作为训练参数；旧版聚合式多模态模拟组曾出现人格模态 AUC 1.0 等"
                "虚高信号，未通过训练可行性验证，仅保留为实验探索。"
            )
            st.markdown("#### 当前可行性结论")
            if dialogue_val and dialogue_val.get("feasible"):
                st.success("**对话模拟组**：通过分布与风险标签一致性验证，可作为 Agent/文本训练参数。")
            if tabular.get("positive_rate") is not None:
                st.success(
                    "**表格模拟组**：已追平真实组样本量，训练对比与真实组接近，可作为训练参数。"
                )
            if v3.get("validation") and v3["validation"].get("feasible"):
                st.info(
                    "**多模态 v3 聚合式模拟**：分布验证门槛已通过，但 subject-aware 训练对比"
                    "仍弱于真实组，暂不作为正式训练参数，保留为实验探索。"
                )
            if v3.get("augmentation_ready") and v3.get("augmentation_final_report"):
                st.success(
                    "**多模态片段级增强**：最终测试集与重复 holdout 均通过，可作为独立训练参数。"
                )
    else:
        st.info("尚未生成模拟数据，点击下面按钮生成。")

    if st.button("生成模拟数据", type="primary"):
        with st.spinner("正在生成对话、表格、多模态模拟组..."):
            backend.generate_synthetic()
        st.rerun()

    with st.expander("删除模拟数据", expanded=False):
        st.warning("删除后模拟组文件会被移除，不会影响真实数据和已有模型。")
        if st.checkbox("确认删除模拟数据"):
            if st.button("删除模拟数据", type="primary"):
                backend.delete_synthetic()
                st.rerun()

    st.markdown("#### 使用边界")
    st.markdown(
        "- 模拟组仅用于训练侧实验，不会混入真实训练集\n"
        "- 模型效果必须用真实保留数据评估\n"
        "- 生成报告记录样本量、阳性率和所用方法，便于论文复现\n"
        "- 旧版聚合式多模态模拟组仅作实验探索，正式训练参数采用通过验证的片段级增强"
    )


def render_mpdd_dataset() -> None:
    st.subheader("MPDD 多模态抑郁检测")
    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("从本地 MPDD 原始特征构建数据表", type="primary", width="stretch"):
            with st.spinner("正在扫描 Train/Test 特征并聚合成表..."):
                backend.build_mpdd_dataset()
                st.cache_data.clear()
            st.rerun()
    with action_cols[1]:
        st.caption("首次构建会读取 data/mpdd 下的官方 .npy 特征，之后可直接训练多模态模型。")

    uploaded = st.file_uploader("上传已合并的多模态特征数据表（可选）", type=["csv"])
    if uploaded is not None:
        st.session_state.mpdd_df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        Path(backend.MPDD_CSV).parent.mkdir(parents=True, exist_ok=True)
        st.session_state.mpdd_df.to_csv(backend.MPDD_CSV, index=False, encoding="utf-8-sig")

    if backend.mpdd_exists():
        mpdd_path = backend.MPDD_UPLOAD_CSV if backend.MPDD_UPLOAD_CSV.exists() else backend.MPDD_CSV
        df = pd.read_csv(mpdd_path)
        st.session_state.mpdd_df = df

    df = st.session_state.mpdd_df
    if df is None:
        st.info(
            "本机还没有多模态特征表。数据包已放入 data/mpdd 后，点击“从本地 MPDD 原始特征构建数据表”即可，"
            "也可以直接上传已合并的数据表。"
        )
        return

    st.markdown("#### 数据概览")
    cols = st.columns(4)
    cols[0].metric("样本量", len(df))
    cols[1].metric("特征数", len(df.columns))
    if "hamd_score" in df.columns:
        cols[2].metric("PHQ-9 量表均值", f"{df['hamd_score'].mean():.1f}")
        binary = (df["hamd_score"] >= 8).astype(int)
        cols[3].metric("PHQ-9 ≥8 比例", f"{binary.mean():.1%}")

    prefixes = []
    for prefix, label in [
        ("face", "面部"),
        ("audio", "语音"),
        ("gait", "步态"),
        ("personality", "人格"),
    ]:
        count = sum(c.startswith(prefix + "_") for c in df.columns)
        if count:
            prefixes.append(f"{label}（{count} 维）")
    st.markdown(
        "**检测到的模态：** " + "、".join(prefixes)
        if prefixes
        else "**未检测到面部/语音/步态/人格特征列**"
    )

    if "hamd_score" in df.columns:
        fig_hamd = px.histogram(
            df,
            x="hamd_score",
            nbins=24,
            color_discrete_sequence=["#7c3aed"],
            labels={"hamd_score": "汉密尔顿量表总分", "count": "样本数"},
        )
        fig_hamd.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_hamd, width="stretch")

    if st.session_state.get("multimodal_report") is None:
        st.session_state.multimodal_report = backend.load_csv(backend.MULTIMODAL_REPORT)
    report = st.session_state.multimodal_report
    if report is not None and not report.empty:
        display_report = chinese_multimodal_report(report)
        st.dataframe(display_report, width="stretch", hide_index=True)
        fig_mm = px.bar(
            display_report,
            x="模态",
            y=["曲线下面积", "F1 分数"],
            barmode="group",
            color_discrete_map={"曲线下面积": "#2563eb", "F1 分数": "#f59e0b"},
            labels={"value": "分数", "variable": "指标", "模态": "模态"},
        )
        fig_mm.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_mm, width="stretch")

    if st.button("训练多模态基线（后期融合）", type="primary"):
        with st.spinner("正在训练各单模态模型与概率融合..."):
            st.session_state.multimodal_report = backend.train_multimodal(df)
        st.rerun()

    st.markdown("---")
    st.markdown("#### 改进版：被试级 utterance 基线")
    st.caption("每段 A/V/IMU 特征单独建模，按被试分组重复交叉验证，再对被试概率求平均；结果带均值和波动范围。")
    subj_action_cols = st.columns(2)
    with subj_action_cols[0]:
        if st.button("构建 utterance 特征表", width="stretch"):
            with st.spinner("正在把每个音频/视频/步态片段转成一行..."):
                backend.build_mpdd_utterance_dataset()
                st.cache_data.clear()
            st.rerun()
    with subj_action_cols[1]:
        if st.button("训练被试级基线（重复交叉验证）", type="primary", width="stretch"):
            with st.spinner("正在做分组重复交叉验证并训练推理管线..."):
                st.session_state.multimodal_subject_report = backend.train_mpdd_subject()
            st.rerun()

    if st.session_state.get("multimodal_subject_report") is None:
        st.session_state.multimodal_subject_report = backend.load_csv(
            backend.MULTIMODAL_SUBJECT_REPORT
        )
    subj_report = st.session_state.multimodal_subject_report
    if subj_report is not None and not subj_report.empty:
        display_subject = chinese_subject_report(subj_report)
        st.dataframe(display_subject, width="stretch", hide_index=True)
        fig_subj = px.bar(
            display_subject,
            x="模态",
            y=["曲线下面积均值", "F1 分数均值"],
            barmode="group",
            color_discrete_map={"曲线下面积均值": "#2563eb", "F1 分数均值": "#f59e0b"},
            labels={"value": "分数", "variable": "指标", "模态": "模态"},
        )
        fig_subj.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_subj, width="stretch")
    else:
        st.info("还没有被试级基线结果，先构建 utterance 特征表，再点击训练。")

    st.markdown("---")
    st.markdown("#### 三分类与 PHQ-9 回归")
    advanced_cols = st.columns(2)
    with advanced_cols[0]:
        if st.button("训练三分类 / 回归评估", type="primary", width="stretch"):
            with st.spinner("正在做 label3 三分类和 PHQ-9 回归的被试级交叉验证..."):
                st.session_state.mpdd_advanced_reports = backend.train_mpdd_advanced()
            st.rerun()
    with advanced_cols[1]:
        st.caption("使用同一套 utterance 特征和被试分组，不重复使用测试集。")

    advanced = st.session_state.mpdd_advanced_reports
    if advanced:
        label3 = advanced.get("label3")
        phq9 = advanced.get("phq9")
        if label3 is not None and not label3.empty:
            st.markdown("**label3 三分类**")
            st.dataframe(chinese_advanced_report(label3), width="stretch", hide_index=True)
        if phq9 is not None and not phq9.empty:
            st.markdown("**PHQ-9 回归**")
            st.dataframe(chinese_advanced_report(phq9), width="stretch", hide_index=True)
    else:
        st.info("还没有三分类/回归结果。")

    st.markdown("---")
    st.markdown("#### 单被试推理与测试集导出")
    inf_file = st.file_uploader(
        "上传被试 utterance 特征 CSV（需包含 subject_id），不传则使用本地测试集",
        type=["csv"],
        key="subject_upload",
    )
    inference_df = None
    if inf_file is not None:
        inference_df = pd.read_csv(io.BytesIO(inf_file.getvalue()))
    elif Path(backend.MPDD_UTTERANCE_CSV).exists():
        inference_df = pd.read_csv(backend.MPDD_UTTERANCE_CSV)

    if inference_df is not None and not inference_df.empty:
        subject_col = "subject_id" if "subject_id" in inference_df.columns else inference_df.columns[0]
        subjects = sorted(inference_df[subject_col].astype(str).unique())
        selected = st.selectbox("选择被试", subjects)
        if st.button("运行推理", type="primary"):
            with st.spinner("正在调用被试级推理管线..."):
                try:
                    result = backend.predict_mpdd_subjects(
                        inference_df[inference_df[subject_col].astype(str) == selected]
                    )
                    st.session_state.mpdd_inference_result = result
                    if not result.empty:
                        st.session_state.mpdd_inference_risk = float(
                            result.iloc[0].get("calibrated_prob", result.iloc[0]["fusion_prob"])
                        )
                except Exception as exc:
                    st.warning(f"推理失败：{exc}")
            st.rerun()

    if st.button("导出全部测试集预测"):
        with st.spinner("正在生成 output/mpdd_test_predictions.csv..."):
            st.session_state.mpdd_inference_result = backend.export_mpdd_test_predictions()
        st.rerun()

    inference_result = st.session_state.mpdd_inference_result
    if inference_result is not None and not inference_result.empty:
        st.markdown("**推理结果**")
        risk_map = {"low": "低风险", "medium": "中风险", "high": "高风险"}
        if "risk_level" in inference_result.columns:
            inference_result["风险等级"] = inference_result["risk_level"].map(risk_map).fillna(inference_result["risk_level"])
        st.dataframe(inference_result, width="stretch", hide_index=True)
        if "calibrated_prob" in inference_result.columns:
            st.metric("校准后抑郁风险概率", f"{inference_result.iloc[0]['calibrated_prob']:.0%}")
        csv_bytes = inference_result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
        st.download_button(
            "下载预测结果 CSV",
            data=csv_bytes,
            file_name="mpdd_subject_predictions.csv",
            mime="text/csv",
        )


def render_cmdc_external() -> None:
    st.subheader("CMDC 中文多模态访谈")
    st.caption(
        "用于中文访谈文本、语音和面部信号的外部验证。"
        "该数据是混合成人样本，不是老年专属队列，不进入当前老年风险主评分。"
    )
    summary = backend.cmdc_summary()
    if not summary.get("ready"):
        st.info(
            "CMDC 原始数据保持在外部的加密压缩包中。本项目只生成本地去标识化的"
            "受试者级数值特征表，不复制音频、转写文本或归档密码。"
        )
        with st.form("cmdc_build_form", clear_on_submit=True):
            archive_password = st.text_input(
                "CMDC 归档密码",
                type="password",
                help="仅用于本次本地读取，不写入项目、日志或数据库。",
            )
            submitted = st.form_submit_button(
                "构建 CMDC 受试者级特征表",
                type="primary",
            )
        if submitted:
            if not archive_password:
                st.warning("请输入归档密码")
            else:
                try:
                    with st.spinner("正在读取加密归档并聚合 78 名受试者..."):
                        backend.build_cmdc_dataset(archive_password)
                        st.cache_data.clear()
                except Exception as exc:
                    st.error(f"CMDC 构建失败：{exc}")
                else:
                    st.rerun()
        st.markdown("#### 使用边界")
        st.markdown(
            "- 仅限非商业学术研究，禁止再分发原始文件或转交密码\n"
            "- 原始访谈含声音和转写，必须留在受控目录\n"
            "- 当前数据没有结构化年龄、性别、共病和量表分数\n"
            "- 只能用于外部验证、领域适配或预训练，不能冒充老年主队列"
        )
        return

    metrics = st.columns(5)
    metrics[0].metric("受试者", int(summary.get("rows", 0)))
    metrics[1].metric(
        "健康对照 / MDD",
        f"{int(summary.get('hc_subjects', 0))} / {int(summary.get('mdd_subjects', 0))}",
    )
    metrics[2].metric(
        "全模态完整病例",
        int(summary.get("complete_multimodal_subjects", 0)),
    )
    metrics[3].metric(
        "每人问题数",
        int(summary.get("questions_per_subject", 12)),
    )
    metrics[4].metric("访谈音频", f"{float(summary.get('audio_hours', 0)):.1f} 小时")

    st.warning(
        "适用边界：CMDC 为混合成人样本，不能代表中国老年人；"
        "当前仅作为中文多模态外部验证数据。"
    )

    modality_rows = []
    for key, meta in (summary.get("modalities") or {}).items():
        modality_rows.append(
            {
                "模态": meta.get("label", key),
                "覆盖人数": int(meta.get("subjects", 0)),
                "覆盖率": float(meta.get("coverage", 0)),
                "特征列": int(meta.get("columns", 0)),
            }
        )
    if modality_rows:
        modality_df = pd.DataFrame(modality_rows)
        left, right = st.columns([1.05, 1.95], gap="large")
        with left:
            st.markdown("#### 模态覆盖")
            display_modalities = modality_df.copy()
            display_modalities["覆盖率"] = display_modalities["覆盖率"].map(
                lambda value: f"{value:.1%}"
            )
            st.dataframe(
                display_modalities,
                width="stretch",
                hide_index=True,
            )
        with right:
            coverage_fig = px.bar(
                modality_df,
                x="模态",
                y="覆盖人数",
                color="模态",
                text="覆盖人数",
                color_discrete_sequence=["#2563eb", "#0f766e", "#f59e0b", "#dc2626"],
                labels={"模态": "", "覆盖人数": "受试者人数"},
            )
            coverage_fig.update_layout(
                height=285,
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False,
            )
            st.plotly_chart(coverage_fig, width="stretch")

    st.markdown("#### 受试者级外部验证")
    st.caption(
        "严格按 subject_id 聚合；每位受试者只出现一次；"
        "重复分层 K 折训练和验证，避免把同一人的 12 句话拆到两边。"
    )
    report, benchmark = backend.cmdc_report()
    action_cols = st.columns([1.15, 2.85], gap="large")
    with action_cols[0]:
        repeats = st.slider(
            "重复次数",
            min_value=3,
            max_value=30,
            value=10,
            key="cmdc_repeats",
        )
        if st.button(
            "训练 CMDC 外部验证模型",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("正在执行受试者级重复交叉验证..."):
                    benchmark = backend.train_cmdc_models(repeats=repeats)
                    report, benchmark = backend.cmdc_report()
                    st.cache_data.clear()
            except Exception as exc:
                st.error(f"CMDC 训练失败：{exc}")
            else:
                st.rerun()
    with action_cols[1]:
        primary = benchmark.get("primary_result") or {}
        if primary:
            primary_cols = st.columns(4)
            primary_cols[0].metric(
                "主模型 AUC",
                f"{float(primary.get('auc_mean', 0)):.3f}",
            )
            primary_cols[1].metric(
                "PR-AUC",
                f"{float(primary.get('pr_auc_mean', 0)):.3f}",
            )
            primary_cols[2].metric(
                "灵敏度",
                f"{float(primary.get('sensitivity_mean', 0)):.3f}",
            )
            primary_cols[3].metric(
                "特异度",
                f"{float(primary.get('specificity_mean', 0)):.3f}",
            )
            st.caption(
                str(benchmark.get("evidence_boundary") or "")
            )
        else:
            st.info("尚未训练 CMDC 外部验证模型。")

    if report is not None and not report.empty:
        variant_labels = {
            "text": "文本",
            "acoustic": "openSMILE 声学",
            "audio": "语音融合",
            "visual": "面部视觉",
            "text+audio_deep+acoustic": "文本＋语音",
            "text+visual+audio_deep+acoustic": "全模态完整病例",
        }
        display_report = report.copy()
        display_report["方案"] = (
            display_report["variant"].map(variant_labels).fillna(display_report["variant"])
        )
        display_report = display_report.rename(
            columns={
                "subjects": "受试者",
                "hc_subjects": "HC",
                "mdd_subjects": "MDD",
                "features": "特征数",
                "auc_mean": "AUC",
                "auc_std": "AUC 波动",
                "pr_auc_mean": "PR-AUC",
                "balanced_accuracy_mean": "平衡准确率",
                "sensitivity_mean": "灵敏度",
                "specificity_mean": "特异度",
                "f1_mean": "F1",
                "brier_mean": "Brier",
            }
        )
        visible_columns = [
            column
            for column in [
                "方案",
                "受试者",
                "HC",
                "MDD",
                "特征数",
                "AUC",
                "AUC 波动",
                "PR-AUC",
                "平衡准确率",
                "灵敏度",
                "特异度",
                "F1",
                "Brier",
            ]
            if column in display_report.columns
        ]
        st.dataframe(
            display_report[visible_columns],
            width="stretch",
            hide_index=True,
        )
        plot_df = display_report.dropna(subset=["AUC"])
        auc_fig = px.bar(
            plot_df,
            x="方案",
            y=["AUC", "PR-AUC"],
            barmode="group",
            color_discrete_map={"AUC": "#2563eb", "PR-AUC": "#f59e0b"},
            labels={"value": "分数", "variable": "指标", "方案": ""},
        )
        auc_fig.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=25, b=10),
            legend_title_text="",
        )
        st.plotly_chart(auc_fig, width="stretch")
        st.download_button(
            "下载 CMDC 评估报告",
            data=report.to_csv(index=False, encoding="utf-8-sig").encode("utf-8"),
            file_name="cmdc_external_validation_report.csv",
            mime="text/csv",
        )

    st.markdown("#### 稳健性与多模态提升验证")
    advanced = summary.get("advanced") or {}
    if advanced.get("ready"):
        bootstrap = advanced.get("primary_bootstrap_ci") or {}
        auc_ci = bootstrap.get("auc") or {}
        adv_metric_cols = st.columns(4)
        adv_metric_cols[0].metric(
            "AUC 95% CI",
            f"{float(auc_ci.get('ci_low', 0)):.3f}-"
            f"{float(auc_ci.get('ci_high', 0)):.3f}",
        )
        adv_metric_cols[1].metric(
            "最佳嵌套模型",
            str(advanced.get("best_nested_model") or "-"),
        )
        missing_compare = advanced.get("missing_modality_comparison") or {}
        adv_metric_cols[2].metric(
            "动态融合 AUC 差",
            f"{float(missing_compare.get('auc_difference', 0)):+.3f}",
        )
        adv_metric_cols[3].metric(
            "Bootstrap",
            f"{int(advanced.get('bootstrap_iterations', 0))} 次",
        )

        advance_tabs = st.tabs(
            ["同受试者配对", "缺失模态", "模型比较", "校准与决策"]
        )
        with advance_tabs[0]:
            paired = backend.load_csv(backend.CMDC_PAIRED_REPORT)
            if paired is not None and not paired.empty:
                display = paired.rename(
                    columns={
                        "group": "分组",
                        "subjects": "受试者",
                        "model_a": "比较模型",
                        "model_b": "基准模型",
                        "auc_a": "AUC A",
                        "auc_b": "AUC B",
                        "auc_difference": "AUC 差值",
                        "ci_low": "差值 CI 下限",
                        "ci_high": "差值 CI 上限",
                        "positive_bootstrap_rate": "Bootstrap 正向率",
                        "paired_repeat_mean_auc_difference": "重复折平均差值",
                        "paired_t_p_value": "配对 p 值",
                    }
                )
                st.dataframe(display, width="stretch", hide_index=True)
        with advance_tabs[1]:
            missing = backend.load_csv(backend.CMDC_MISSING_MODALITY_REPORT)
            if missing is not None and not missing.empty:
                st.dataframe(missing, width="stretch", hide_index=True)
        with advance_tabs[2]:
            model_compare = backend.load_csv(backend.CMDC_MODEL_COMPARISON_REPORT)
            if model_compare is not None and not model_compare.empty:
                st.dataframe(model_compare, width="stretch", hide_index=True)
        with advance_tabs[3]:
            calibration = backend.load_csv(backend.CMDC_CALIBRATION_CURVE)
            decision_curve = backend.load_csv(backend.CMDC_DECISION_CURVE)
            if calibration is not None and not calibration.empty:
                cal_fig = px.line(
                    calibration,
                    x="predicted_frequency",
                    y="observed_frequency",
                    markers=True,
                    labels={
                        "predicted_frequency": "预测概率",
                        "observed_frequency": "实际阳性率",
                    },
                )
                cal_fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        line=dict(color="#dc2626", dash="dash"),
                        name="理想校准线",
                    )
                )
                cal_fig.update_layout(
                    height=320,
                    margin=dict(l=10, r=10, t=25, b=10),
                )
                st.plotly_chart(cal_fig, width="stretch")
            if decision_curve is not None and not decision_curve.empty:
                dca_fig = px.line(
                    decision_curve,
                    x="threshold",
                    y=[
                        "net_benefit",
                        "treat_all_net_benefit",
                        "treat_none_net_benefit",
                    ],
                    labels={"threshold": "阈值", "value": "净获益", "variable": "策略"},
                )
                dca_fig.update_layout(
                    height=320,
                    margin=dict(l=10, r=10, t=25, b=10),
                )
                st.plotly_chart(dca_fig, width="stretch")
        st.caption(
            "这些分析用于提高结果可信度和选择一个可部署的简单模型；"
            "重复交叉验证与 Bootstrap 仍不能替代独立外部测试集。"
        )
    else:
        st.info("尚未运行同受试者配对、缺失模态、模型比较和校准分析。")

    if st.button(
        "运行稳健性与多模态提升验证",
        type="primary",
        width="stretch",
    ):
        try:
            with st.spinner("正在运行配对 Bootstrap、缺失模态和嵌套模型比较..."):
                backend.run_cmdc_advanced(
                    repeats=5,
                    model_repeats=3,
                    bootstrap_iterations=500,
                )
                st.cache_data.clear()
        except Exception as exc:
            st.error(f"稳健性分析失败：{exc}")
        else:
            st.rerun()

    st.markdown("#### 锁定留出集验证")
    holdout = summary.get("holdout") or {}
    st.caption(
        "固定随机种子按受试者分层划分 80% 训练、20% 测试；"
        "分类阈值只在训练集内选择，测试集不参与调参。"
    )
    if holdout.get("ready"):
        holdout_metrics = holdout.get("holdout_metrics") or {}
        locked = holdout_metrics.get("locked_threshold") or {}
        default = holdout_metrics.get("default_threshold") or {}
        locked_sensitivity_ci = locked.get("sensitivity_exact_ci") or {}
        locked_specificity_ci = locked.get("specificity_exact_ci") or {}
        holdout_ci = (holdout.get("holdout_bootstrap_ci") or {}).get("auc") or {}
        holdout_cols = st.columns(4)
        holdout_cols[0].metric(
            "留出 AUC",
            f"{float(holdout_metrics.get('auc', 0)):.3f}",
            help=(
                f"95% CI {float(holdout_ci.get('ci_low', 0)):.3f}-"
                f"{float(holdout_ci.get('ci_high', 0)):.3f}"
            ),
        )
        holdout_cols[1].metric(
            "锁定阈值",
            f"{float(holdout.get('locked_threshold', 0)):.3f}",
        )
        holdout_cols[2].metric(
            "留出灵敏度",
            f"{float(locked.get('sensitivity', 0)):.3f}",
            help=(
                "精确 95% CI "
                f"{float(locked_sensitivity_ci.get('ci_low', 0)):.3f}-"
                f"{float(locked_sensitivity_ci.get('ci_high', 0)):.3f}"
            ),
        )
        holdout_cols[3].metric(
            "留出特异度",
            f"{float(locked.get('specificity', 0)):.3f}",
            help=(
                "精确 95% CI "
                f"{float(locked_specificity_ci.get('ci_low', 0)):.3f}-"
                f"{float(locked_specificity_ci.get('ci_high', 0)):.3f}"
            ),
        )
        boundary_cols = st.columns(3)
        boundary_cols[0].metric(
            "训练 / 测试受试者",
            f"{int(holdout.get('train_subjects', 0))} / {int(holdout.get('test_subjects', 0))}",
        )
        boundary_cols[1].metric(
            "测试集 HC / MDD",
            f"{int(holdout.get('test_hc', 0))} / {int(holdout.get('test_mdd', 0))}",
        )
        boundary_cols[2].metric(
            "默认阈值准确率",
            (
                f"{float(default.get('balanced_accuracy', 0)):.3f}"
                if default
                else "-"
            ),
            help="默认阈值 0.50 下的平衡准确率，仅用于对照。",
        )
        holdout_table = pd.DataFrame(
            [
                {
                    "评估集": "锁定测试集",
                    "受试者": int(holdout.get("test_subjects", 0)),
                    "AUC": round(float(holdout_metrics.get("auc", 0)), 4),
                    "PR-AUC": round(float(holdout_metrics.get("pr_auc", 0)), 4),
                    "灵敏度": round(float(locked.get("sensitivity", 0)), 4),
                    "灵敏度精确 95% CI": (
                        f"{float(locked_sensitivity_ci.get('ci_low', 0)):.3f}-"
                        f"{float(locked_sensitivity_ci.get('ci_high', 0)):.3f}"
                    ),
                    "特异度": round(float(locked.get("specificity", 0)), 4),
                    "特异度精确 95% CI": (
                        f"{float(locked_specificity_ci.get('ci_low', 0)):.3f}-"
                        f"{float(locked_specificity_ci.get('ci_high', 0)):.3f}"
                    ),
                    "平衡准确率": round(
                        float(locked.get("balanced_accuracy", 0)),
                        4,
                    ),
                    "Brier": round(float(holdout_metrics.get("brier", 0)), 4),
                }
            ]
        )
        st.dataframe(holdout_table, width="stretch", hide_index=True)
        st.warning(str(holdout.get("evidence_boundary") or ""))
        st.warning(str(holdout.get("small_sample_warning") or ""))
    else:
        st.info(
            "尚未运行锁定留出验证。该结果只检验固定流程，"
            "不能替代老年多中心外部测试。"
        )

    if st.button(
        "运行锁定留出验证",
        type="primary",
        width="stretch",
        key="cmdc_holdout_run",
    ):
        try:
            with st.spinner("正在固定受试者划分、训练阈值并一次性评估测试集..."):
                backend.run_cmdc_holdout(
                    repeats=10,
                    bootstrap_iterations=1000,
                )
                st.cache_data.clear()
        except Exception as exc:
            st.error(f"锁定留出验证失败：{exc}")
        else:
            st.rerun()

    with st.expander("重新构建特征表", expanded=False):
        st.caption("重建只替换本地数值特征和评估报告，不会把原始文件复制进项目。")
        with st.form("cmdc_rebuild_form", clear_on_submit=True):
            rebuild_password = st.text_input(
                "CMDC 归档密码",
                type="password",
                key="cmdc_rebuild_password",
            )
            rebuild = st.form_submit_button("重新构建")
        if rebuild:
            if not rebuild_password:
                st.warning("请输入归档密码")
            else:
                try:
                    with st.spinner("正在重新读取 CMDC 加密归档..."):
                        backend.build_cmdc_dataset(rebuild_password)
                        st.cache_data.clear()
                except Exception as exc:
                    st.error(f"CMDC 构建失败：{exc}")
                else:
                    st.rerun()

    st.markdown("#### 引用与合规")
    st.markdown(
        "论文：`10.1109/TAFFC.2022.3181210`  \n"
        "数据集：`10.21227/jng1-rn06`  \n"
        "仅限非商业学术研究；禁止再分发原始数据、公开受试者音频或转交归档密码。"
    )


def render_mpdd() -> None:
    st.subheader("多模态数据")
    st.caption("两个数据域独立展示、独立评估，不把不同年龄范围和采集协议的数据混成一个训练集。")
    mpdd_tab, cmdc_tab = st.tabs(
        ["MPDD｜老年多模态", "CMDC｜中文外部验证"]
    )
    with mpdd_tab:
        render_mpdd_dataset()
    with cmdc_tab:
        render_cmdc_external()


def render_diy() -> None:
    st.subheader("数据平衡处理结果")
    diy_dir = PROJECT_ROOT.parent / "DIY"
    if not diy_dir.exists():
        st.info("还没有 DIY 结果文件夹，请先运行数据处理。")
        return

    st.markdown("#### 文件位置")
    st.code(str(diy_dir))

    balanced_path = diy_dir / "charls_smote_tomek_train.csv"
    if balanced_path.exists():
        balanced = pd.read_csv(balanced_path)
        st.markdown("#### 平衡后训练数据")
        cols = st.columns(4)
        cols[0].metric("样本量", f"{len(balanced):,}")
        cols[1].metric("特征数", balanced.shape[1] - 1)
        cols[2].metric("正负比例", "1 : 1")
        cols[3].metric("处理方法", "SMOTE-Tomek")
    else:
        st.info("还没有 SMOTE-Tomek 平衡训练数据。")

    comparison_path = diy_dir / "imbalance_comparison.csv"
    if comparison_path.exists():
        st.markdown("#### 失衡处理方法对比")
        comparison = pd.read_csv(comparison_path)
        st.dataframe(chinese_imbalance_report(comparison), width="stretch", hide_index=True)

    paper_path = diy_dir / "imbalance_paper_report.csv"
    if paper_path.exists():
        st.markdown("#### 论文完整指标")
        paper = pd.read_csv(paper_path)
        st.dataframe(paper, width="stretch", hide_index=True)

    model_path = diy_dir / "tabular_model_report.csv"
    if model_path.exists():
        st.markdown("#### 模型结果")
        report = pd.read_csv(model_path)
        st.dataframe(chinese_tabular_report(report), width="stretch", hide_index=True)

    importance_path = diy_dir / "feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        importance["model"] = importance["model"].map(MODEL_LABELS).fillna(importance["model"])
        top = importance.sort_values("importance", ascending=False).head(20)
        fig = px.bar(
            top,
            x="importance",
            y="feature",
            color="model",
            orientation="h",
            labels={"importance": "重要性", "feature": "特征", "model": "模型"},
            color_discrete_sequence=["#2563eb", "#f59e0b", "#16a34a"],
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")


def render_records() -> None:
    st.subheader("用户记录")
    st.caption("查询用户 → 点击查看详情 → 查看评估趋势；新建、编辑、删除分开操作。")
    all_users = backend.list_users()
    st.session_state.user_list = all_users
    all_by_id = {u["id"]: u for u in all_users}
    risk_cn = {"low": "低风险", "medium": "中风险", "high": "高风险"}
    risk_short = {"low": "低", "medium": "中", "high": "高"}

    if st.session_state.selected_user_id not in all_by_id:
        st.session_state.selected_user_id = st.session_state.active_user_id
    if st.session_state.selected_user_id not in all_by_id and all_users:
        st.session_state.selected_user_id = all_users[0]["id"]

    histories_by_user = {
        u["id"]: backend.list_assessments(u["id"], limit=5000) for u in all_users
    }
    risk_counts = {"low": 0, "medium": 0, "high": 0, "none": 0}
    total_assessments = 0
    for user in all_users:
        history = histories_by_user[user["id"]]
        total_assessments += len(history)
        latest = history[0] if history else None
        risk_counts[latest.get("risk_level") if latest else "none"] += 1

    st.markdown("#### 用户风险总览")
    overview_cols = st.columns(5)
    overview_cols[0].metric("用户总数", len(all_users))
    overview_cols[1].metric("高风险", risk_counts["high"])
    overview_cols[2].metric("中风险", risk_counts["medium"])
    overview_cols[3].metric("低风险", risk_counts["low"])
    overview_cols[4].metric("评估次数", total_assessments)
    if total_assessments:
        dist_df = pd.DataFrame(
            {
                "风险等级": ["高风险", "中风险", "低风险"],
                "人数": [risk_counts["high"], risk_counts["medium"], risk_counts["low"]],
            }
        )
        dist_fig = px.bar(
            dist_df,
            x="风险等级",
            y="人数",
            color="风险等级",
            color_discrete_map={"高风险": "#dc2626", "中风险": "#f59e0b", "低风险": "#16a34a"},
            text="人数",
        )
        dist_fig.update_layout(height=240, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
        st.plotly_chart(dist_fig, width="stretch")

    left, right = st.columns([0.34, 0.66], gap="large")

    with left:
        st.markdown("#### 查询用户")
        search = st.text_input("按姓名 / 编号 / 备注查询", key="user_query")
        search = (search or "").strip()
        if search:
            q = search.lower()
            users = [
                u
                for u in all_users
                if q in str(u.get("name") or "").lower()
                or q in str(u.get("id") or "").lower()
                or q in str(u.get("note") or "").lower()
            ]
        else:
            users = all_users
        user_by_id = {u["id"]: u for u in users}
        st.caption(f"共 {len(users)} 位用户")
        st.markdown("#### 用户列表")
        if not users:
            st.info("没有匹配的用户，可调整查询条件或新建用户。")
        for user in users:
            uid = user["id"]
            history = histories_by_user.get(uid, [])
            latest = history[0] if history else None
            risk = latest.get("risk_level") if latest else None
            risk_text = risk_cn.get(risk, "暂无评估")
            risk_color = {"low": "#16a34a", "medium": "#f59e0b", "high": "#dc2626"}.get(
                risk, "#64748b"
            )
            is_active = uid == st.session_state.active_user_id
            gender = "女" if user.get("gender") == "female" else "男"
            name = user.get("name", "未命名")
            st.markdown(
                f'<div class="user-card{" active" if is_active else ""}">'
                f'<div class="user-avatar">{name[:1]}</div>'
                f'<div class="user-card-body">'
                f'<div class="user-card-name">{name}</div>'
                f'<div style="color:var(--muted);font-size:0.85rem;">'
                f"{user.get('age', '-')} 岁 · {gender} · 共病 {user.get('comorbidity_count', '-')} 项 · "
                f"评估 {len(history)} 次</div>"
                f'<div style="color:{risk_color};font-weight:650;">'
                f'<span class="risk-dot" style="background:{risk_color};"></span>'
                f"最近评估：{risk_text}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
            card_cols = st.columns(2)
            with card_cols[0]:
                if st.button("查看详情", key=f"view_user_{uid}", width="stretch"):
                    st.session_state.selected_user_id = uid
                    st.rerun()
            with card_cols[1]:
                if st.button("设为当前", key=f"set_user_{uid}", width="stretch"):
                    st.session_state.active_user_id = uid
                    st.session_state.selected_user_id = uid
                    st.rerun()

        with st.expander("新建用户", expanded=not all_users):
            with st.form("new_user_form"):
                new_name = st.text_input("新用户姓名 / 编号", key="new_user_name_record")
                new_age = st.number_input("新用户年龄", 60, 100, 70, key="new_user_age")
                new_gender = st.selectbox(
                    "新用户性别",
                    ["女", "男"],
                    key="new_user_gender",
                )
                new_marital = st.selectbox(
                    "新用户婚姻状况",
                    ["有伴侣", "独居 / 丧偶 / 离异"],
                    key="new_user_marital",
                )
                new_comorbidity = st.number_input(
                    "新用户慢性病数量", 0, 10, 1, key="new_user_comorbidity"
                )
                new_health = st.select_slider(
                    "新用户自评健康",
                    [1, 2, 3, 4, 5],
                    key="new_user_health",
                )
                new_followup = st.selectbox(
                    "随访 / 独立诊断标签",
                    ["未知", "无抑郁", "有抑郁"],
                    key="new_user_followup",
                )
                new_note = st.text_area("新用户备注（可选）", key="new_user_note")
                submitted_new = st.form_submit_button("创建用户", type="primary")
            if submitted_new:
                new_name = (new_name or "").strip()
                if not new_name:
                    st.warning("姓名 / 编号不能为空")
                else:
                    created = backend.create_user(
                        name=new_name,
                        age=int(new_age),
                        gender="female" if new_gender == "女" else "male",
                        marital_status="partner" if new_marital == "有伴侣" else "single",
                        comorbidity_count=int(new_comorbidity),
                        self_rated_health=int(new_health),
                        followup_label=(
                            None
                            if new_followup == "未知"
                            else 1
                            if new_followup == "有抑郁"
                            else 0
                        ),
                        note=(new_note or "").strip() or None,
                    )
                    st.session_state.selected_user_id = created["id"]
                    st.session_state.active_user_id = created["id"]
                    st.session_state.profile_loaded_user_id = created["id"]
                    st.session_state.profile_loaded_updated_at = created.get("updated_at")
                    st.session_state.user_list = backend.list_users()
                    st.rerun()

        edit_id = st.session_state.selected_user_id
        with st.expander("编辑所选用户", expanded=bool(edit_id and edit_id in all_by_id)):
            if not edit_id or edit_id not in all_by_id:
                st.caption("先在左侧选择一位用户再编辑。")
            else:
                default = all_by_id[edit_id]
                with st.form("edit_user_form"):
                    edit_name = st.text_input(
                        "姓名 / 编号", value=default.get("name", ""), key="edit_user_name"
                    )
                    edit_age = st.number_input(
                        "年龄", 60, 100, int(default.get("age") or 70), key="edit_user_age"
                    )
                    edit_gender = st.selectbox(
                        "性别",
                        ["女", "男"],
                        index=0 if default.get("gender") != "male" else 1,
                        key="edit_user_gender",
                    )
                    edit_marital = st.selectbox(
                        "婚姻状况",
                        ["有伴侣", "独居 / 丧偶 / 离异"],
                        index=0 if default.get("marital_status") != "single" else 1,
                        key="edit_user_marital",
                    )
                    edit_comorbidity = st.number_input(
                        "慢性病数量",
                        0,
                        10,
                        int(default.get("comorbidity_count") or 1),
                        key="edit_user_comorbidity",
                    )
                    edit_health = st.select_slider(
                        "自评健康",
                        [1, 2, 3, 4, 5],
                        value=int(default.get("self_rated_health") or 3),
                        key="edit_user_health",
                    )
                    edit_followup = st.selectbox(
                        "随访 / 独立诊断标签",
                        ["未知", "无抑郁", "有抑郁"],
                        index={
                            None: 0,
                            1: 2,
                            0: 1,
                        }.get(default.get("followup_label"), 0),
                        key="edit_user_followup",
                    )
                    edit_note = st.text_area(
                        "备注（可选）", value=default.get("note", ""), key="edit_user_note"
                    )
                    submitted_edit = st.form_submit_button("保存修改", type="primary")
                if submitted_edit:
                    edit_name = (edit_name or "").strip()
                    if not edit_name:
                        st.warning("姓名 / 编号不能为空")
                    else:
                        backend.update_user(
                            edit_id,
                            name=edit_name,
                            age=int(edit_age),
                            gender="female" if edit_gender == "女" else "male",
                            marital_status="partner" if edit_marital == "有伴侣" else "single",
                            comorbidity_count=int(edit_comorbidity),
                            self_rated_health=int(edit_health),
                            followup_label=(
                                None
                                if edit_followup == "未知"
                                else 1
                                if edit_followup == "有抑郁"
                                else 0
                            ),
                            note=(edit_note or "").strip() or None,
                        )
                        st.session_state.user_list = backend.list_users()
                        st.rerun()

        with st.expander("数据备份 / 恢复", expanded=False):
            backup = backend.export_users()
            backup_json = json.dumps(backup, ensure_ascii=False, indent=2)
            st.download_button(
                "导出全部用户数据",
                data=backup_json.encode("utf-8"),
                file_name="elderly_depression_users_backup.json",
                mime="application/json",
                width="stretch",
            )
            uploaded = st.file_uploader("导入备份 JSON", type=["json"], key="user_backup_upload")
            replace_mode = st.checkbox("清空现有数据后恢复（谨慎）", key="backup_replace_mode")
            if uploaded is not None:
                try:
                    payload = json.loads(uploaded.getvalue().decode("utf-8"))
                    if st.button("导入备份", type="primary", key="import_backup_btn"):
                        mode = "replace" if replace_mode else "merge"
                        result = backend.import_users(payload, mode=mode)
                        st.success(
                            f"已导入 {result.get('users', 0)} 位用户、"
                            f"{result.get('assessments', 0)} 条评估记录。"
                        )
                        st.session_state.user_list = backend.list_users()
                        st.rerun()
                except Exception as exc:
                    st.error(f"备份文件解析失败：{exc}")

    with right:
        selected_id = st.session_state.selected_user_id
        if not selected_id or selected_id not in all_by_id:
            st.info("点击左侧用户卡片查看档案、评估历史和风险趋势。")
            return

        user = all_by_id[selected_id]
        all_history = histories_by_user.get(selected_id, [])
        gender = "女" if user.get("gender") == "female" else "男"
        marital = (
            "有伴侣" if user.get("marital_status") == "partner" else "独居 / 丧偶 / 离异"
        )
        health_labels = {1: "很好", 2: "好", 3: "一般", 4: "较差", 5: "很差"}

        name = user.get("name", "未命名")
        st.markdown(
            f'<div style="display:flex;gap:1rem;align-items:center;margin-bottom:0.5rem;">'
            f'<div class="user-avatar" style="width:56px;height:56px;font-size:1.35rem;">{name[:1]}</div>'
            f'<div>'
            f'<span style="font-size:1.35rem;font-weight:700;">{name}</span><br>'
            f'<span class="tag-pill">{user.get("age", "-")} 岁</span>'
            f'<span class="tag-pill">{gender}</span>'
            f'<span class="tag-pill">共病 {user.get("comorbidity_count", "-")} 项</span>'
            f'<span class="tag-pill">评估 {len(all_history)} 次</span>'
            f"</div></div>",
            unsafe_allow_html=True,
        )
        if all_history:
            probs = [float(r.get("depression_probability") or 0) for r in all_history]
            high_count = sum(1 for r in all_history if r.get("risk_level") == "high")
            metric_cols = st.columns(4)
            metric_cols[0].metric(
                "最近风险",
                risk_short.get(all_history[0].get("risk_level"), "暂无"),
            )
            metric_cols[1].metric("最近概率", f"{probs[0]:.0%}")
            metric_cols[2].metric("平均概率", f"{sum(probs) / len(probs):.0%}")
            metric_cols[3].metric("高风险次数", high_count)
        else:
            metric_cols = st.columns(4)
            metric_cols[0].metric("评估次数", 0)
            metric_cols[1].metric("最近概率", "暂无")
            metric_cols[2].metric("平均概率", "暂无")
            metric_cols[3].metric("高风险次数", 0)
        st.caption(
            f"婚姻：{marital} · 自评健康：{health_labels.get(int(user.get('self_rated_health') or 3), '-')}"
            + (f" · 备注：{user.get('note')}" if user.get("note") else "")
        )
        followup_cn = {1: "有抑郁", 0: "无抑郁", None: "未知"}.get(
            user.get("followup_label"), "未知"
        )
        st.caption(f"随访 / 独立诊断标签：{followup_cn}")

        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("设为当前用户并去评估", type="primary", width="stretch"):
                st.session_state.active_user_id = selected_id
                navigate_to("风险评估")
        with action_cols[1]:
            if st.button("删除这位用户", width="stretch"):
                st.session_state.pending_delete_user = selected_id
        if st.session_state.get("pending_delete_user") == selected_id:
            st.warning("确认删除？该用户档案和全部评估记录都会被移除，无法恢复。")
            if st.button("确认删除", type="primary"):
                backend.delete_user(selected_id)
                if st.session_state.active_user_id == selected_id:
                    st.session_state.active_user_id = None
                    st.session_state.profile_loaded_user_id = None
                st.session_state.selected_user_id = None
                st.session_state.pending_delete_user = None
                st.rerun()

        single_export = {
            "exported_at": datetime.now().isoformat(),
            "user": user,
            "assessments": all_history,
        }
        st.download_button(
            "导出该用户档案与评估",
            data=json.dumps(single_export, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"user_{selected_id}_backup.json",
            mime="application/json",
            width="stretch",
            key=f"export_user_{selected_id}",
        )

        st.markdown("#### 查询评估记录")
        filter_cols = st.columns(2)
        risk_filter = filter_cols[0].selectbox(
            "风险等级",
            ["全部", "低", "中", "高"],
            key=f"risk_filter_{selected_id}",
        )
        scale_filter = filter_cols[1].selectbox(
            "量表",
            ["全部"] + list(SCALE_LABELS.keys()),
            format_func=lambda x: "全部量表" if x == "全部" else SCALE_LABELS[x],
            key=f"scale_filter_{selected_id}",
        )
        risk_code = {"低": "low", "中": "medium", "高": "high"}.get(risk_filter)
        history = [
            r
            for r in all_history
            if (risk_code is None or r.get("risk_level") == risk_code)
            and (scale_filter == "全部" or r.get("scale_name") == scale_filter)
        ]
        if not history:
            st.info("这位用户还没有符合筛选条件的评估记录。")
            return

        latest = history[0]
        latest_color = {"low": "#16a34a", "medium": "#f59e0b", "high": "#dc2626"}.get(
            latest.get("risk_level"), "#64748b"
        )
        st.markdown("#### 最近一次评估")
        latest_cols = st.columns([0.40, 0.60])
        with latest_cols[0]:
            st.plotly_chart(
                risk_gauge(float(latest.get("depression_probability") or 0)),
                width="stretch",
            )
        with latest_cols[1]:
            st.markdown(
                f'<span class="risk-badge" style="color:{latest_color};">'
                f"{risk_cn.get(latest.get('risk_level'), '暂无')}</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"量表：{SCALE_LABELS.get(latest.get('scale_name'), latest.get('scale_name'))} · "
                f"量表分 {latest.get('scale_score')}"
            )
            cues = latest.get("language_cues") or []
            if cues:
                cue_text = "、".join(
                    c["cue"] if isinstance(c, dict) else str(c) for c in cues
                )
                st.markdown(f"**语言线索：** {cue_text}")

        trend = pd.DataFrame(
            [
                {
                    "记录时间": r.get("created_at", ""),
                    "抑郁概率": float(r.get("depression_probability") or 0),
                    "风险等级": r.get("risk_level"),
                    "量表分": r.get("scale_score"),
                    "量表": r.get("scale_name"),
                    "id": r.get("id"),
                }
                for r in history
            ]
        )
        trend["时间"] = pd.to_datetime(trend["记录时间"], errors="coerce")
        trend = trend.sort_values("时间").reset_index(drop=True)

        st.markdown("#### 抑郁风险趋势")
        point_colors = [{"low": "#16a34a", "medium": "#f59e0b", "high": "#dc2626"}.get(
            r, "#64748b"
        ) for r in trend["风险等级"]]
        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=trend["时间"],
                y=trend["抑郁概率"],
                mode="lines+markers",
                line=dict(color="#2563eb", width=2.5),
                marker=dict(size=11, color=point_colors, line=dict(color="#ffffff", width=1)),
                hovertemplate="%{y:.0%}<extra></extra>",
            )
        )
        fig_trend.add_hline(y=0.4, line_dash="dash", line_color="#f59e0b", annotation_text="中风险线")
        fig_trend.add_hline(y=0.7, line_dash="dash", line_color="#dc2626", annotation_text="高风险线")
        fig_trend.update_layout(
            height=300,
            yaxis_tickformat=".0%",
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_trend, width="stretch")

        hist = trend.rename(
            columns={
                "时间": "记录时间",
                "量表分": "量表分",
            }
        )
        hist["记录时间"] = hist["记录时间"].dt.strftime("%m-%d %H:%M")
        hist["风险等级"] = hist["风险等级"].map(risk_short).fillna(hist["风险等级"])
        display_cols = ["记录时间", "抑郁概率", "风险等级", "量表分", "量表"]
        st.markdown("#### 评估记录")
        st.dataframe(hist[display_cols], width="stretch", hide_index=True)

        csv_bytes = hist[display_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
        st.download_button(
            "导出评估历史 CSV",
            data=csv_bytes,
            file_name=f"user_{selected_id}_history.csv",
            mime="text/csv",
        )

        record_id = st.selectbox(
            "查看单次评估详情",
            trend["id"].tolist(),
            format_func=lambda rid: next(
                (r["记录时间"] for r in hist.to_dict("records") if r["id"] == rid),
                rid,
            ),
        )
        detail = next((r for r in history if r["id"] == record_id), None)
        if detail:
            detail_cols = st.columns(4)
            detail_cols[0].metric(
                "抑郁概率", f"{float(detail.get('depression_probability') or 0):.0%}"
            )
            detail_cols[1].metric(
                "风险等级", risk_short.get(detail.get("risk_level"), detail.get("risk_level"))
            )
            detail_cols[2].metric("量表分", detail.get("scale_score"))
            detail_cols[3].metric(
                "融合方式", "已学习" if detail.get("fusion_mode") == "learned" else "文献先验"
            )
            st.json({k: v for k, v in detail.items() if k not in {"user_id", "id"}})


def data_health_markdown(report: dict) -> str:
    lines = [
        "# 数据健康体检报告",
        "",
        f"生成时间：{report.get('generated_at', '')}",
        "",
        f"总体状态：{'健康' if report.get('healthy') else '有异常'}",
        f"阻断问题：{report.get('issue_count', 0)}",
        f"风险提示：{report.get('warning_count', 0)}",
        "",
    ]
    for item in report.get("issues", []) + report.get("warnings", []):
        lines.append(f"- [{item.get('source', '')}] {item.get('message', '')}")
    sources = report.get("sources", {})
    charls = sources.get("charls", {})
    mpdd = sources.get("mpdd", {})
    cmdc = sources.get("cmdc", {})
    syn = sources.get("synthetic", {})
    users = sources.get("users", {})
    lines += [
        "",
        "## CHARLS",
        "",
        f"- 就绪：{charls.get('ready', False)}",
        f"- 样本量：{charls.get('rows', 0)}",
        f"- 缺失率：{float(charls.get('missing_rate', 0)):.1%}",
        f"- 阳性率：{float(charls.get('positive_rate') or 0):.1%}",
        "",
        "## MPDD",
        "",
        f"- 就绪：{mpdd.get('ready', False)}",
        f"- 样本量：{mpdd.get('rows', 0)}",
        f"- 缺失率：{float(mpdd.get('missing_rate', 0)):.1%}",
        f"- 标签：{mpdd.get('has_label', False)}",
        "",
        "## CMDC 中文外部验证",
        "",
        f"- 就绪：{cmdc.get('ready', False)}",
        f"- 受试者：{cmdc.get('rows', 0)}",
        f"- HC / MDD：{cmdc.get('hc_subjects', 0)} / {cmdc.get('mdd_subjects', 0)}",
        f"- 全模态完整病例：{cmdc.get('complete_multimodal_subjects', 0)}",
        f"- 适用人群：{cmdc.get('age_scope_warning', '混合成人，非老年专属')}",
        f"- 许可：{cmdc.get('license_boundary', '')}",
        "",
        "## 模拟数据",
        "",
        f"- 就绪：{syn.get('ready', False)}",
        f"- 对话模拟组验证：{syn.get('dialogue_feasible', False)}",
        f"- 表格模拟组样本：{syn.get('tabular_rows', 0)}",
        f"- 片段级增强：{syn.get('augmentation_ready', False)}",
        "",
        "## 用户记录",
        "",
        f"- 用户数：{users.get('users', 0)}",
        f"- 评估数：{users.get('assessments', 0)}",
        f"- 缺失关键字段：{sum((users.get('missing_fields') or {}).values())}",
        "",
        "---",
        "",
        "本报告仅用于科研数据质量检查，不替代临床诊断。",
    ]
    return "\n".join(lines)


def render_data_health() -> None:
    st.subheader("数据健康体检")
    st.caption(
        "自动检查 CHARLS、MPDD、CMDC、模拟组与用户记录的数据质量；接入新数据前建议先体检。"
    )
    report = backend.data_health()
    cols = st.columns(4)
    cols[0].metric("总体状态", "健康" if report.get("healthy") else "有异常")
    cols[1].metric("阻断问题", report.get("issue_count", 0))
    cols[2].metric("风险提示", report.get("warning_count", 0))
    cols[3].metric(
        "生成时间",
        str(report.get("generated_at", ""))[11:19] or "-",
    )

    issues = report.get("issues", [])
    warnings = report.get("warnings", [])
    if not issues and not warnings:
        st.success("所有数据源均通过健康检查。")
    for item in issues:
        st.error(f"{item.get('source', '')}：{item.get('message', '')}")
    for item in warnings:
        st.warning(f"{item.get('source', '')}：{item.get('message', '')}")

    sources = report.get("sources", {})
    charls = sources.get("charls", {})
    if charls.get("ready"):
        with st.expander("CHARLS 养老调查数据", expanded=True):
            charls_cols = st.columns(4)
            charls_cols[0].metric("样本量", charls.get("rows", 0))
            charls_cols[1].metric("缺失率", f"{float(charls.get('missing_rate', 0)):.1%}")
            charls_cols[2].metric("阳性率", f"{float(charls.get('positive_rate') or 0):.1%}")
            imbalance = charls.get("imbalance_ratio")
            charls_cols[3].metric(
                "失衡比例",
                f"1:{imbalance:.1f}" if imbalance is not None else "-",
            )
    else:
        st.info("CHARLS 标准表尚未生成，可在“养老调查数据”页构建。")
        if st.button("前往养老调查数据", key="goto_charls_page"):
            navigate_to("养老调查数据")

    mpdd = sources.get("mpdd", {})
    if mpdd.get("ready"):
        with st.expander("MPDD 多模态数据", expanded=True):
            mpdd_cols = st.columns(4)
            mpdd_cols[0].metric("样本量", mpdd.get("rows", 0))
            mpdd_cols[1].metric("缺失率", f"{float(mpdd.get('missing_rate', 0)):.1%}")
            mpdd_cols[2].metric(
                "阳性率",
                f"{float(mpdd.get('positive_rate') or 0):.1%}"
                if mpdd.get("positive_rate") is not None
                else "无标签",
            )
            fold_counts = mpdd.get("fold_counts") or {}
            mpdd_cols[3].metric(
                "训练/测试",
                f"{fold_counts.get('train', 0)}/{fold_counts.get('test', 0)}",
            )
            modalities = mpdd.get("modalities") or {}
            if modalities:
                mod_df = pd.DataFrame(
                    [
                        {
                            "模态": MODALITY_LABELS.get(key, key),
                            "特征数": info.get("columns", 0),
                            "覆盖率": f"{float(info.get('coverage', 0)):.1%}",
                        }
                        for key, info in modalities.items()
                    ]
                )
                st.dataframe(mod_df, width="stretch", hide_index=True)
    else:
        st.info("MPDD 特征表尚未构建，可在“多模态数据”页生成。")
        if st.button("前往多模态数据", key="goto_mpdd_page"):
            navigate_to("多模态数据")

    cmdc = sources.get("cmdc", {})
    if cmdc.get("ready"):
        with st.expander("CMDC 中文多模态外部验证", expanded=False):
            cmdc_cols = st.columns(4)
            cmdc_cols[0].metric("受试者", cmdc.get("rows", 0))
            cmdc_cols[1].metric("HC / MDD", f"{cmdc.get('hc_subjects', 0)} / {cmdc.get('mdd_subjects', 0)}")
            cmdc_cols[2].metric("全模态", cmdc.get("complete_multimodal_subjects", 0))
            cmdc_cols[3].metric("访谈音频", f"{float(cmdc.get('audio_hours', 0)):.1f} 小时")
            st.caption(str(cmdc.get("age_scope_warning") or "混合成人样本，非老年专属"))
            modalities = cmdc.get("modalities") or {}
            if modalities:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "模态": info.get("label", key),
                                "覆盖人数": int(info.get("subjects", 0)),
                                "特征数": int(info.get("columns", 0)),
                                "覆盖率": f"{float(info.get('coverage', 0)):.1%}",
                            }
                            for key, info in modalities.items()
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
    else:
        st.info("CMDC 外部验证表尚未构建，可在“多模态数据”页生成。")
        if st.button("前往多模态数据构建 CMDC", key="goto_cmdc_page"):
            navigate_to("多模态数据")

    syn = sources.get("synthetic", {})
    if syn.get("ready"):
        with st.expander("模拟数据", expanded=False):
            syn_cols = st.columns(4)
            syn_cols[0].metric("对话样本", syn.get("dialogue_rows", 0))
            syn_cols[1].metric("表格样本", syn.get("tabular_rows", 0))
            syn_cols[2].metric("v3 验证", "通过" if syn.get("v3_feasible") else "未通过")
            syn_cols[3].metric("片段增强", "已生成" if syn.get("augmentation_ready") else "未生成")
    else:
        st.info("尚未生成模拟数据，不影响真实数据训练。")

    users = sources.get("users", {})
    with st.expander("用户记录", expanded=False):
        user_cols = st.columns(4)
        user_cols[0].metric("用户数", users.get("users", 0))
        user_cols[1].metric("评估数", users.get("assessments", 0))
        user_cols[2].metric(
            "缺失关键字段",
            sum((users.get("missing_fields") or {}).values()),
        )
        user_cols[3].metric("用户库大小", human_bytes(int(users.get("db_size_bytes", 0))))
        risk_dist = users.get("risk_distribution") or {}
        if sum(risk_dist.values()):
            st.dataframe(
                pd.DataFrame(
                    {
                        "风险等级": ["高风险", "中风险", "低风险", "暂无评估"],
                        "人数": [
                            risk_dist.get("high", 0),
                            risk_dist.get("medium", 0),
                            risk_dist.get("low", 0),
                            risk_dist.get("none", 0),
                        ],
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    st.download_button(
        "导出体检报告",
        data=data_health_markdown(report).encode("utf-8"),
        file_name=f"data_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        width="stretch",
    )


def render_smoothing_calibration() -> None:
    st.subheader("统计校正")
    st.caption(
        "针对小样本、缺失和时序噪声做统计校正：Beta-Binomial 概率收缩用于小亚组，"
        "稳健聚合用于比较多段特征的均值、中位数和截尾均值。"
    )
    flash = st.session_state.pop("smoothing_flash", None)
    if flash:
        st.success(flash)

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("运行小样本概率收缩", width="stretch", key="run_subgroup_shrinkage"):
            with st.spinner("正在计算小样本概率收缩..."):
                result = backend.run_subgroup_shrinkage()
            if result:
                st.session_state.smoothing_flash = "小样本概率收缩结果已更新"
            else:
                st.error("概率收缩运行失败，请检查后端和亚组报告")
            st.rerun()
    with action_cols[1]:
        if st.button("运行稳健聚合对比", width="stretch", key="run_robust_aggregation"):
            with st.spinner("正在比较 mean / median / trimmed mean，可能需要几分钟..."):
                result = backend.run_robust_aggregation_ablation()
            if result:
                st.session_state.smoothing_flash = "稳健聚合对比结果已更新"
            else:
                st.error("稳健聚合对比运行失败，请检查 MPDD utterance 表和后端")
            st.rerun()

    reports = backend.research_reports()
    subgroup_rows = reports.get("charls_subgroup_shrunk")
    if subgroup_rows:
        subgroup = pd.DataFrame(subgroup_rows)
        meta = reports.get("charls_subgroup_shrunk_meta") or {}
        st.markdown("#### 小亚组概率收缩")
        metric_cols = st.columns(3)
        metric_cols[0].metric("总体抑郁率", f"{float(meta.get('global_rate', 0)):.1%}")
        metric_cols[1].metric("先验强度", f"{float(meta.get('prior_strength', 0)):.1f}")
        metric_cols[2].metric(
            "平均收缩幅度", f"{float(meta.get('mean_absolute_change', 0)):.3f}"
        )
        display = subgroup.rename(
            columns={
                "group": "分组变量",
                "value": "分组",
                "n": "样本数",
                "raw_rate": "原始比例",
                "shrunk_rate": "收缩后比例",
                "ci_low": "95%CI下限",
                "ci_high": "95%CI上限",
                "absolute_change": "变化幅度",
                "auc": "AUC",
                "f1": "F1",
            }
        )
        for col in ["原始比例", "收缩后比例", "95%CI下限", "95%CI上限", "变化幅度"]:
            display[col] = display[col].map(lambda value: f"{float(value):.3f}")
        st.dataframe(display, width="stretch", hide_index=True)
        chart = subgroup.copy()
        chart["分组"] = chart["group"].astype(str) + ": " + chart["value"].astype(str)
        fig = px.bar(
            chart,
            x="分组",
            y=["raw_rate", "shrunk_rate"],
            barmode="group",
            labels={"value": "比例", "variable": "估计方式"},
            title="原始比例与 Beta-Binomial 收缩比例",
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("暂未生成小样本概率收缩报告。")

    robust_rows = reports.get("robust_aggregation")
    robust_meta = reports.get("robust_aggregation_meta") or {}
    if robust_rows:
        robust = pd.DataFrame(robust_rows)
        fusion = robust[robust["modality"] == "fusion"].copy()
        st.markdown("#### 稳健聚合对比")
        if robust_meta.get("recommended"):
            st.success(f"当前实验推荐聚合方式：{robust_meta['recommended']}")
        if not fusion.empty:
            fusion_display = fusion[
                ["aggregation", "auc_mean", "f1_mean", "brier_mean"]
            ].rename(
                columns={
                    "aggregation": "聚合方式",
                    "auc_mean": "AUC",
                    "f1_mean": "F1",
                    "brier_mean": "Brier",
                }
            )
            for col in ["AUC", "F1", "Brier"]:
                fusion_display[col] = fusion_display[col].map(lambda value: f"{float(value):.4f}")
            st.dataframe(fusion_display, width="stretch", hide_index=True)
            fig = px.bar(
                fusion,
                x="aggregation",
                y="auc_mean",
                color="aggregation",
                labels={"aggregation": "聚合方式", "auc_mean": "融合 AUC"},
                title="不同聚合方式的融合模型 AUC",
            )
            fig.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")
        with st.expander("查看各模态完整结果"):
            st.dataframe(robust, width="stretch", hide_index=True)
    else:
        st.info("暂未生成稳健聚合对比报告。")

    st.markdown("#### 校正边界")
    st.markdown(
        "- 概率收缩只改变小亚组风险估计，不改变原始标签和个体预测。\n"
        "- 稳健聚合目前推荐作为实验性训练选项，正式替换主模型前需要扩大重复次数并复查校准指标。\n"
        "- 平滑和收缩不能替代独立真实样本，也不能用于验证集或测试集。"
    )


def render_about() -> None:
    st.subheader("系统说明")
    account = st.session_state.get("auth_account") or {}
    is_admin = account.get("role") == "admin"
    status = backend.status()
    charls_ready = status.get("charls_ready", False)
    mpdd_ready = status.get("mpdd_ready", False)
    cmdc_ready = status.get("cmdc_ready", False)

    charls_text = "CHARLS 标准表已就绪" if charls_ready else "CHARLS 标准表尚未生成"
    if mpdd_ready:
        mpdd_path = backend.MPDD_UPLOAD_CSV if backend.MPDD_UPLOAD_CSV.exists() else backend.MPDD_CSV
        mpdd_df = pd.read_csv(mpdd_path)
        train_n = int((mpdd_df["fold"] == "train").sum()) if "fold" in mpdd_df else 0
        test_n = int((mpdd_df["fold"] == "test").sum()) if "fold" in mpdd_df else 0
        mpdd_text = (
            f"MPDD 特征表已就绪：{len(mpdd_df)} 人（训练 {train_n} / 测试 {test_n}），"
            "包含语音、面部、步态、人格四类模态"
        )
    else:
        mpdd_text = "MPDD 特征表尚未构建，可在多模态页一键生成"
    cmdc_text = (
        "CMDC 中文多模态外部验证表已就绪（混合成人样本，不作为老年主队列）"
        if cmdc_ready
        else "CMDC 外部验证表尚未构建"
    )

    st.markdown(
        f"""
        <div class="doc-block">
        <b>整体流程</b><br>
        对话采集语言线索 → 动态量表计分 → 面部表情/步态/语音特征 →
        养老调查表格风险模型 → 多模态概率融合 → 风险等级与因素解释。<br><br>
        <b>当前数据</b><br>
        · {charls_text}<br>
        · {mpdd_text}<br>
        · {cmdc_text}<br><br>
        <b>已集成模块</b><br>
        · 账号体系：登录注册、随机加减法人机验证、管理员/普通用户角色、后端数据权限隔离<br>
        · 风险评估页：对话、量表、人口学/共病、面部表情摘要<br>
        · 用户记录页：本地保存多用户档案、评估历史、风险趋势、JSON 备份与导入恢复、CSV 导出<br>
        · 评估报告：风险评估页可一键导出 Markdown 筛查报告，含风险构成、语言线索与对话记录<br>
        · 自适应访谈 Agent：温情问询与日常陪伴式聊天两种模式，各自保留对话记忆，可选大模型增强<br>
        · 养老调查页：标准表构建、数据分布、三类基线模型<br>
        · 数据平衡处理页：SMOTE、SMOTE-ENN、SMOTE-Tomek、ADASYN、过采样、欠采样和阈值优化<br>
        · 多模态页：MPDD 老年多模态基线，以及 CMDC 中文访谈外部验证；两个数据域独立建表和评估<br>
        · 系统说明页：当前模块、数据状态、论文依据与项目位置<br><br>
        <b>边界说明</b><br>
        本工作台仅用于科研筛查与风险提示，不替代临床诊断。<br>
        账号数据库只保存密码哈希，不保存明文密码；本地数据库权限限制为当前用户。<br>
        对话默认在本地处理。启用第三方大模型前必须单独同意，并在发送前自动隐藏姓名、住址、电话、证件号和邮箱。<br>
        真实可识别数据的采集、科研使用和保存期限必须另行通过伦理审批与知情同意程序。<br>
        MPDD 数据按官方许可协议申请获得，仅供本项目学术研究使用，禁止商用或任何形式对外再分发。
        CMDC 仅用于非商业学术研究，禁止再分发原始音频、转写或归档密码；其适用人群为混合成人，不是老年专属数据。
        </div>
        """,
        unsafe_allow_html=True,
    )
    if is_admin:
        st.markdown("#### 运行状态")
        status_metrics = st.columns(4)
        status_metrics[0].metric("用户档案", int(status.get("users_count", 0)))
        status_metrics[1].metric("评估记录", int(status.get("assessments_count", 0)))
        status_metrics[2].metric(
            "用户库大小", human_bytes(int(status.get("db_size_bytes", 0)))
        )
        status_metrics[3].metric(
            "后端连接", "在线" if status.get("connected") else "本地直连"
        )
        sync_version = st.session_state.get("_backend_version_seen") or status.get(
            "backend_version", ""
        )
        sync_started = st.session_state.get("_backend_started_at") or ""
        if sync_version:
            st.caption(
                f"后端版本：{sync_version}"
                + (f" · 启动于 {sync_started}" if sync_started else "")
            )
        artifacts = status.get("model_artifacts") or []
        if artifacts:
            artifact_df = pd.DataFrame(artifacts)
            artifact_df["大小"] = artifact_df["size"].map(human_bytes)
            artifact_df = artifact_df.rename(
                columns={"path": "文件", "modified_at": "更新时间"}
            )
            artifact_df = artifact_df.drop(columns=["size"], errors="ignore")
            st.dataframe(
                artifact_df[["文件", "更新时间", "大小"]],
                width="stretch",
                hide_index=True,
            )
        st.markdown("#### 项目位置")
        st.code(str(PROJECT_ROOT))
    else:
        st.info(
            "普通用户可进行风险评估、量表作答、对话记录和个人历史查看；"
            "数据构建、模型训练、模拟数据与全局统计仅供管理员使用。"
        )


def render_account_admin() -> None:
    st.subheader("账号管理")
    st.caption(
        "管理员可创建其他管理员或普通用户账号、停用账号和重置密码。"
        "当前登录管理员不能停用自己的账号或取消自己的管理员角色。"
    )
    flash = st.session_state.pop("admin_flash", None)
    if flash:
        st.success(flash)

    accounts = backend.admin_list_accounts()
    if accounts:
        account_df = pd.DataFrame(accounts)
        account_df["角色"] = account_df["role"].map(
            {"admin": "管理员", "user": "普通用户"}
        )
        account_df["状态"] = account_df["status"].map(
            {"active": "正常", "disabled": "已停用"}
        )
        account_df = account_df.rename(
            columns={
                "username": "用户名",
                "display_name": "显示名称",
                "created_at": "创建时间",
                "last_login_at": "最后登录",
            }
        )
        st.dataframe(
            account_df[
                ["用户名", "显示名称", "角色", "状态", "创建时间", "最后登录"]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("暂未读取到账号列表，请确认后端服务在线。")

    st.markdown("#### 新增账号")
    with st.form("admin_create_account_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("用户名", key="admin_new_username")
            display_name = st.text_input("显示名称", key="admin_new_display_name")
        with col2:
            password = st.text_input(
                "初始密码", type="password", key="admin_new_password"
            )
            confirm = st.text_input(
                "确认密码", type="password", key="admin_new_password_confirm"
            )
        role_label = st.selectbox(
            "账号角色",
            ["普通用户", "管理员"],
            key="admin_new_role",
        )
        create_submitted = st.form_submit_button("创建账号", width="stretch")
    if create_submitted:
        if password != confirm:
            st.session_state.admin_flash = "创建失败：两次输入的密码不一致"
        else:
            result = backend.admin_create_account(
                username.strip(),
                password,
                "admin" if role_label == "管理员" else "user",
                display_name.strip(),
            )
            if result.get("ok"):
                st.session_state.admin_flash = f"已创建账号：{username.strip()}"
            else:
                st.session_state.admin_flash = result.get("error") or "账号创建失败"
        st.rerun()

    if not accounts:
        return
    st.markdown("#### 修改账号")
    account_options = {
        f'{item["username"]}（{item["display_name"]}）': item["id"]
        for item in accounts
    }
    selected = st.selectbox("选择账号", list(account_options), key="admin_selected_account")
    selected_id = account_options[selected]
    selected_account = next(item for item in accounts if item["id"] == selected_id)
    with st.form("admin_update_account_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_display = st.text_input(
                "显示名称",
                value=selected_account.get("display_name") or "",
                key="admin_update_display_name",
            )
            role_label = st.selectbox(
                "角色",
                ["普通用户", "管理员"],
                index=0 if selected_account.get("role") == "user" else 1,
                key="admin_update_role",
            )
        with col2:
            status_label = st.selectbox(
                "状态",
                ["正常", "停用"],
                index=0 if selected_account.get("status") == "active" else 1,
                key="admin_update_status",
            )
            new_password = st.text_input(
                "新密码（留空则不修改）",
                type="password",
                key="admin_update_password",
            )
        update_submitted = st.form_submit_button("保存账号修改", width="stretch")
    if update_submitted:
        result = backend.admin_update_account(
            selected_id,
            role="admin" if role_label == "管理员" else "user",
            status="active" if status_label == "正常" else "disabled",
            display_name=new_display.strip(),
            password=new_password or None,
        )
        if result.get("ok"):
            st.session_state.admin_flash = f"已更新账号：{selected_account['username']}"
        else:
            st.session_state.admin_flash = result.get("error") or "账号更新失败"
        st.rerun()


def render_account_profile() -> None:
    st.subheader("个人主页")
    account = st.session_state.get("auth_account") or {}
    role_label = "管理员" if account.get("role") == "admin" else "普通用户"
    flash = st.session_state.pop("profile_flash", None)
    if flash:
        st.success(flash)

    metric_cols = st.columns(4)
    metric_cols[0].metric("用户名", account.get("username", ""))
    metric_cols[1].metric("角色", role_label)
    metric_cols[2].metric(
        "密保问题", "已设置" if account.get("has_security_questions") else "未设置"
    )
    metric_cols[3].metric("显示名称", account.get("display_name", ""))

    st.markdown("#### 修改账号资料")
    with st.form("profile_identity_form"):
        new_username = st.text_input(
            "用户名",
            value=account.get("username", ""),
            key="profile_username",
        )
        new_display_name = st.text_input(
            "显示名称",
            value=account.get("display_name", ""),
            key="profile_display_name",
        )
        identity_password = st.text_input(
            "当前密码",
            type="password",
            key="profile_identity_password",
        )
        identity_submitted = st.form_submit_button("保存账号资料", width="stretch")
    if identity_submitted:
        result = backend.update_own_profile(
            identity_password,
            username=new_username.strip(),
            display_name=new_display_name.strip(),
        )
        if result.get("ok"):
            st.session_state.auth_account = result.get("account") or {}
            st.session_state.profile_flash = "账号资料已更新"
        else:
            st.session_state.profile_flash = result.get("error") or "账号资料更新失败"
        st.rerun()

    st.markdown("#### 修改密码")
    with st.form("profile_password_form", clear_on_submit=True):
        current_password = st.text_input(
            "当前密码", type="password", key="profile_current_password"
        )
        new_password = st.text_input(
            "新密码", type="password", key="profile_new_password"
        )
        confirm_password = st.text_input(
            "确认新密码", type="password", key="profile_confirm_password"
        )
        password_submitted = st.form_submit_button("更新密码", width="stretch")
    if password_submitted:
        if new_password != confirm_password:
            st.session_state.profile_flash = "两次输入的新密码不一致"
        else:
            result = backend.update_own_profile(
                current_password,
                new_password=new_password,
            )
            if result.get("ok"):
                st.session_state.auth_account = result.get("account") or {}
                st.session_state.profile_flash = "密码已更新"
            else:
                st.session_state.profile_flash = result.get("error") or "密码更新失败"
        st.rerun()

    st.markdown("#### 修改密保问题")
    question_options = backend.auth_security_options() or [
        "您最喜欢的颜色是？",
        "您小时候居住的城市是？",
        "您最喜欢的一道菜是？",
        "您第一只宠物的名字是？",
    ]
    current_questions = list(account.get("security_questions") or [])
    with st.form("profile_security_form", clear_on_submit=True):
        security_password = st.text_input(
            "当前密码",
            type="password",
            key="profile_security_password",
        )
        selected_questions = []
        security_answers = []
        for index in range(3):
            q_col, a_col = st.columns([1.25, 1])
            with q_col:
                default_index = 0
                if (
                    index < len(current_questions)
                    and current_questions[index] in question_options
                ):
                    default_index = question_options.index(current_questions[index])
                question = st.selectbox(
                    f"问题 {index + 1}",
                    question_options,
                    index=default_index,
                    key=f"profile_security_question_{index}",
                )
            with a_col:
                answer = st.text_input(
                    f"新答案 {index + 1}",
                    key=f"profile_security_answer_{index}",
                )
            selected_questions.append(question)
            security_answers.append(answer)
        security_submitted = st.form_submit_button("更新密保问题", width="stretch")
    if security_submitted:
        if len(set(selected_questions)) != 3:
            st.session_state.profile_flash = "三个密保问题不能重复"
        else:
            result = backend.update_own_profile(
                security_password,
                security_questions=selected_questions,
                security_answers=security_answers,
            )
            if result.get("ok"):
                st.session_state.auth_account = result.get("account") or {}
                st.session_state.profile_flash = "密保问题已更新"
            else:
                st.session_state.profile_flash = result.get("error") or "密保问题更新失败"
        st.rerun()

    st.markdown("#### 账号安全提示")
    st.caption(
        "修改用户名或密码后，其他已登录设备会退出；当前页面会保持登录。"
        "明文账号记录文件仅供本机管理员保管，请勿上传或转发。"
    )


inject_css()
inject_care_css()
lock_chinese_language()
if not ensure_auth_session():
    render_auth_gate()
    st.stop()

account = st.session_state.get("auth_account") or {}
is_admin = account.get("role") == "admin"
admin_console_page = st.session_state.get("admin_console_page", "工作台")
render_header(
    show_admin_menu=is_admin,
    admin_console_page=admin_console_page,
)
if welcome_name := st.session_state.pop("login_welcome", None):
    st.toast(f"欢迎回来，{welcome_name}")
init_state()
prime_backend_version()
auto_sync_backend()
api_key = render_sidebar()

if is_admin:
    page_defs = [
        ("风险评估", lambda: render_assessment(api_key)),
        ("用户记录", render_records),
        ("养老调查数据", render_charls),
        ("不平衡对比", render_imbalance),
        ("融合校准", render_fusion),
        ("数据优化", render_data_cache),
        ("模拟数据", render_synthetic),
        ("多模态数据", render_mpdd),
        ("数据平衡处理", render_diy),
        ("后台任务", render_jobs),
        ("数据健康", render_data_health),
        ("统计校正", render_smoothing_calibration),
    ]
else:
    page_defs = [
        ("风险评估", lambda: render_assessment(api_key)),
        ("用户记录", render_records),
        ("个人主页", render_account_profile),
        ("系统说明", render_about),
    ]

console_pages = {
    "账号管理": render_account_admin,
    "个人主页": render_account_profile,
    "系统说明": render_about,
}
if is_admin and admin_console_page in console_pages:
    st.caption(f"管理菜单 · {admin_console_page}")
    console_pages[admin_console_page]()
else:
    apply_tab_navigation()
    nav_labels = [label for label, _ in page_defs]
    if st.session_state.get("main_nav") not in nav_labels:
        st.session_state.main_nav = nav_labels[0]
    selected = st.radio(
        "功能导航",
        options=nav_labels,
        key="main_nav",
        horizontal=True,
        label_visibility="collapsed",
    )
    renderers = dict(page_defs)
    renderers[selected]()
