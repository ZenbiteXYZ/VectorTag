import streamlit as st
import sys
import os
from typing import Dict
from PIL import Image

# Ensure src is in path BEFORE importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)

# Now import from src
from src.ui.modes.base import InferenceMode, TagManagementMixin
from src.ui.modes.standard import StandardMode


# Page Config
st.set_page_config(
    page_title="VectorTag AI",
    page_icon="🏷️",
    layout="wide"
)

# --- Sidebar: Configuration ---
st.sidebar.title("⚙️ Configuration")

# Mode Selection
modes: Dict[str, InferenceMode] = {
    "Standard": StandardMode(),
}


selected_mode_name = st.sidebar.selectbox("Inference Mode", list(modes.keys()))
active_mode = modes[selected_mode_name]

st.sidebar.info(f"**{active_mode.name}**\n\n{active_mode.description}")

# --- Model Selection ---
available_models = active_mode.get_available_models()
if available_models and len(available_models) > 1:
    st.sidebar.subheader("Model Version")
    selected_model_display = st.sidebar.selectbox(
        "Choose Version",
        list(available_models.keys()),
        key="model_selector"
    )
    selected_model_key = available_models[selected_model_display]

    # Toggle model if changed
    if active_mode.current_model_key != selected_model_key:
        with st.spinner("Loading model..."):
            try:
                active_mode.set_model(selected_model_key)
                st.sidebar.success(f"✅ Loaded: {selected_model_display}")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")


# Threshold Slider
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)


# --- Main Area ---
st.title("🏷️ VectorTag: Image Auto-Tagging")

tab_predict, tab_manage = st.tabs(["🖼️ Predict", "🗂️ Manage Tags"])


# --- Tab 1: Predict ---
with tab_predict:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Layout: Left (Image), Right (Tags & Controls)
        col_img, col_tags = st.columns([3, 2])

        # Load Image
        original_image = Image.open(uploaded_file).convert('RGB')

        # Run Inference
        with st.spinner(f"Running {active_mode.name}..."):
            try:
                results = active_mode.predict(original_image, threshold)
            except Exception as e:
                st.error(f"Error: {e}")
                results = []

        # Remember selection across reruns
        if "selected_tag" not in st.session_state:
            st.session_state["selected_tag"] = None

        # --- Right Column: Tags & Search ---
        with col_tags:
            st.subheader("🏷️ Detected Tags")

            # 1. Search Bar
            search_query = st.text_input("🔍 Search tags", "").lower()

            # 2. Filter Results
            filtered_results = [
                (tag, prob) for tag, prob in results
                if search_query in tag.lower()
            ]
            # 3. Tag Selection + Probabilities (single panel, clickable tags)
            selected_tag = st.session_state.get("selected_tag")
            if filtered_results:
                # Drop selection if not in filtered set
                filtered_tag_names = [tag for tag, _ in filtered_results]
                if selected_tag not in filtered_tag_names:
                    selected_tag = None
                    st.session_state["selected_tag"] = None

                st.write(f"Found {len(filtered_results)} tags above threshold.")
                st.caption("Click on tag to show Grad-CAM. 'Clear' clears selection.")

                for tag, prob in filtered_results:
                    c1, c2 = st.columns([1, 3])
                    button_label = f"✅ {tag}" if selected_tag == tag else tag
                    if c1.button(button_label, key=f"select_{tag}"):
                        st.session_state["selected_tag"] = tag
                        st.rerun()
                    c2.progress(prob, text=f"{prob:.1%}")

                cols_clear = st.columns([1, 3])
                with cols_clear[0]:
                    if st.button("Clear", key="clear_tag", icon="🔄"):
                        st.session_state["selected_tag"] = None
                        st.rerun()
            else:
                st.info("No tags found above threshold.")

        # --- Left Column: Image Visualization ---
        with col_img:
            st.subheader("Visualization")

            display_image = original_image

            # Check if user selected a tag
            if selected_tag and hasattr(active_mode, 'get_gradcam_image'):
                with st.spinner(f"Generating heatmap for '{selected_tag}'..."):
                    try:
                        display_image = active_mode.get_gradcam_image(original_image, selected_tag)
                        st.caption(f"🔥 Heatmap for **{selected_tag}**")
                    except Exception as e:
                        st.error(f"Grad-CAM error: {e}")

            st.image(display_image, width="stretch")

    else:
        st.info("Please upload an image to start.")


# --- Tab 2: Manage tags ---
with tab_manage:
    # If mode supports tag management
    if isinstance(active_mode, TagManagementMixin):
        st.header("Tag Management")

        # Add Tag Form
        with st.form("add_tag_form"):
            col_form_1, col_form_2 = st.columns([1, 2])

            with col_form_1:
                new_tag = st.text_input("New Tag Name", placeholder="e.g., 'golden_retriever'")

            with col_form_2:
                new_description = st.text_input("Description (optional)", placeholder="e.g., 'A friendly dog breed with golden fur'")

            submitted = st.form_submit_button("Add Tag")
            if submitted and new_tag:
                try:
                    active_mode.add_tag(new_tag, new_description)
                    st.success(f"Tag '{new_tag}' added!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()

        # List Tags
        st.subheader("Existing Tags")
        try:
            tags = active_mode.get_all_tags()
            if not tags:
                st.info("Database is empty.")

            for tag in tags:
                c1, c2 = st.columns([4, 1])
                c1.text(tag)
                if c2.button("Delete", key=f"del_{tag}"):
                    active_mode.remove_tag(tag)
                    st.rerun()
        except Exception as e:
            st.error(f"Error fetching tags: {e}")

    else:
        st.info(f"The current mode '{active_mode.name}' is read-only (Standard Mode). Switch to Semantic Mode to manage tags.")
