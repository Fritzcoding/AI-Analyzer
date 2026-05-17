import streamlit as st
from pathlib import Path
import json
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Hypothyroid ML", layout="wide")
st.title("🏥 Hypothyroid ML Benchmark Dashboard")

results_file = Path("outputs/results.json")
if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Models", len(results))
    
    best_model = max(results, key=lambda x: x['accuracy'])
    with col2:
        st.metric("Best Accuracy", f"{best_model['accuracy']:.2%}")
    with col3:
        st.metric("Best Model", best_model['system'])
    with col4:
        st.metric("Best F1-Score", f"{best_model['f1']:.4f}")
    
    st.divider()
    
    # Results table
    st.subheading("📊 Model Performance Comparison")
    df = pd.DataFrame(results)
    df_display = df[['system', 'classifier', 'accuracy', 'precision', 'recall', 'f1']].copy()
    df_display['accuracy'] = df_display['accuracy'].apply(lambda x: f"{x:.2%}")
    df_display['precision'] = df_display['precision'].apply(lambda x: f"{x:.4f}")
    df_display['recall'] = df_display['recall'].apply(lambda x: f"{x:.4f}")
    df_display['f1'] = df_display['f1'].apply(lambda x: f"{x:.4f}")
    st.dataframe(df_display, use_container_width=True)
    
    st.divider()
    
    # Display figures if they exist
    st.subheading("📈 Visualizations")
    
    figures_dir = Path("outputs/figures")
    if figures_dir.exists():
        png_files = sorted(figures_dir.glob("*.png"))
        
        # Show accuracy comparison first
        acc_comp = figures_dir / "accuracy_comparison.png"
        if acc_comp.exists():
            st.image(str(acc_comp), caption="Accuracy Comparison", use_column_width=True)
        
        # Group confusion matrices
        st.subheading("Confusion Matrices")
        cm_files = [f for f in png_files if f.name.startswith("cm_")]
        cols = st.columns(2)
        for idx, fig_file in enumerate(cm_files[:4]):
            with cols[idx % 2]:
                st.image(str(fig_file), caption=fig_file.stem, use_column_width=True)
    else:
        st.info("No figures found. Run: python src/04_evaluate.py")
else:
    st.warning("⚠️ No results found. Please run the training pipeline:")
    st.code("""
python src/01_preprocessing.py
python src/02_sklearn_classifiers.py
python src/03_tensorflow_dnn.py
python src/04_evaluate.py
    """)
