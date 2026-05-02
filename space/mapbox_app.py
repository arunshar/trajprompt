"""Gradio HF Space: Mapbox-themed trajectory search."""
from __future__ import annotations

import gradio as gr


def search(prompt: str, lookback_days: int):
    """Run TGARD + traj-CLIP scoring + SAM 2 confirmation."""
    return None, "[Stub: this would render a folium / Mapbox map with traces]"


def build_ui():
    with gr.Blocks(title="TrajPrompt") as demo:
        gr.Markdown(
            "# TrajPrompt\n"
            "Type a question. Get vessel trajectories on a live map.\n"
            "Examples: 'ships drifting near pipelines in the Gulf of Mexico last March', "
            "'tankers loitering > 6 hours within 5km of an offshore platform'.\n"
        )
        with gr.Row():
            prompt = gr.Textbox(label="Search prompt", placeholder="ships drifting near pipelines...")
            lookback = gr.Slider(1, 90, value=30, step=1, label="Lookback (days)")
        out_map = gr.Plot(label="Trajectory map")
        out_log = gr.Textbox(label="Hits", lines=8, interactive=False)
        gr.Button("Search", variant="primary").click(search, [prompt, lookback], [out_map, out_log])
    return demo


if __name__ == "__main__":
    build_ui().launch()
