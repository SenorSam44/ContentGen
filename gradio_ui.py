# gradio_ui.py
import json
import os

import gradio as gr
import requests


def create_campaign_ui(business_type, platform, total_posts):
    try:
        response = requests.post("http://localhost:8000/api/campaigns/create", json={
            "business_type": business_type,
            "platform": platform,
            "total_posts": int(total_posts)
        })
        return json.dumps(response.json(), indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


def check_campaign_status(campaign_id):
    try:
        response = requests.get(f"http://localhost:8000/api/campaigns/{campaign_id}/status")
        return json.dumps(response.json(), indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


def get_campaign_posts_ui(campaign_id):
    try:
        response = requests.get(f"http://localhost:8000/api/campaigns/{campaign_id}/posts")
        data = response.json()
        if data.get("success"):
            posts = data["posts"]
            result = ""
            for post in posts:
                result += f"=== Post #{post['post_order']}: {post['headline']} ===\n"
                result += f"Status: {post['status']}\n"
                if post['content']:
                    result += f"Content: {post['content']}\n"
                result += "\n" + "=" * 50 + "\n\n"
            return result
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


def generate_post_from_headline(business_type, platform):
    try:
        response = requests.post("http://localhost:8000/api/develop_post", json={
            "business_type": business_type,
            "platform": platform
        })
        return response.json().get("post", "No content generated.")
    except Exception as e:
        return f"Error: {str(e)}"


def launch_gradio_ui():
    with gr.Blocks(title="Social Media Post Generator") as demo:
        gr.Markdown("# 🚀 Social Media Post Generator")

        with gr.Tab("Create Campaign"):
            business_type = gr.Textbox(label="Business Type", value="Coffee Shop")
            platform = gr.Dropdown(["Facebook", "Instagram", "LinkedIn", "Twitter"], value="Instagram")
            total_posts = gr.Number(label="Total Posts", value=5, minimum=1, maximum=30)
            create_btn = gr.Button("Create Campaign")
            campaign_output = gr.Code(label="Campaign Result", language="json")

            create_btn.click(create_campaign_ui, inputs=[business_type, platform, total_posts], outputs=campaign_output)

        with gr.Tab("Check Status"):
            campaign_id_input = gr.Textbox(label="Campaign ID")
            check_btn = gr.Button("Check Status")
            status_output = gr.Code(label="Status", language="json")

            check_btn.click(check_campaign_status, inputs=campaign_id_input, outputs=status_output)

        with gr.Tab("View Posts"):
            campaign_id_posts = gr.Textbox(label="Campaign ID")
            view_btn = gr.Button("View Posts")
            posts_output = gr.Textbox(label="Generated Posts", lines=20)

            view_btn.click(get_campaign_posts_ui, inputs=campaign_id_posts, outputs=posts_output)

        with gr.Tab("Develop Post"):
            business_type_post = gr.Textbox(label="Business Type")
            platform_post = gr.Dropdown(["Facebook", "Instagram", "LinkedIn", "Twitter"], value="Instagram")
            generate_btn = gr.Button("Generate Post")
            post_output = gr.Textbox(label="Generated Post", lines=10)

            generate_btn.click(generate_post_from_headline,
                               inputs=[business_type_post, platform_post],
                               outputs=post_output)


    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("WS_PORT", 7860)),
    )

if __name__ == "__main__":
    launch_gradio_ui()
