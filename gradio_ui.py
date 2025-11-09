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


def get_campaign_summaries_ui(campaign_id):
    """
    Fetch summaries from API and return as a list of [topic, summary] for Gradio Dataframe.
    """
    try:
        response = requests.get(f"http://localhost:8000/api/campaigns/{campaign_id}/summaries")
        data = response.json()
        if data.get("success"):
            summaries = data["summaries"]
            # Convert to list of [topic, summary] rows
            table_data = [[s.get("topic", "N/A"), s.get("summary", "N/A")] for s in summaries]
            return table_data
        return [["Error", json.dumps(data)]]
    except Exception as e:
        return [["Error", str(e)]]



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

        with gr.Tab("Develop Post"):
            business_type_post = gr.Textbox(label="Business Type", value="Coffee Shop")
            platform_post = gr.Dropdown(["Facebook", "Instagram", "LinkedIn", "Twitter"], label="Platform",  value="Instagram")
            total_posts_post = gr.Number(label="Number of Summaries", value=5, minimum=1, maximum=10)

            generate_summaries_btn = gr.Button("1️⃣ Generate Summaries")
            summaries_list = gr.Textbox(label="Generated Summaries", lines=10, interactive=False)
            state_campaign = gr.State()  # to hold campaign_id and summaries

            generate_posts_btn = gr.Button("2️⃣ Generate Posts from Summaries", interactive=False)
            posts_output = gr.Textbox(label="Generated Posts", lines=20)

            # Step 1 — create campaign and get summaries
            def create_campaign_and_show_summaries(business_type, platform, total_posts):
                resp = requests.post(
                    "http://localhost:8000/api/campaigns/create",
                    json={"business_type": business_type, "platform": platform, "total_posts": int(total_posts)},
                )
                data = resp.json()
                if not data.get("success"):
                    return f"Error: {data}", None, gr.update(interactive=False)


                # headlines = data.get("headlines", {}).get("headlines", [])
                # text = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
                # return text, {"campaign_id": data["campaign_id"], "headlines": headlines}, gr.update(interactive=True)

            generate_summaries_btn.click(
                create_campaign_and_show_summaries,
                inputs=[business_type_post, platform_post, total_posts_post],
                outputs=[summaries_list, state_campaign, generate_posts_btn],
            )

            # Step 2 — iterate summaries and call /api/develop_post for each
            def generate_posts_from_summaries(state, business_type, platform):
                if not state or "headlines" not in state:
                    return "No summaries found."
                results = []
                for i, summary in enumerate(state["headlines"], start=1):
                    resp = requests.post(
                        "http://localhost:8000/api/develop_post",
                        json={"business_type": business_type, "platform": platform},
                        stream=True,
                    )
                    post_text = ""
                    for line in resp.iter_lines():
                        if line and line.startswith(b"data:"):
                            payload = json.loads(line[5:])
                            if "post" in payload:
                                post_text = payload["post"]
                    results.append(f"=== Post {i} ===\n{post_text}\n")
                return "\n\n".join(results)

            generate_posts_btn.click(
                generate_posts_from_summaries,
                inputs=[state_campaign, business_type_post, platform_post],
                outputs=posts_output,
            )

        with gr.Tab("Create Campaign"):
            business_type = gr.Textbox(label="Business Type", value="Coffee Shop")
            platform = gr.Dropdown(["Facebook", "Instagram", "LinkedIn", "Twitter"], label="Platform", value="Instagram")
            total_posts = gr.Number(label="Total Posts", value=5, minimum=1, maximum=30)
            create_btn = gr.Button("Create Campaign")
            campaign_output = gr.Code(label="Campaign Result", language="json")

            create_btn.click(create_campaign_ui, inputs=[business_type, platform, total_posts], outputs=campaign_output)

        with gr.Tab("View Summaries"):
            campaign_id_summaries = gr.Textbox(label="Campaign ID")
            view_summaries_btn = gr.Button("Fetch Summaries")
            summaries_table = gr.Dataframe(
                headers=["Topic", "Summary"],
                datatype=["str", "str"],
                interactive=False
            )

            view_summaries_btn.click(
                get_campaign_summaries_ui,
                inputs=campaign_id_summaries,
                outputs=summaries_table,
            )


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


    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("WS_PORT", 7860)),
    )

if __name__ == "__main__":
    launch_gradio_ui()
