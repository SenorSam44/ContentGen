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


def develop_posts_ui(campaign_id, summaries=None):
    # try:
    payload = {"campaign_id": campaign_id}
    if summaries:
        payload["summaries"] = summaries  # expects list of {topic, summary}

    response = requests.post("http://localhost:8000/api/campaigns/develop_posts", json=payload)
    data = response.json()

    if data.get("success"):
        posts = data.get("posts", [])
        return "\n\n".join(
            [f"=== Post #{i+1} ===\n{p}\n" for i, p in enumerate(posts)]
        )
    return json.dumps(data, indent=2)
    # except Exception as e:
    #     return f"Error: {str(e)}"


def launch_gradio_ui():
    with gr.Blocks(title="Social Media Post Generator") as demo:
        gr.Markdown("# 🚀 Social Media Post Generator")


        with gr.Tab("Develop Post"):
            business_type_post = gr.Textbox(label="Business Type", value="Coffee Shop")
            platform_post = gr.Dropdown(["Facebook", "Instagram", "LinkedIn", "Twitter"],
                                        label="Platform", value="Instagram")
            total_posts_post = gr.Number(label="Number of Summaries", value=5, minimum=1, maximum=10)

            generate_summaries_btn = gr.Button("1️⃣ Generate Summaries")
            summaries_json_box = gr.Textbox(
                label="Generated Summaries (Editable JSON)",
                lines=10,
                value="[]",
                interactive=True
            )
            state_campaign = gr.State()

            develop_posts_btn = gr.Button("2️⃣ Develop Posts from Summaries", interactive=False)
            posts_output = gr.Textbox(label="Generated Posts", lines=20)

            # Step 1: Generate summaries as JSON and display them
            def create_campaign_and_show_summaries(business_type, platform, total_posts):
                resp = requests.post(
                    "http://localhost:8000/api/campaigns/create",
                    json={"business_type": business_type, "platform": platform, "total_posts": int(total_posts)},
                )
                data = resp.json()
                if not data.get("success"):
                    return json.dumps(data, indent=2), None, gr.update(interactive=False)

                summaries = data.get("summaries", [])
                json_text = json.dumps(summaries, indent=2)
                state = {"campaign_id": data["campaign_id"]}
                return json_text, state, gr.update(interactive=True)

            generate_summaries_btn.click(
                create_campaign_and_show_summaries,
                inputs=[business_type_post, platform_post, total_posts_post],
                outputs=[summaries_json_box, state_campaign, develop_posts_btn],
            )

            # Step 2: Use edited JSON summaries to generate posts
            def develop_posts_from_summaries(state, summaries_text):
                if not state or "campaign_id" not in state:
                    return "No valid campaign found."

                try:
                    summaries = json.loads(summaries_text)
                except json.JSONDecodeError as e:
                    return f"Invalid JSON format: {e}"

                return develop_posts_ui(state["campaign_id"], summaries)

            develop_posts_btn.click(
                develop_posts_from_summaries,
                inputs=[state_campaign, summaries_json_box],
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
