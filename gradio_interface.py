import gradio as gr
from tool_functions import tools

# Gradio UI
with gr.Blocks() as demo:
    tools_for_gradio = tools()
    with gr.Tab("Instagram Tools"):
        # Create Instagram username input box
        instagram_username = gr.Textbox(label="Public instagram username")
        # Create slider to set the maximum images to download
        slider = gr.Slider(minimum=0, maximum=100, value=20)
        # Create button to extract images
        download_images_btn = gr.Button("Download images")
        # Bind scrapping function to the button
        download_images_btn.click(fn=tools_for_gradio.scrape_instagram_images, inputs=[instagram_username, slider], outputs=None)
        # Path where the folders are located
        path = "scrapped"
        # Get the list of folders in the specified path
        folder_list = tools_for_gradio.get_folder_list(path)
        # Create a dropdown input component with the folder list
        folder_dropdown = gr.inputs.Dropdown(choices=folder_list, label="Select Instagram")
        # Add the "Resize and Crop" button component
        resize_and_crop_btn = gr.Button("Resize and Crop from folder")
        # Bind cropping function to the button
        resize_and_crop_btn.click(fn=tools_for_gradio.process_images, inputs=folder_dropdown, outputs=None)
    
    with gr.Tab("Pre-training Tools"):
        # Path where the folders are located
        path = "scrapped"
        # Get the list of folders in the specified path
        folder_list = tools_for_gradio.get_folder_list(path)
        # Create a dropdown input component with the folder list
        folder_dropdown = gr.inputs.Dropdown(choices=folder_list, label="Select Instagram")
        # Buttons ...
        prompting_btn = gr.Button("Prompt images with BLIP from cropped images")
        prompting_btn.click(fn=tools_for_gradio.blip_captioning, inputs=folder_dropdown, outputs=None)
        with gr.Accordion("Check and correct prompting"):
            # Interface ...
            image_path = gr.State()
            txt_prompt = gr.State()
            image_index = gr.State(value=0)
            img = gr.Image(input=image_path)
            promtp = gr.Textbox(label="Generated prompt",value=txt_prompt, interactive=True)
            with gr.Row():
                checking_previous_btn = gr.Button("Previous Image")
                checking_previous_btn.click(fn=tools_for_gradio.change_image_blip_previous, 
                                            inputs=[folder_dropdown, image_index],
                                            outputs=[img, image_index, promtp])
                
                checking_good_btn = gr.Button("Change Blip Caption")
                checking_good_btn.click(fn=tools_for_gradio.change_caption,
                                        inputs=[folder_dropdown, promtp, image_index],
                                        outputs=None)
                
                checking_next_btn = gr.Button("Next Image")
                checking_next_btn.click(fn=tools_for_gradio.change_image_blip_next, 
                                        inputs=[folder_dropdown, image_index],
                                        outputs=[img, image_index, promtp])
        
demo.launch()