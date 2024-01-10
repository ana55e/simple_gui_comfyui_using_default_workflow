import json
import gradio as gr
import google.generativeai as genai
import PIL
genai.configure(api_key='your api')  # Replace with your API key
history = []  

def generate_text(text):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(text)
    return response.text

def generate_text_from_image(image, text):
    global history
    model = genai.GenerativeModel('gemini-pro-vision')
    image = PIL.Image.fromarray(image.astype('uint8'), 'RGB')
    history.append(("User", text))
    response = model.generate_content([text, image])
    history.append(("Bot", response.text))
    return response.text,history


def interactive_chat(message, chat_history=None):
    if chat_history is None:
        chat_history = []
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat()
    response = chat.send_message(message)
    chat_history.append(("User", message))
    chat_history.append(("Bot", response.text))
    
    return chat_history

# Creating interfaces for each function
text_interface = gr.Interface(
    fn=generate_text,
    inputs=gr.inputs.Textbox(label="Enter text"),
    outputs=gr.outputs.Textbox(label="Generated Text")
)

image_interface = gr.Interface(
    fn=generate_text_from_image,
    inputs=[gr.inputs.Image(label="Upload Image"), gr.inputs.Textbox(label="Enter text")],
    outputs=[gr.outputs.Textbox(label="Generated Text"), gr.outputs.Textbox(label="Chat History", type="text")]
)

chat_interface = gr.Interface(
    fn=interactive_chat,
    inputs=gr.inputs.Textbox(label="Chat with the bot"),
    outputs=gr.outputs.Chatbot(label="Chatbot Response")
)





def open_workflow(path):
    
    a=path
    with open(a) as f:
        data = json.load(f)
    
    
    return data

    

    



def all_changes(path,seed,model,sampler,steps,cfg,scheduler,width,height,batch_size,positive_prompt,negative_prompt):
    a=open_workflow(path)
    a["3"]['inputs']['seed']=seed
    a["4"]["inputs"]["ckpt_name"]=model
    a["3"]["inputs"]["sampler_name"]=sampler
    a["3"]["inputs"]["steps"]=steps
    a["3"]["inputs"]["cfg"]=cfg
    a["3"]["inputs"]["scheduler"]=scheduler
    a["5"]["inputs"]["width"]=width
    a["5"]["inputs"]["height"]=height
    a["5"]["inputs"]["batch_size"]=batch_size
    a["6"]["inputs"]["text"]=positive_prompt
    a["7"]["inputs"]["text"]=negative_prompt
    with open(path, 'w') as outfile:
        json.dump(a, outfile)

def queue_prompt(path,seed,model,sampler,steps,cfg,scheduler,width,height,batch_size,positive_prompt,negative_prompt):
    all_changes(path,seed,model,sampler,steps,cfg,scheduler,width,height,batch_size,positive_prompt,negative_prompt)
    prompt=open_workflow(path)
    from urllib import request, parse
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data) # this will make the method "POST", its like requests.post(url, data=data)
    request.urlopen(req)
    return 'done'


        

    


from pathlib import Path

n=input("please enter the path of stable diffusion models: ")
path=Path(n)

def find_stable_diffusion_models(path):
    a=path.glob('*')
    # take just the files with .pt or .pth or .safetensors not the whole path
    return [str(i).split('\\')[-1] for i in a if str(i).split('\\')[-1].endswith('.pt') or str(i).split('\\')[-1].endswith('.pth') or str(i).split('\\')[-1].endswith('.safetensors')]

 

txt2img=gr.Interface(
    fn=queue_prompt,
    inputs=[
        gr.inputs.Textbox(lines=1, label="workflow file Path"),

        # slider for seed
        gr.inputs.Slider(1,999999999999, 100,12584223658, label="Seed"), # min, max, step, default, label
        #  dropdown for model using the text box input above
        gr.inputs.Dropdown(find_stable_diffusion_models(path), label="Model"),
        # dropdown for sampler
        gr.inputs.Dropdown(["euler", "euler_ancestral", "heun","heunpp2","dpm_2","dpm_2_ancestral","lms","dpm_fast","dpm_adaptive","dpmpp_2s_ancestral","dpmpp_sde","dpmpp_sde_gpu","dpmpp_2m","dpmpp_2m_sde","dpmpp_2m_sde_gpu","dpmpp_3m_sde","dpmpp_3m_sde_gpu","ddpm","lcm","ddim","uni_pc","uni_pc_bh2"], label="Sampler"),
        # slider for steps
        gr.inputs.Slider(1,100, 1,30, label="Steps"), # min, max, step, default, label
        # slider for cfg
        gr.inputs.Slider(1,12, 1,8, label="CFG"), # min, max, step, default, label
        # dropdown for scheduler
        gr.inputs.Dropdown(["simple", "normal", "karras", "sgm_uniform","exponential","ddim_uniform"], label="Scheduler"),
        # slider for width
        gr.inputs.Slider(1,3200, 1,512, label="Width"), # min, max, step, default, label
        # slider for height
        gr.inputs.Slider(1,3200, 1,512, label="Height"), # min, max, step, default, label
        # slider for batch_size
        gr.inputs.Slider(1,100, 1,1, label="Batch Size"), # min, max, step, default, label
        # textbox for positive_prompt
        gr.inputs.Textbox(lines=10, label="Positive Prompt"),
        # textbox for negative_prompt
        gr.inputs.Textbox(lines=10, label="Negative Prompt")
        

    ],
    outputs=[
        gr.inputs.Textbox(lines=1, label="please wait for the output to appear in the comfy output and check the cmd"),
    ],
    #outputs="text",
    title="File Editor",
    description="you can use this to change the content of a file"
)

iface = gr.TabbedInterface([text_interface, image_interface, chat_interface,txt2img])
iface.launch()

