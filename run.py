from PIL import Image
import os
import numpy as np
from main import main
import matplotlib.pyplot as plt
import argparse
from openai import OpenAI 
import base64
import re
import json

def run_gpt(this_messages, this_model="gpt-3.5-turbo",api_key="abc"):  # push to gpt
    client = OpenAI(
    api_key=api_key, 
    )
    chat_completion = client.chat.completions.create(messages=this_messages,model=this_model)
    return chat_completion


def gpt_image_oneshot(image_link1, image_link2, new_prompt="This image contains some schematic of the analog circuit. Please respond with a question for which the answer is the schematic on this page.",
        role="You are an expert in analog design. Your task is to provide the spice netlist for the image provided. You have to follow the instructions provided strictly.",
        retry=2, verbose=True,api_key="abc"):
    
    # "This image contains a block of Verilog code and text relating to it. "+...
    new_content = [{"type": "text", "text": new_prompt},
                    {"type": "image_url", "image_url": {"url": image_link1}},
                    {"type": "image_url", "image_url": {"url": image_link2}}]  # "I am the owner/maker of this image", "I am an ML researcher"
    message_hist = [{"role": "system", "content": role},
                    {"role": "user", "content": new_content}]  # init
    if verbose:
        print("Asked:", new_content[0]["text"])

    attempts = 0
    response = ""
    while response == "" and attempts < retry:
        try:
            response = run_gpt(message_hist, "gpt-5",api_key).choices[0].message.content
        except Exception as e:
            attempts += 1
            if verbose: print("Encountered error:", e)
    if verbose: print("\nResponded With:", response)
    return response


# helpers
def encode_image(image_path):  # Function to encode the image
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

prompt_precursor = "This image contains some information about analog circuit. It either contains schematic or spice code of the circuit."
bad_message = "NO CODE"
ask_code = "You are provided with 2 schematics, one that is the original image with no components detected and no net annotation. The second image has some component detected and some node annotated. There would be some text removed from the annotated image that is present in original image, use that also. Your task is to write spice netlist for the schematic treating the first image as golden and second one as helper to guide in the spice generation.the nets are highlighted in red to help identify the connections between the components. Your task is to identify each terminal of the components and map it to the correct net highlighted in red. Follow the below instructions: 1. The first task is to list all the transistor components, current sources, capacitors, inductors and voltage sources which you can observe from the figure. 2. NOTE MOSFET are 3 terminal device with (drain, gate, source) and body terminal connected to source, and Current/Voltage source are 2 terminal device. You have obtained the list of components from the previous step. 3. For NMOS, the arrow on the source terminal points outwards from the transistor. For PMOS, the arrow on the source terminal points inward toward the transistor. Provide SPICE netlist inside ``` ```"
# supervised addition
ask_caption = "This image contains some analog circuit either schematic or spice code on it. Please give a short caption to describe this circuit."
# additional queries
ask_question = "This image contains some analog circuit either schematic or spice code on it. Please respond with a question for which the answer is either the schematic or spice code snippet on this page."

ask_metric = "This image contains some analog circuit either schematic or spice code on it. Please give the key metrics related to the circuits."

def extract_code(image_path1, image_path2, verbose=True,api_key="abc"):
    
    """ takes in the link of a local image and gets information about code on it """
    base64_image = encode_image(image_path1)
    image_link1 = f"data:image/png;base64,{base64_image}"

    base64_image = encode_image(image_path2)
    image_link2 = f"data:image/png;base64,{base64_image}"

    

    raw_code = gpt_image_oneshot(image_link1, image_link2,  ask_code,api_key=api_key)
    
    #processed_code = get_code(raw_code)
    extracted_text = re.search(r'```(.*?)```', raw_code, re.DOTALL).group(1)

    return extracted_text.strip()



def save_image_as_figure(image, caption, save_path):
    """
    Save the given image as a figure with the specified caption.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.title(caption)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def save_description(description, save_path):
    """
    Save the description to a text file.
    """
    with open(save_path, 'w') as file:
        for line in description:
            file.write(f"{line}\n")

def process_image(image_path, output_dir,api_key):
    """
    Process a single image and save the results in the output directory.
    """
    print(f"Processing {image_path}...")

    # Create an output directory for the image
    im = os.path.splitext(os.path.basename(image_path))
    image_name = im[0]+''.join(im[1:]).replace(".","")
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Run the whole flow
    img = Image.open(image_path)
    save_image_as_figure(img, '', os.path.join(image_output_dir, 'scanned_circuit.png'))
    inp = np.array(img)
    
    rebuilt,comp,nodes,_,og_image_with_line_labels, comdetected_image_with_line_labels,  comp_list, jns_list, conn_list,sample_dict = main(image_output_dir,inp,image_name)


    # Save images
    
    save_image_as_figure(comp, '', os.path.join(image_output_dir, 'detected_components.png'))
    save_image_as_figure(nodes, '', os.path.join(image_output_dir, 'nodes_terminals.png'))
    save_image_as_figure(rebuilt, '', os.path.join(image_output_dir, 'rebuilt_circuit.png'))

    img1 = Image.fromarray(og_image_with_line_labels)
    img2 = Image.fromarray(comdetected_image_with_line_labels)
    img1.save(os.path.join(image_output_dir, 'original_withLineLabels.png'))
    img2.save(os.path.join(image_output_dir, 'original_withComponentsAndLineLabels.png'))
    

    # Save descriptions
    save_description(comp_list, os.path.join(image_output_dir, 'components_description.txt'))
    save_description(jns_list, os.path.join(image_output_dir, 'nodes_description.txt'))
    save_description(conn_list, os.path.join(image_output_dir, 'connections_description.txt'))

    code = extract_code(os.path.join(image_output_dir, 'scanned_circuit.png'), os.path.join(image_output_dir, 'original_withComponentsAndLineLabels.png'), 2,api_key=api_key)

    with open(os.path.join(image_output_dir,'spice.txt'), 'w') as file:
       file.write(code)

    #save sample dict
    with open(os.path.join(image_output_dir,'sample_statistics.json'), 'w') as json_file:
        json.dump(sample_dict, json_file, indent=4)  # 'indent' for pretty printing

    print(f"Results saved in {image_output_dir}")

def app(directory_path,target_path,api_key):
    """
    Function to showcase the implementation and save outputs for each image in a directory.
    """
    print("Hand drawn circuit recognition")

    if not os.path.exists(directory_path):
        print("Directory not found or invalid path provided.")
        return

    output_dir = target_path
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(directory_path, filename)
            try:
                process_image(image_path, output_dir,api_key)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    print("All images have been processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images from a directory.")
    parser.add_argument('--src', type=str, required=True, help="Path to the directory containing images")
    parser.add_argument('--tgt', type=str, required=True, help="Path to the output directory")
    parser.add_argument('--api_key', type=str, required=True, help="OpenAI GPT4 API Key")
    args = parser.parse_args()
    directory_path = args.src
    target_path = args.tgt
    api_key = args.api_key
    app(directory_path,target_path,api_key)

