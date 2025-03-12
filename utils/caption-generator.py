import json
from PIL import Image, ImageDraw
from pdf2image import convert_from_path
import openai
import base64
import os
import sys
from io import BytesIO

def save_annotated_image(image, annotate_image_path, max_width=800, quality=70):
    """
    Resizes the `image` to `max_width` if it's larger, 
    then saves it with specified JPEG `quality` to reduce file size.
    """
    w, h = image.size
    if w > max_width:
        ratio = max_width / float(w)
        new_height = int(h * ratio)
        # Resize using a high-quality resampling filter
        image = image.resize((max_width, new_height), Image.LANCZOS)

    # Save as JPEG with reduced quality
    image.save(annotate_image_path, format="JPEG", quality=quality)

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def draw_bounding_boxes(image, box):
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline="red", width=2)
    return image

def encode_image(image_path):
    # Since the image is already saved smaller, we can just encode it as is
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def run_gpt(messages, model="gpt-3.5-turbo", api_key="abc"):
    client = openai.OpenAI(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=messages, 
        model=model, 
        # Limit output length
        max_tokens=200 
    )
    return chat_completion

def generate_description(image_path, api_key):
    encoded_image = encode_image(image_path)
    image_link = f"data:image/png;base64,{encoded_image}"
    # Changed prompt to ensure brevity
    prompt = "You are provided with a page from a textbook showing a circuit. Write a short description of the circuit in less than 3 sentences. Your description should contain sufficient information for a text to spice model conversion."
    new_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_link}},
    ]
    messages = [
        {"role": "system", "content": "You are an expert in analog design."},
        {"role": "user", "content": new_content},
    ]
    response = run_gpt(messages, model="gpt-4o", api_key=api_key)
    return response.choices[0].message.content

def save_description(description, page_num, image_num,description_path):
    file_name = f'{description_path}/{page_num}_{image_num}.txt'
    with open(file_name, 'w') as file:
        file.write(description)

def process_images(json_data, pdf_path, api_key,annotated_pages_path,description_path):
    pages = convert_from_path(pdf_path)
    for page_num, page_data in json_data['pages'].items():
        if page_data['num_images'] > 0:
            print("Found images on page", page_num)
            page_index = int(page_num) - 1
            page_image = pages[page_index]
            boxes = [image_info['coords'] for image_info in page_data['images']]
            for x, box in enumerate(boxes):
                annotated_image = draw_bounding_boxes(page_image, box)
                annotate_image_path = f'{annotated_pages_path}/output_page_{page_num}_{x+1}.png'
                save_annotated_image(annotated_image, annotate_image_path, max_width=800, quality=70)
            print("Running GPT4o for description generation")
            for i, image_info in enumerate(page_data['images']):
                annotate_image_path = f'{annotated_pages_path}/output_page_{page_num}_{x+1}.png'
                description = generate_description(annotate_image_path, api_key)
                save_description(description, page_num, i+1,description_path)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python description-generator.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        sys.exit(1)

    # get name of PDF 
    pdf_name = pdf_path.split('/')[-1].split('.pdf')[0]

    # create 2 directories for saving annotated pages and descriptions
    os.makedirs(f'./descriptions_short_{pdf_name}', exist_ok=True)
    os.makedirs(f'./annotatedpages_short_{pdf_name}', exist_ok=True)

    json_file_path = f'./annotation_data_{pdf_name}.json'  # Update with your JSON file path
    
    api_key = 'api-key' # Update with your OpenAI API key

    json_data = load_json(json_file_path)
    
    process_images(json_data, pdf_path, api_key,annotated_pages_path=f'./annotatedpages_{pdf_name}',description_path=f'./descriptions_{pdf_name}')
