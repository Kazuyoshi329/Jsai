import json
from openai import OpenAI
import base64

client = OpenAI(api_key="xxxxx")
MODEL = "gpt-4o-2024-08-06"

input_file_path = "~/Qwen2-VL-72B-Instruct/input1/input_150.json"
output_file_path = "~/Qwen2-VL-72B-Instruct/output/GPT_CoT.json"
input_backup_path = "~/GPT/input/input01.json"
output_backup_path = "~/GPT/output/output01.json"

# 画像をBase64形式にエンコードする関数
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

with open(input_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(input_backup_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

results = []

for entry in data:
    image_path = entry["image"]
    text_prompt = entry["text"]
    reference_translation = entry.get("reference", "")
    base64_image = encode_image("~/Qwen2-VL-72B-Instruct/input/" + image_path)
    image_url = "data:image/jpg;base64," + base64_image

    # 画像説明の生成
    try:
        description_response = client.chat.completions.create(model=MODEL,
        messages=[
            {"role": "user","content": [
                {"type": "text", "text": "この画像について説明してください"},
                {"type": "image_url", "image_url": {
                    "url": image_url}
                }
            ]}
        ])
        image_description = description_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating image description for {'~/Qwen2-VL-72B-Instruct/input1/' + image_path}: {e}")
        image_description = ""

    # 翻訳の生成
    try:
        translation_response = client.chat.completions.create(model=MODEL,
        messages=[
            {
                "role": "user","content": [
                    {"type": "text", "text": f"画像の説明: '{image_description}'\nこの説明を参考にして、次の英語の文章を日本語訳してください。: '{text_prompt}'"}
                ]
            }
        ])
        output_translation = translation_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating translation for {text_prompt}: {e}")
        output_translation = ""

    results.append({
        "image": image_path,
        "image_description": image_description,
        "input_text": text_prompt,
        "generated_translation": output_translation,
        "reference_translation": reference_translation
    })

with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

with open(output_backup_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

