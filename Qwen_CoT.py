import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")

input_file_path = "~/Qwen2-VL-72B-Instruct/input/input_150.json"
output_file_path = "~/Qwen2-VL-72B-Instruct/output/Qwen_CoT.json"

with open(input_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

for entry in data:
    image_path = entry["image"]
    text_prompt = entry["text"]
    reference_translation = entry.get("reference", "")

    # 画像説明の生成
    description_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "~/Qwen2-VL-72B-Instruct/input1/" + image_path},
                {"type": "text", "text": "この画像について説明してください。"}
            ]
        }
    ]

    description_text = processor.apply_chat_template(description_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(description_messages)
    inputs = processor(
        text=[description_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    image_description = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 翻訳の生成
    translation_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"画像の説明: '{image_description}'\nこの文章を参考にして、次の英語の文章を日本語訳してください。: '{text_prompt}'"}
            ]
        }
    ]

    translation_text = processor.apply_chat_template(translation_messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[translation_text],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_translation = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    results.append({
        "image": image_path,
        "image_description": image_description,
        "input_text": text_prompt,
        "generated_translation": output_translation,
        "reference_translation": reference_translation
    })

with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
