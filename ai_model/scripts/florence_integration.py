import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

class Florence2Model:
    def __init__(self, model_id="microsoft/Florence-2-large"):
        print(f"[*] Initializing Florence-2 ({model_id})...")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, trust_remote_code=True
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("[*] Florence-2 ready.")

    def run_inference(self, image_path, task_prompt="<OD>"):
        """
        Run inference using Florence-2.
        Common prompts:
        '<OD>' for Object Detection
        '<CAPTION>' for Image Captioning
        '<DETAILED_CAPTION>' for Detailed Captioning
        """
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            return {"error": str(e)}

        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )

        return parsed_answer

if __name__ == "__main__":
    # Test
    # model = Florence2Model()
    # print(model.run_inference("../data/raw/sample_eagle.jpg", "<OD>"))
    print("[*] Florence-2 Integration Script loaded.")
