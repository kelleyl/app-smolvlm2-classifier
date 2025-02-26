import argparse
import logging
import yaml
from pathlib import Path
import tqdm
import time
from PIL import Image

from clams import ClamsApp, Restifier
from clams.appmetadata import AppMetadata
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch


class SmolVLM2Classifier(ClamsApp):

    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct", 
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        super().__init__()

    def _appmetadata(self) -> AppMetadata:
        pass
    
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.logger.debug(f"Annotating with parameters: {parameters}")
        config_file = parameters.get('config')
        self.logger.info(f"Using config file: {config_file}")
        config_dir = Path(__file__).parent
        config_file = config_dir / config_file
        config = self.load_config(config_file)
        
        frame_interval = parameters.get('frameInterval', 30)
        classifier_prompt = parameters.get('classifierPrompt', 
                                        'Classify what is shown in this video frame. Output only the primary category that best describes the content.')
        
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(AnnotationTypes.TimePoint)
        new_view.new_contain(DocumentTypes.TextDocument)

        # Get first video document
        video_documents = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        if not video_documents:
            self.logger.error("No video documents found in the input MMIF")
            return mmif
            
        video_doc = video_documents[0]
        fps = float(video_doc.get_property('fps'))
        total_frames = int(video_doc.get_property('frameCount'))
        
        # Calculate frames to process
        frame_numbers = list(range(0, total_frames, frame_interval))
        self.logger.info(f"Processing {len(frame_numbers)} frames at interval {frame_interval}")
        
        # Process frames
        for frame_number in tqdm.tqdm(frame_numbers):
            try:
                # Extract frame as image
                image = vdh.extract_frame_as_image(video_doc, frame_number, as_PIL=True)
                
                # Prepare multimodal input for SmolVLM2 using chat template
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": classifier_prompt},
                            {"type": "image", "image": image},
                        ]
                    }
                ]
                
                # Process through model
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device)
                
                # Generate response
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=50,  # Shorter for classification outputs
                    min_length=1,
                    temperature=0.7,
                )
                
                # Decode output
                classification = self.processor.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                )[0].strip()
                
                # Create text document for the classification result
                text_document = new_view.new_textdocument(classification)
                
                # Create timepoint annotation
                timepoint = new_view.new_annotation(AnnotationTypes.TimePoint)
                timepoint.add_property("timePoint", frame_number / fps)  # TimePoint in seconds
                timepoint.add_property("frameNumber", frame_number)
                timepoint.add_property("classification", classification)
                timepoint.add_property("document", text_document.id)
                
                self.logger.debug(f"Frame {frame_number} classified as: {classification}")
                
            except Exception as e:
                self.logger.error(f"Error processing frame {frame_number}: {str(e)}")
            finally:
                torch.cuda.empty_cache()
        
        return mmif

    
def get_app():
    return SmolVLM2Classifier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    app = SmolVLM2Classifier()

    http_app = Restifier(app, port=int(parsed_args.port))

    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
