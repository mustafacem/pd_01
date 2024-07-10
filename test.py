import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
import numpy as np
from doc_creation.doc_creation_c import *

from ocr_processing.ocr_processing_c import *

# Load the processor and model



#create_document(r"C:\Users\cem\Desktop\cekya_ıs\s\proposal_droid\Screenshot 2024-05-23 171557.png",items_dict )


#preprocess_handwritten_image

#extract_text_from_image
#preprocess_handwritten_image(r"C:\Users\cem\Desktop\cekya_ıs\s\proposal_droid\IMG_7506.png")
#preprocess_handwritten_image("IMG_7506.png")
#sen = extract_text_from_image('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',"IMG_7506.png")
#print(sen)

asd = load_model_and_predict("IMG_7506.png")
print(asd)
