#basically the same as above but now also gender and age estiamtions are made. If used in future clean it and merge it

from tqdm import tqdm, trange
import os
#imports second part:
import torch
import csv
import cv2
import decord
from torch.utils.data import Dataset
import torchvision

import dlib
from eval_functions import estimate_age_gender_MiVolo, estimate_age_gender_FairFace, estimate_age_gender_AgeSelf
from mivolo.predictor import Predictor
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")



class VideoDataset(Dataset):
    def __init__(self, video_path):
        """
        Args: video_path (str): Path to the video file
        """
        # Initialize the VideoReader
        self.vr = decord.VideoReader(video_path, ctx=decord.cpu(0))  # Load video in CPU memory
        self.length = len(self.vr)  # Total number of frames
        self.video_size = self.vr[0].asnumpy().shape
    
    def load_annotations(self, annotation_path):
        annotations = {}
        with open(annotation_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                frame_idx = int(row[0]) - 1  # Assuming frame indices in the file start from 1
                annotation = list(map(float, row[1:]))  # Convert the rest of the row to floats
                if frame_idx not in annotations:
                    annotations[frame_idx] = []
                annotations[frame_idx].append(annotation)
        return annotations
    
    def split_frame(self, np_array):
        entire_image = np_array
        top_view = entire_image[0:540, 62:892]  # Crop from top view
        coming_in_view = entire_image[540:1500, 0:540]  # Crop from left side
        going_out_view = entire_image[540:1500, 540:1080]  # Crop from right side
        return top_view, coming_in_view, going_out_view

    def draw_annotations(self, frame, annotations, frame_number):
        """
        All annotations for bbox and age gender

        """
        cv2.putText(frame, f"{frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        for annotation in annotations:
            obj_id, x, y, w, h, gender, age = int(annotation[0]), annotation[1], annotation[2], annotation[3], annotation[4], annotation[-2], annotation[-1]
            top_left = (int(x), int(y))
            bottom_right = (int(x + w), int(y + h))
            gender = gender if gender !=-1 else None
            age = age if age !=-1 else None
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj_id}', (int(x), int(y - 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
            cv2.putText(frame, f'Age: {age}', (int(x), int(y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
            cv2.putText(frame, f'Gender: {gender}', (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
        return frame
    
    def save_annotated_video(self, output_path, annotation_path, view, predictor_age_gender,age_gender_est_func, age_gender_estimation=True):
        """
        ## Args:
        - output_path (str): Path to save the annotated video
        - annotation_path (str): Path to the annotation file
        - view (str): One of 'top', 'coming_in', or 'going_out'
        - age_gender_est_func: function that uses image, annotations and predictor, estimates age and gender
        and puts it in the annotations to the bounding box as additional information
        - predictor:  Initilized model for the age and gender prediciton
        """
        annotations = self.load_annotations(annotation_path)
        age_gender_basepath = "/".join(os.path.dirname(annotation_path).split("/")[:-1]) + "/tracker_age_gender"
        os.makedirs(age_gender_basepath, exist_ok=True)
        annotation_with_age_gender_path = os.path.join(age_gender_basepath,os.path.basename(annotation_path).split(".")[0] + "_age_gender.txt")
        
        if view == 'top':
            width, height = 830, 540
        elif view == 'coming_in':
            width, height = 540, 960
        elif view == 'going_out':
            width, height = 540, 960
        else:
            #raise ValueError("View must be one of 'top', 'coming_in', or 'going_out'")
            view = "entire"
            height, width = self.video_size[0], self.video_size[1]
        
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        
        for idx in trange(self.length):
            frame = self.vr[idx].asnumpy()
            if view != 'entire':
                top_view, coming_in_view, going_out_view = self.split_frame(frame)
                if view == 'top':
                    selected_view = top_view
                elif view == 'coming_in':
                    selected_view = coming_in_view
                elif view == 'going_out':
                    selected_view = going_out_view
            else:
                selected_view = frame
                
            frame_annotations = annotations.get(idx, [])
            if idx < 150000 and age_gender_estimation:
                age_gender_est_func(selected_view, frame_annotations, predictor_age_gender)
            annotated_frame = self.draw_annotations(selected_view, frame_annotations, idx)
            out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            if age_gender_estimation is not True:
                continue
            # Write the annotations to a file
            if idx == 0 and os.path.exists(annotation_with_age_gender_path):
                os.remove(annotation_with_age_gender_path)
            if len(annotation_with_age_gender_path) > 0:
                for annotation in frame_annotations:
                    indices = [0, 1, 2, 3, 4, -2, -1]
                    content_to_write = str(idx) + " " +' '.join(str(annotation[i]) for i in indices)
                    with open(annotation_with_age_gender_path, 'a') as file:
                        file.write(content_to_write + '\n')
        out.release()

    def __getitem__(self, idx):
        frame = self.vr[idx].asnumpy()
        top_view, coming_in_view, going_out_view = self.split_frame(frame)
        return top_view, coming_in_view, going_out_view, frame

    def __len__(self):
        return self.length

# Usage example




#video_name = "2024_05_04_10_57_26"
video_name = "2024_05_19_10_51_24"

video_path = f"/usr/users/vhassle/datasets/Wortschatzinsel/Neon_complete/Neon/{video_name.replace('_','-')}/{video_name}.mp4"
annotation_path = f"/usr/users/vhassle/psych_track/MOTIP/outputs/Wortschatzinsel/Neon_test/tracker/{video_name}.mp4.txt"

output_path = f'/usr/users/vhassle/psych_track/MOTIP/outputs/{os.path.basename(annotation_path).split(".")[0]}_AgeSelf_pad.mp4'#_MiVOLO.mp4'


# estimate_age_gender_MiVolo
# Initialize Predictor
model_weights_path = '/usr/users/vhassle/psych_track/AgeSelf/models/age_classification_model_20_focal_pad.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model_age = torchvision.models.resnet50(weights=None)
num_ftrs = model_age.fc.in_features
model_age.fc = torch.nn.Linear(num_ftrs, 3)
model_age.load_state_dict(torch.load(model_weights_path))
model = model_age.to(device)
model.eval()
specific_arguments = [model]
dataset = VideoDataset(video_path)

dataset.save_annotated_video(output_path, annotation_path, "quatsch", specific_arguments, estimate_age_gender_AgeSelf, age_gender_estimation=True)

# # estimate_age_gender_MiVolo
# # Initialize Predictor
# class Args:
#     def __init__(self):
#         self.detector_weights = "/usr/users/vhassle/psych_track/MiVOLO/models/yolov8x_person_face.pt"
#         self.checkpoint = "/usr/users/vhassle/psych_track/MiVOLO/models/mivolo_imbd.pth.tar"
#         self.with_persons = True
#         self.disable_faces = False
#         self.draw = False
#         self.device = "cuda"

# args = Args()
# predictor = Predictor(args, verbose=False)
# specific_arguments = [predictor]
# dataset = VideoDataset(video_path)

# dataset.save_annotated_video(output_path, annotation_path, "quatsch", specific_arguments, estimate_age_gender_MiVolo, age_gender_estimation=False)

# # estimate_age_gender_FairFace
# cnn_face_detector = dlib.cnn_face_detection_model_v1('/usr/users/vhassle/psych_track/FairFace/dlib_models/mmod_human_face_detector.dat')
# sp = dlib.shape_predictor('/usr/users/vhassle/psych_track/FairFace/dlib_models/shape_predictor_5_face_landmarks.dat')
# model_path = "/usr/users/vhassle/psych_track/FairFace/fair_face_models/res34_fair_align_multi_7_20190809.pt"

# trans = torchvision.transforms.Compose([
#     torchvision.transforms.ToPILImage(),
#     torchvision.transforms.Resize((224, 224)),
#     torchvision.transforms.ToTensor(),
#     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                      std=[0.229, 0.224, 0.225])
# ])

# device = "cuda:0"
# model_fair_7 = torchvision.models.resnet34(weights=True)
# model_fair_7.fc = torch.nn.Linear(model_fair_7.fc.in_features, 18)
# model_fair_7.load_state_dict(torch.load(model_path))
# model_fair_7 = model_fair_7.to(device)
# model_fair_7.eval()


# specific_arguments = [cnn_face_detector, sp, model_fair_7, trans, device]

# dataset = VideoDataset(video_path)
# dataset.save_annotated_video(output_path, annotation_path, "non_specific_view", specific_arguments, estimate_age_gender_FairFace, age_gender_estimation=True)

