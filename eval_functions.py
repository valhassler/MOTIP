import pandas as pd
import numpy as np
#imports second part:
import cv2
import dlib

MINIMAL_SIZE = 20000

# FairFace
def reverse_resized_rect(rect,resize_ratio):
    l = int(rect.left() / resize_ratio)
    t = int(rect.top() / resize_ratio)
    r = int(rect.right() / resize_ratio)
    b = int(rect.bottom() / resize_ratio)
    new_rect = dlib.rectangle(l,t,r,b)
    
    return [l,t,r,b] , new_rect

def resize_image(img, default_max_size=800):
    old_height, old_width, _ = img.shape
    if old_width > old_height:
        resize_ratio = default_max_size / old_width
        new_width, new_height = default_max_size, int(old_height * resize_ratio)
    else:
        resize_ratio = default_max_size / old_height
        new_width, new_height =  int(old_width * resize_ratio), default_max_size
    #img = dlib.resize_image(img, cols=new_width, rows=new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return img, resize_ratio

def extract_faces(img, cnn_face_detector, sp):
    """
    Extracts faces from an image using dlib's cnn_face_detector and shape_predictor
    param img: image to extract faces from
    return: list of extracted faces and their corresponding bboxes
    """
    rects = []
    img, resize_ratio = resize_image(img)
    dets = cnn_face_detector(img, 1) #takes the longest
    num_faces = len(dets)
    faces = dlib.full_object_detections()

    for detection in dets:
        rect = detection.rect
        faces.append(sp(img, rect))
        rect_tpl ,rect_in_origin = reverse_resized_rect(rect,resize_ratio)
        rects.append(rect_in_origin)
    # seems to extract the faces and size them to 300x300
    if len(faces) > 0:
        faces_image = dlib.get_face_chips(img, faces, size=300, padding = 0.25) #predefined
        return faces_image, rects
    else:
        return [], []

def softmax_numpy(logits):
    exps = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exps / np.sum(exps)

def estimate_age_gender_FairFace(img, annotations, specific_arguments):
    cnn_face_detector, sp, model_g_a, trans, device = specific_arguments
    for annotation in annotations:
        x, y, w, h = int(annotation[1]), int(annotation[2]), int(annotation[3]), int(annotation[4])
        cropped_image = img[y:y+h, x:x+w]
        if cropped_image.shape[0]*cropped_image.shape[1] < MINIMAL_SIZE: #20000 for going_out, 200 for top
            continue
        faces_image, rects = extract_faces(cropped_image, cnn_face_detector, sp)
        # Now prediction of the images
        #zoom on one image
        observations = []
        for i, image in enumerate(faces_image):
            image = trans(image)
            image = image.view(1, 3, 224, 224) 
            image = image.to(device)
            outputs = model_g_a(image)

            outputs = model_g_a(image)
            outputs = outputs.cpu().detach().numpy()
            outputs = np.squeeze(outputs)
            ## Postprocessing 
            #race_outputs = outputs[:7]
            gender_outputs = outputs[7:9]
            age_outputs = outputs[9:18]

            #race_score = softmax_numpy(race_outputs)
            gender_score = softmax_numpy(gender_outputs)
            age_score = softmax_numpy(age_outputs)

            gender_pred = np.argmax(gender_score)
            age_pred = np.argmax(age_score)
            observations.append([gender_pred, age_pred,
                                gender_score, age_score])

        if len(observations) == 0:
            return
        else:
            result = pd.DataFrame(observations)
            result.columns = ['gender_preds','age_preds',
                                'gender_scores','age_scores']
            #bboxes
            # Mapping for gender predictions
            #in case of doubt take the older one
            gender_mapping = {0: 'Male', 1: 'Female'}
            age_mapping = {
                0: '0-2', 1: '3-9', 2: '10-19', 3: '20-29',
                4: '30-39', 5: '40-49', 6: '50-59', 7: '60-69', 8: '70+'}
            result['gender_preds'] = result['gender_preds'].map(gender_mapping)
            result['age_preds'] = result['age_preds'].map(age_mapping)
        
            annotation.append(result['gender_preds'][0]) 
            annotation.append(result['age_preds'][0])
# MiVOLO
def estimate_age_gender_MiVolo(image, annotations, specific_arguments):
    predictor = specific_arguments[0]
    for annotation in annotations:
        x, y, w, h = int(annotation[1]), int(annotation[2]), int(annotation[3]), int(annotation[4])
        cropped_image = image[y:y+h, x:x+w]
        if cropped_image.shape[0]*cropped_image.shape[1] < MINIMAL_SIZE: #20000 for going_out, 200 for top
            continue

        detected_objects, _ = predictor.recognize(cropped_image)
        if detected_objects.n_persons == 0:
            annotation.append(None)  # No gender
            annotation.append(None)  # No age
        else:
            annotation.append(detected_objects.genders[0])  # Gender
            annotation.append(np.mean(detected_objects.ages))  # Age