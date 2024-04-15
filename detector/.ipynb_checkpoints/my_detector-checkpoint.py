import io
import numpy as np
import cv2
import onnxruntime
from sort.sort import *
from ultralytics import YOLO
from _paths import detect_znak_dir
from detect_znak import pipeline
from detect_znak.tools import unzip
from detect_znak.image_loaders import DumpyImageLoader
from sort.sort import *
from sort.visualize import visualize_detections
from sort.util import get_car, read_license_plate, write_csv
from sort.add_missing_data import interp

        

class Detector():

    def __init__(self) -> None:
        """
        Initialize Detector object
        
        Args:
            None
        Returns:
            None
        """
#         self.yolo = YOLO("yolov8n.pt")
        path = os.path.join(detect_znak_dir, "detector")
        yolo_path = os.path.join(path, "yolov8n.onnx")
        self.yolo = YOLO(yolo_path)
        self.pipeline = pipeline("multiline_number_plate_detection_and_reading",image_loader=DumpyImageLoader)

    
    def forward(self, rgb_img):
        """
        Perform forward pass on the input image.

        Args:
            rgb_img (numpy.ndarray): Input RGB image.

        Returns:
            images (list): Original images
            images_bboxs (list): Bboxes predicted by YOLO
            images_points (list): Points predicted by CRAFT
            images_zones (list): Crop image(s) of predicted region(s)
            region_ids (list): Ids of predicted region class
            region_names (list): Names of predicted region class
            count_lines (list): count of predicted instances
            confidences (list): confidence scores for predictions
            predicted_image_texts (list): Predicted texts
        """
        return self.pipeline([rgb_img])
    
    def post_process(self, result):
        """
        Perform forward pass on the input image.

        Args:
            result (list): output from forward pass

        Returns:
            image (numpy.ndarray): Image with predictions drawn
            image_zones (list): Crop image(s) of predicted region(s)
            predicted_image_texts (list): Predicted texts
        """
        (images, images_bboxs, 
         images_points, images_zones, region_ids, 
         region_names, count_lines, 
         confidences, predicted_image_texts, read_confs) = unzip(result)
        for predicted_image_texts, \
            image, image_bboxs, \
            image_points, image_zones, \
            image_region_ids, image_region_names, \
            image_count_lines, \
            image_confidences in zip(predicted_image_texts,
                            images, images_bboxs,
                            images_points, images_zones,
                            region_ids, region_names,
                            count_lines, confidences):


            image = image.astype(np.uint8)
            for cntr in image_points:
                cntr = np.array(cntr, dtype=np.int32)
                try:
                    cv2.drawContours(image, [cntr], -1, (0, 0, 255), 2)
                except:
                    continue

            target_pixels = int(image.shape[0] * image.shape[1] * 0.01)

            check_color = (0, 255, 0)
            square_size = int(np.sqrt(target_pixels))
            rectangle_width = int(np.sqrt(target_pixels * 4.642857142857143))
            rectangle_height = int(np.sqrt(target_pixels / 4.642857142857143))
            square_width = int(np.sqrt(target_pixels * 1.705882353))
            square_height = int(np.sqrt(target_pixels / 1.705882353))

            cv2.rectangle(image, (0, 0), (square_width, square_height), check_color, 1)  
            cv2.rectangle(image, (0, 0), (rectangle_width, rectangle_height), check_color, 1)  
            cv2.putText(image, '1%', (10, round(image.shape[0]/25)), cv2.FONT_HERSHEY_PLAIN, (image.shape[0]/300), check_color, 1)

            for target_box in image_bboxs:
                cv2.rectangle(image,
                              (int(target_box[0]), int(target_box[1])),
                              (int(target_box[2]), int(target_box[3])),
                              (0, 255, 0),
                              1)
        return image, image_zones, predicted_image_texts, confidences, image_region_ids
    
    def _get_bbox_from_craft(self, cntr):
        x1, y1 = min(cntr[::2]), min(cntr[1::2])
        x2, y2 = max(cntr[::2]), max(cntr[1::2])
        return [x1, y1, x2, y2]
    
    def _get_bbox_from_detections(self, result):
        (images, images_bboxs, 
         images_points, images_zones, region_ids, 
         region_names, count_lines, 
         confidences, predicted_image_texts) = unzip(result)

        return images_bboxs
    def video_call(self, vid, interpolate=False):
        results = {}

        mot_tracker = Sort()
        cap = cv2.VideoCapture(vid)
        ret = True
        frame_nmr = -1
        vehicles = [2, 3, 5, 7]
        while ret:
            frame_nmr += 1
            ret, frame = cap.read()

            if ret:
                results[frame_nmr] = {}
                # detect vehicles
                detections = self.yolo(frame)[0] 
                detections_ = []
                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in vehicles:
                        detections_.append([x1, y1, x2, y2, score])

                # track vehicles
                try:
                    track_ids = mot_tracker.update(np.asarray(detections_))
                except:
                    print('IndexError: index 1 is out of bounds for axis 1 with size 1')
                    continue
                # detect license plates #CURRENT
                result = self.forward(frame)
                (images, images_bboxs, 
                 images_points, images_zones, region_ids, 
                 region_names, count_lines, 
                 confidences, predicted_image_texts, read_confs) = unzip(result)
#                 print(np.array(images_bboxs).shape)
                
                for i, license_plate in enumerate(images_bboxs[0]):
                    
                    x1, y1, x2, y2, score, class_id = license_plate

                    # assign license plate to car
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                    print(np.array(images_bboxs[0]).shape)
                    if car_id != -1:

                        # crop license plate
                        
#                         license_plate_crop = images_zones[0][i]

                        # process license plate
#                         license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#                         _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                        if i >= len(predicted_image_texts[0]):
                            continue
                        # read license plate number
                        print(predicted_image_texts, read_confs, )
                        license_plate_text, license_plate_text_score = predicted_image_texts[0][i], read_confs[0][i]
                        print('===', license_plate_text, license_plate_text_score)
                        if license_plate_text is not None:
                            print(frame_nmr, 'LICENSE_PLATE', license_plate_text)
                            results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                          'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                            'text': license_plate_text,
                                                                            'bbox_score': score,
                                                                            'text_score': license_plate_text_score}}
                if not interpolate:
                    if frame_nmr !=0:
                        for res in track_ids:
                            xcar1, ycar1, xcar2, ycar2, car_id = res
                            prev = results[frame_nmr-1].get(car_id, None)
                            now = results[frame_nmr].get(car_id, None)
                            if prev is not None and now is None:
                                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                              'license_plate': {'bbox': [],
                                                                                'text': prev['license_plate']['text'],
                                                                                'bbox_score': prev['license_plate']['bbox_score'],
                                                                                'text_score': prev['license_plate']['text_score']}}
        # write results
        write_csv(results, './test.csv')
        if interpolate:
            file = interp('./test.csv')
        else:
            file = "./test.csv"
        visualize_detections(file, vid)
        return results
        
    def __call__(self, rgb_img):
        return self.post_process(self.forward(rgb_img))