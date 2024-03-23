import io
import numpy as np
import cv2
import onnxruntime
from _paths import detect_znak_dir
from detect_znak import pipeline
from detect_znak.tools import unzip
from detect_znak.image_loaders import DumpyImageLoader

class Detector():

    def __init__(self) -> None:
        """
        Initialize Detector object
        
        Args:
            None
        Returns:
            None
        """
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
         confidences, predicted_image_texts) = unzip(result)
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

    def __call__(self, rgb_img):

        return self.post_process(self.forward(rgb_img))
    
if __name__ == '__main__':
    det = Detector()
#     img = cv2.imread('../datasets/ru_test/М052НН69_А052КО69.jpeg')
    img = cv2.imread('28.jpeg')
    image, image_zones, texts = det(img)
    for i in range(len(image_zones)):
        cv2.imwrite(f'zone{i}.jpg', image_zones[i])
    cv2.imwrite('image.jpg', image)
    print(texts)
    