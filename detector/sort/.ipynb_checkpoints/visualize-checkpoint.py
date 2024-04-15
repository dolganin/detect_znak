import ast
from sort.util import draf_rectangle_1_percentile
import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(255, 0, 0), thickness=5):
    x1, y1 = top_left
    x2, y2 = bottom_right
    line_lenght = min((abs(x1-x2))//3, (abs(y1-y2))//3)
#     print(line_lenght)
    cv2.line(img, (x1, y1), (x1, y1 + line_lenght), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_lenght, y1), color, thickness)
    
    cv2.line(img, (x1, y2), (x1, y2 - line_lenght), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_lenght, y2), color, thickness)
    
    cv2.line(img, (x2, y1), (x2 - line_lenght, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_lenght), color, thickness)
    
    cv2.line(img, (x2, y2), (x2, y2 - line_lenght), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_lenght, y2), color, thickness)

#     cv2.rectangle(img, top_left, bottom_right, color, thickness)
    return img

def visualize_detections(results_path, video_path):
    results = pd.read_csv(results_path)
    # results = pd.read_csv('./test.csv')
    # load video
    # video_path = 'mmm.mp4'
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))
    print('=============', fourcc, fps, (width, height))
    license_plate = {}
    for car_id in np.unique(results['car_id']):
        try:

            max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
            license_plate[car_id] = {'license_crop': None,
                                     'license_plate_number': results[(results['car_id'] == car_id) &
                                                                     (results['license_number_score'] == max_)]['license_number'].iloc[0]}
#             license_plate[car_id] = {'license_crop': None,
#                                      'license_plate_number':
#                                          results[(results['car_id'] == car_id)]['license_number'].mode()[0]}

            cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                                     (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
            ret, frame = cap.read()

            x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                                      (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            license_crop = cv2.resize(license_crop, (int((x2 - x1) * 50 / (y2 - y1)), 50))

            license_plate[car_id]['license_crop'] = license_crop
        except:
            print('IndexError: single positional indexer is out-of-bounds')
            continue

    frame_nmr = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # read frames
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            df_ = results[results['frame_nmr'] == frame_nmr]
            for row_indx in range(len(df_)):
                # draw car
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)))
                # crop license plate
                license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
                # license_crop = frame[round(y1): round(y2), round(x1): round(x2)]
                # license_crop = cv2.resize(license_crop, ((license_crop.shape[1]*2, license_crop.shape[0]*2)))


                # draw license plate
                try:
                    x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=5)
                except:
                    pass

                H, W, _ = license_crop.shape
                print(H, W, _)
                print(car_x1, car_y1, car_x2, car_y2)
                try:
                    frame[int(car_y1) - H - 50:int(car_y1) - 50,
                          int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

        # white rectangle for background
                    # frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                    #       int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                    (text_width, text_height), _ = cv2.getTextSize(
                        # license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                        str(license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        10)

                    cv2.putText(frame,
                                # license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                str(license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']),
                                (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 100 + (text_height / 2))),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (0, 0, 0),
                                7)

                except:
                    pass
            print('frame_nmr', frame_nmr, frame.shape)
            
            # draw 1% rectangle number plate on image
            frame = draf_rectangle_1_percentile(frame)
            
            out.write(frame)
            frame = cv2.resize(frame, (width, height))

    out.release()
    cap.release()
