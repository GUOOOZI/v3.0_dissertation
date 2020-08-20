import argparse
import os
import platform
import shutil
import time
import numpy
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging, plot_one_arrow, plot_one_circle, arrow_direction)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from A_weed import AStar, MAP

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    arrow = opt.arrow
    navigation = opt.navigation
    merge_number = opt.number
    conf_thres = opt.conf_thres
    if arrow:
        conf_thres = 0.5 * conf_thres

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                if navigation:
                    weed_location = numpy.array([[0], [0], [im0.shape[0]/merge_number], [im0.shape[1]/merge_number]])
                    crop_location = numpy.array([[], [], [], []])
                for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                            if arrow:
                                if conf < 2*conf_thres:
                                    arrow0 = arrow_direction(xywh)
                                    plot_one_arrow(im0, arrow0, xyxy, color=colors[int(cls)], line_thickness=3)
                                elif conf > 2*conf_thres:
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                                    plot_one_circle(im0, xyxy, color=colors[int(cls)], line_thickness=3)

                            else:
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                            if navigation:
                                crcr = (int(xyxy[1] / merge_number), int(xyxy[0] / merge_number),
                                        int(xyxy[3] / merge_number), int(xyxy[2] / merge_number))  # column and raw
                                # print(crcr)
                                if names[int(cls)] == 'weed':
                                    weed_location = numpy.c_[weed_location, crcr]
                                elif names[int(cls)] == 'crop':
                                    crop_location = numpy.c_[crop_location, crcr]
                            # print(crop_location)

                                weed_centres = numpy.array([[], []])

                                for weed in range(weed_location.shape[1]):
                                    weed_centre = ((weed_location[0, weed] + weed_location[2, weed]) / 2,
                                                   (weed_location[1, weed] + weed_location[3, weed]) / 2)
                                    weed_centres = numpy.c_[weed_centres, weed_centre]

                                weed_centres = weed_centres.astype(int)
                                crop_location = crop_location.astype(int)
                                print(weed_centres)
                                for move in range(weed_location.shape[1]):
                                    if weed_location.shape[1] < 2:
                                        break
                                    else:
                                        if move + 1 < weed_location.shape[1]:

                                            weed_now = weed_centres[:, move]
                                            weed_next = weed_centres[:, move + 1]
                                            width = int(im0.shape[1] / merge_number)
                                            height = int(im0.shape[0] / merge_number)
                                            a_star = AStar(weed_now, weed_next, crop_location, width, height)
                                            a_star.main()
                                            way = a_star.path_backtrace()
                                            print('way:', way)
                                            m1 = MAP()
                                            m1.draw_three_axes(a_star)

                                            way_on_img = (way * merge_number).astype(int)
                                            print(way_on_img)

                                            for ways in range(way_on_img.shape[1]):
                                                if way_on_img.shape[1] < 1:
                                                    break
                                                else:
                                                    if ways + 1 < way_on_img.shape[1]:
                                                        way_now = (way_on_img[1, ways], way_on_img[0, ways])
                                                        way_next = (way_on_img[1, ways + 1], way_on_img[0, ways + 1])
                                                        red = (0, 0, 255)
                                                        cv2.line(im0, way_now, way_next, red, thickness=3)
                                        else:
                                            continue


            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--arrow', action='store_true', help='Object position estimation')
    parser.add_argument('--navigation', action='store_true', help='navigation by A star algorithm')
    parser.add_argument('--number', type=int, default=10, help='merged pixels number for navigation')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
