import sys
sys.path.insert(0, './yolov5')
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import threading
from threading import Thread
import os
from queue import Queue
import sys
import matplotlib.mlab as mlab
import pyaudio
from keras.models import load_model

os.environ['KMP_DUPLICATE_LIB_OK']='True'
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

global track_modes, cli, height, width, cli_time, keyword, que,t

keyword=""
track = {}
cli =0
cli_time=0


data_c = None
# Use 1101 for 2sec input audio
Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

# Use 272 for 2sec input audio
Ty = 1375# The number of time steps in the output of our model

model = load_model('./keyword_spotting/tr_model_t.h5')

def detect_triggerword_spectrum(x):
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    return predictions.reshape(-1)

def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold=0.5):
    predictions = predictions > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred
    return False

"""# Record audio stream from mic"""
chunk_duration = 0.5 # Each read length in seconds from mic.
fs = 44100 # sampling rate for mic
chunk_samples = int(fs * chunk_duration) # Each read length in number of samples.

# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_duration = 10
feed_samples = int(fs * feed_duration)

assert feed_duration/chunk_duration == int(feed_duration/chunk_duration)
def get_spectrogram(data):
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, _, _ = mlab.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

"""### Audio stream"""

def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=0,
        stream_callback=callback)
    return stream

# Queue to communiate between the audio callback and main thread
q = Queue()
que = Queue()
run = True
silence_threshold = 100
timeout = 2*60  # 0.1 minutes from now
# Data buffer for the input wavform
data = np.zeros(feed_samples, dtype='int16')

def callback(in_data, frame_count, time_info, status):
    global run, timeout, data, silence_threshold
    if time.time() > timeout:
        run = False
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        sys.stdout.write('------------dddd-----')
        return (in_data, pyaudio.paContinue)
    else:
        sys.stdout.write('.............dddd....')
    data = np.append(data,data0)
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)

def check_where():
    stream = get_audio_input_stream(callback)
    stream.start_stream()
    count=0
    global run, timeout,s
    try:
        while count<timeout:
            data = q.get()
            spectrum = get_spectrogram(data)
            preds = detect_triggerword_spectrum(spectrum)
            new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
            if new_trigger:
                print('I CAN HEAR TRIGGER##################')
                que.put('RESTART TRACKING')
            else:
                print('I CAN HEAR NOTHING$$$$$$$$$$$$$$$$$$')
                que.put('HEAR NOTHING')
            #time.sleep(1)
            count = count+1
    except (KeyboardInterrupt, SystemExit):
        stream.stop_stream()
        stream.close()
        timeout = time.time()
        run = False
    stream.stop_stream()
    stream.close()


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

#class mapping-> 객체 받는식

def draw_boxes_after_no(img, bbox, identities=None, offset=(0,0)):
    global cli_time,track_modes, track, cli,t
    cli_time += 1
    if (cli_time < 20):
        if (cli in identities):
            track_modes=2
            return draw_boxes_after_yes(img, bbox, identities, offset)
        else:
            return draw_boxes_after_yes(img, bbox, identities, offset)
    if (cli_time == 20):
        t = threading.Thread(target=check_where)
        t.start()
        # t.join()
        cv2.putText(img, "Client Missing! Listening...", (int(width / 5), int(height / 9)),cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 0], 10)
        return draw_boxes_plain(img, bbox,identities, offset)
    if (cli_time > 20 and cli_time <110):
        cv2.putText(img, "Listening the word...", (int(width / 5), int(height / 9)),cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 0], 10)
        string = que.get()
        if(string is 'RESTART TRACKING'):
            track_modes = 1
            cli = 0
            for i in range(1, 100):
                track[i] = 0
            cv2.putText(img, 'RESTART TRACKING ', (int(width / 3), int(height / 4)), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 0], 10)
            return draw_boxes_before(img, bbox, identities, offset)
        else:
            cv2.putText(img,'HEAR NOTHING ', (int(width/3), int(height / 4)),cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 0], 10)
            return draw_boxes_plain(img, bbox, identities, offset)
    if (cli_time >= 110):
        track_modes =1
        cli=0
        for i in range(1, 100):
            track[i] = 0
        return draw_boxes_before(img, bbox, identities, offset)
    return img


def draw_boxes_plain(img, bbox, identities=None, offset=(0,0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        track[id] = track[id] + 1
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def draw_boxes_before(img, bbox, identities=None, offset=(0,0)):
    global track_modes, cli, track
    if(track_modes==1):
        cv2.putText(img, "Tracking Client...", (int(width/3), int(height/9)), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 0], 10)
    if(track_modes==3):
        cv2.putText(img, "Client Missing! Start Finding", (int(width / 5), int(height / 9)),cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 0], 10)
    for i in range(1, 100):
        if (track[i] >= 500):
            track_modes = 2
            cli = i
    print("Finding Client client is " + str(cli))
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        track[id] = track[id] + 1
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def draw_boxes_after(img, bbox, identities=None, offset=(0,0)):
    global cli_time, track_modes
    if(cli in identities):  # detect 잘되면
        cli_time=0
        track_modes=2
        return draw_boxes_after_yes(img, bbox, identities, offset)
    else:                   # detect 안되
        track_modes=3
        return draw_boxes_after_no(img, bbox, identities, offset)

def draw_boxes_after_yes(img, bbox, identities=None, offset=(0,0)):
    global cli_time, track_modes
    if(track_modes==2):
        cli_time=0
        cv2.putText(img, "Client Detected! Following...",  (int(width/5), int(height/9)), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 0], 10)
    if(track_modes==3):
        cv2.putText(img, "Client Missing! Start Finding " + str(cli_time), (int(width / 5), int(height / 9)),cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 0], 10)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        track[id] = track[id] + 1
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        if (id == cli):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 6)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 100, y1 + t_size[1] + 8), (0, 0, 255), -1)
            cv2.putText(img, "CLIENT", (x1, y1 +t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            cv2.ellipse(img, (int(width/2), int(height/10*9)), (300, 300), 0, 180, 360, (255, 255, 255), -1)
            d = pow(pow(int(width/2)-int((x1 + x2) / 2),2)+pow(int(height/10*9)- int((y1 + y2) / 2),2),1/2)
            line_x = int(width/2)+(int((x1 + x2) / 2)-int(width/2))*300/d
            line_y = int(height/10*9)+ (int((y1 + y2) / 2) - int(height/10*9)) * 300 / d
            cv2.arrowedLine(img, (int(width/2), int(height/10*9)), (int(line_x), int(line_y)), (0, 0, 255), 10, 8, 0, 0.1)
        else:
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return img


def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    global track_modes
                    if(track_modes==1):
                        draw_boxes_before(im0, bbox_xyxy, identities)
                    if (track_modes==2):
                        draw_boxes_after(im0, bbox_xyxy, identities)
                    if (track_modes==3):
                        draw_boxes_after_no(im0, bbox_xyxy, identities)

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                global height, width
                height = im0.shape[0]
                width = im0.shape[1]
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)
    pool = ProcessPoolExecutor(2)
    for i in range(1, 100):
        track[i]=0
    cli = 0;
    track_modes= 1
    with torch.no_grad():
        detect(args)