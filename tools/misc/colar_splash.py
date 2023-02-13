import os.path as osp
import sys
import cv2
import mmcv
import numpy as np
from tqdm import tqdm
from mmdet.apis import init_detector,inference_detector

config_file = 'configs/htc/htc_without_semantic_r50_fpn_1x_balloon.py'
checkpoint_file = 'work_dirs/htc_without_semantic_r50_fpn_1x_balloon/epoch_12.pth'




def main():
    src = 'data/test_video.mp4'
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    video = mmcv.VideoReader(src)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_write = cv2.VideoWriter('data/color_splash_balloon.mp4',fourcc,
                                  video.fps, (video.width, video.height))
    for frame in tqdm(video):
        result = inference_detector(model,frame)
        masks = result[1][0]
        map = np.full(masks[0].shape,False)
        for index, mask in enumerate(masks):
            if result[0][0][index][-1] > 0.8:
                map = (map | mask)
        #整张图的一个mask
        frame_mask = map * frame.transpose(2,0,1)
        
        #转为灰度图
        un_mask = (1 - map).astype(np.bool)
        frame_unmask = un_mask * frame.transpose(2,0,1)
        frame_gray = cv2.cvtColor(frame_unmask.transpose(1,2,0), cv2.COLOR_BGR2GRAY)
        
        #将mask放入图像中
        frame_color_gray = frame_gray + frame_mask
        frame_color_gray = frame_color_gray.transpose(1,2,0)

        video_write.write(frame_color_gray)
    video_write.release()
    cv2.destroyAllWindows()
        





        
        






if __name__ == '__main__':
    main()