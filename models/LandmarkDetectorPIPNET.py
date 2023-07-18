import importlib
import torchvision.transforms as transforms
import torch.nn as nn
import os
import torchvision.models as models
import dlib

#from models.PIPNET.FaceBoxesV2.faceboxes_detector import *
from models.PIPNET.lib.networks import *
from models.PIPNET.lib.functions import *


class LandmarkDetectorPIPENET(nn.Module):
    def __init__(self,
                 source_format: str = "data_300W_COFW_WFLW",
                 experiment: str = "pip_32_16_60_r18_l2_l1_10_1_nb10_wcc",
                 device: str = 'cuda'
    ):
        super(LandmarkDetectorPIPENET, self).__init__()
        config_path = 'models.PIPNET.experiments.{}.{}'.format(source_format, experiment)
        config = importlib.import_module(config_path, package='PIPNet')
        Config = getattr(config, 'Config')
        self.cfg = Config()
        self.cfg.experiment_name = experiment
        self.cfg.data_name = source_format

        self.device = device

        # Get face detector
        self.det_face_thresh = 0.5
        self.det_box_scale = 1.2

        self.face_detector = dlib.get_frontal_face_detector()

        # Get landmark detector
        _, self.reverse_index1, self.reverse_index2, self.max_len = get_meanface(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PIPNET/data', self.cfg.data_name, 'meanface.txt'),
            self.cfg.num_nb)

        if self.cfg.backbone == 'resnet18':
            resnet18 = models.resnet18(pretrained=self.cfg.pretrained)
            self.net = Pip_resnet18(resnet18, self.cfg.num_nb, num_lms=self.cfg.num_lms, input_size=self.cfg.input_size,
                                    net_stride=self.cfg.net_stride)
        elif self.cfg.backbone == 'resnet50':
            resnet50 = models.resnet50(pretrained=self.cfg.pretrained)
            self.net = Pip_resnet50(resnet50, self.cfg.num_nb, num_lms=self.cfg.num_lms, input_size=self.cfg.input_size,
                                    net_stride=self.cfg.net_stride)
        elif self.cfg.backbone == 'resnet101':
            resnet101 = models.resnet101(pretrained=self.cfg.pretrained)
            self.net = Pip_resnet101(resnet101, self.cfg.num_nb, num_lms=self.cfg.num_lms, input_size=self.cfg.input_size,
                                     net_stride=self.cfg.net_stride)
        else:
            print('No such backbone!')
            exit(0)

        weight_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'../data/checkpoints/pipnet/{self.cfg.data_name}/{self.cfg.experiment_name}/epoch%d.pth' % (self.cfg.num_epochs - 1))
        state_dict = torch.load(weight_file, map_location=device)
        self.net = self.net.to(device)
        self.net.load_state_dict(state_dict)

        # Get data related settings
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose(
            [transforms.Resize((self.cfg.input_size, self.cfg.input_size)), transforms.ToTensor(), self.normalize])

    def forward(self, image: np.ndarray):
        detections = self.face_detector(image, 1)
        if len(detections) == 0:
            return None
        detection = detections[0]
        det_xmin = detection.left()
        det_ymin = detection.top()
        det_width = detection.right() - detection.left()
        det_height = detection.bottom() - detection.top()
        det_xmax = det_xmin + det_width - 1
        det_ymax = det_ymin + det_height - 1

        det_xmin -= int(det_width * (self.det_box_scale - 1) / 2)
        # remove a part of top area for alignment, see paper for details
        det_ymin += int(det_height * (self.det_box_scale - 1) / 2)
        det_xmax += int(det_width * (self.det_box_scale - 1) / 2)
        det_ymax += int(det_height * (self.det_box_scale - 1) / 2)
        det_xmin = max(det_xmin, 0)
        det_ymin = max(det_ymin, 0)
        det_xmax = min(det_xmax, image.shape[1] - 1)
        det_ymax = min(det_ymax, image.shape[0] - 1)
        det_width = det_xmax - det_xmin + 1
        det_height = det_ymax - det_ymin + 1

        det_crop = image[det_ymin:det_ymax + 1, det_xmin:det_xmax + 1, :]
        det_crop = cv2.resize(det_crop, (self.cfg.input_size, self.cfg.input_size))
        det_crop = Image.fromarray(det_crop[:, :, ::-1].astype('uint8'), 'RGB')
        det_crop = self.preprocess(det_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(self.net, det_crop,
                                                                                                 self.preprocess,
                                                                                                 self.cfg.input_size,
                                                                                                 self.cfg.net_stride,
                                                                                                 self.cfg.num_nb)
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[self.reverse_index1, self.reverse_index2].view(self.cfg.num_lms, self.max_len)
        tmp_nb_y = lms_pred_nb_y[self.reverse_index1, self.reverse_index2].view(self.cfg.num_lms, self.max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()
        # cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
        # for i in range(self.cfg.num_lms):
        #     x_pred = lms_pred_merge[i * 2] * det_width
        #     y_pred = lms_pred_merge[i * 2 + 1] * det_height
        #     cv2.circle(image, (int(x_pred) + det_xmin, int(y_pred) + det_ymin), 1, (0, 0, 255), 2)
        # cv2.imshow('image', cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4)))
        # cv2.waitKey(0)
        lms_pred_merge = np.reshape(lms_pred_merge, (self.cfg.num_lms, 2))
        lms_pred_merge[:, 0] = lms_pred_merge[:, 0] * det_width + det_xmin
        lms_pred_merge[:, 1] = lms_pred_merge[:, 1] * det_height + det_ymin
        lms_pred_merge = lms_pred_merge.astype(np.int32)
        return lms_pred_merge
