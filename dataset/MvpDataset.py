# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Dataset class for multiview video datasets"""
import multiprocessing
import os
from typing import Optional, Callable

import numpy as np

from scipy.ndimage.morphology import binary_dilation

from PIL import Image

import torch.utils.data

import cv2
cv2.setNumThreads(0)

from models.mvp.utils import utils

def load_camera_info_from_capturepath(capturepath: str, conversion_matrix: np.array = np.eye(3), scale: float = 1.0):
    cameras = {}
    list_of_cam_folders = os.listdir(capturepath)
    list_of_cam_folders = [folder for folder in list_of_cam_folders if folder != '.gitkeep']
    for cam_folder in list_of_cam_folders:
        extrinsics_path = os.path.join(capturepath, cam_folder, 'extrinsics.npy')
        intrinsics_path = os.path.join(capturepath, cam_folder, 'intrinsics.npy')
        dist = np.array([0, 0, 0, 0, 0]).astype(np.float32)
        extrin = np.load(extrinsics_path)
        extrin = np.linalg.inv(extrin)  # c2w -> w2c
        # extrin = conversion_matrix @ extrin[:3, :4]
        # extrin[1:3, :3] *= -1
        extrin[:3, 3] *= scale
        intrin = np.load(intrinsics_path)
        cameras[cam_folder] = {
            "intrin": intrin,
            "dist": dist,
            "extrin": extrin
        }
    return cameras

class ImageLoader:
    def __init__(self, bgpath, blacklevel):
        self.bgpath = bgpath
        self.blacklevel = np.array(blacklevel, dtype=np.float32)

    def __call__(self, cam, size):
        try:
            imagepath = self.bgpath.format(cam)
            # determine downsampling necessary
            image = np.asarray(Image.open(imagepath), dtype=np.uint8)
            if image.shape[0] != size[1]:
                image = utils.downsample(image, image.shape[0] // size[1])
            image = image.transpose((2, 0, 1)).astype(np.float32)
            image = np.clip(image - self.blacklevel[:, None, None], 0., None)
            bgmask = (~binary_dilation(np.mean(image, axis=0) > 128, iterations=16)).astype(np.float32)[None, :, :]
        except:
            print("exception loading bg image", cam)
            image = None
            bgmask = None

        return cam, image, bgmask

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
            geomdir : str,
            capturepath : str,
            keyfilter : list,
            camerafilter : Callable[[str], bool],
            framelist : Optional[list]=None,
            maxframes : int=-1,
            bgpath : Optional[str]=None,
            returnbg : bool=False,
            baseposepath : Optional[str]=None,
            fixedcameras : list=[],
            fixedframesegframe : Optional[tuple]=None,
            fixedcammean : float=0.,
            fixedcamstd : float=1.,
            fixedcamdownsample : int=4,
            standardizeverts : bool=True,
            standardizeavgtex : bool=True,
            standardizetex : bool=False,
            avgtexsize : int=1024,
            subsampletype : Optional[str]=None,
            subsamplesize : int=0,
            downsample : float=1.,
            blacklevel : list=[0., 0., 0.],
            maskbright : bool=False,
            maskbrightbg : bool=False,
            img_size: np.array = None,
            convention: str = "opencv",
            scale: float = 1.0
            ):
        """
        Dataset class for loading synchronized multi-view video (optionally
        with tracked mesh data).

        Parameters
        ----------
        geomdir : str,
            base path to geometry data (tracked meshes, unwrapped textures,
            rigid transforms)
        keyfilter : list,
            list of items to load and return (e.g., images, textures, vertices)
            available in this dataset:
            'fixedcamimage' -- image from a particular camera (unlike 'image',
                this image will always be from a specified camera)
            'fixedframeimage' -- image from a particular frame and camera
                (always the same)
            'verts' -- tensor of Kx3 vertices
            'tex' -- texture map as (3 x T_h x T_w) tensor
            'avgtex' -- texture map averaged across all cameras
            'modelmatrix' -- rigid transformation at frame
                (relative to 'base' pose)
            'camera' -- camera pose (intrinsic and extrinsic)
            'image' -- camera image as (3 x I_h x I_w) tensor
            'bg' -- background image as (3 x I_h x I_w) tensor
            'pixelcoords' -- pixel coordinates of image to evaluate
                (used to subsample training images to reduce memory usage)
        camerafilter : Callable[[str], bool],
            lambda function for selecting cameras to include in dataset
        segmentfilter : Callable[[str], bool]
            lambda function for selecting segments to include in dataset
            Segments are contiguous sets of frames.
        framelist=None : list[tuple[str, str]],
            list of (segment, framenumber), used instead of segmentfilter
        frameexclude : list[str],
            exclude these frames from being loaded
        maxframes : int,
            maximum number of frames to load
        bgpath : Optional[str],
            path to background images if available
        returnbg : bool,
            True to return bg images in each batch, false to store them
            into self.bg
        baseposesegframe : tuple[str, str]
            segment, frame of headpose to be used as the "base" head pose
            (used for modelmatrix)
        baseposepath : str,
            path to headpose to be used as the "base" head pose, used instead
            of transfseg, transfframe
        fixedcameras : list,
            list of cameras to be returned for 'fixedcamimage'
        fixedframesegframe : tuple[str, str]
            segment and frame to be used for 'fixedframeimage'
        fixedcammean : float,
        fixedcamstd : float,
            norm stats for 'fixedcamimage' and 'fixedframeimage'
        standardizeverts : bool,
            remove mean/std from vertices
        standardizeavgtex : bool,
            remove mean/std from avgtex
        standardizetex : bool,
            remove mean/std from view-dependent texture
        avgtexsize : int,
            average texture map (averaged across viewpoints) dimension
        texsize : int,
            texture map dimension
        subsampletype : Optional[str],
            type of subsampling to do (determines how pixelcoords is generated)
            one of [None, "patch", "random", "random2", "stratified"]
        subsamplesize : int,
            dimension of subsampling
        downsample : float,
            downsample target image by factor
        blacklevel : tuple[float, float, float]
            black level to subtract from camera images
        maskbright : bool,
            True to not include bright pixels in loss
        maskbrightbg : bool,
            True to not include bright background pixels in loss
        img_size: np.array,
            size of the image
        convention: str,
            convention of the camera matrix, mvp expects opencv convention, if other convention is used,
            the coordinates will be flipped
        """
        # options
        self.keyfilter = keyfilter
        self.fixedcameras = fixedcameras
        self.fixedframesegframe = fixedframesegframe
        self.fixedcammean = fixedcammean
        self.fixedcamstd = fixedcamstd
        self.fixedcamdownsample = fixedcamdownsample
        self.standardizeverts = standardizeverts
        self.standardizeavgtex = standardizeavgtex
        self.standardizetex = standardizetex
        self.subsampletype = subsampletype
        self.subsamplesize = subsamplesize
        self.downsample = downsample
        self.returnbg = returnbg
        self.blacklevel = blacklevel
        self.maskbright = maskbright
        self.maskbrightbg = maskbrightbg

        self.conversion_matrix = np.eye(3)
        if convention == "opengl":
            self.conversion_matrix[1, 1] = -1
            self.conversion_matrix[2, 2] = -1
        self.scale = scale

        # compute camera/frame list
        camera_info = load_camera_info_from_capturepath(capturepath, conversion_matrix=self.conversion_matrix, scale=self.scale)

        self.allcameras = sorted(list(camera_info.keys()))
        self.cameras = list(filter(camerafilter, self.allcameras))

        # compute camera positions
        self.campos, self.camrot, self.focal, self.princpt, self.size = {}, {}, {}, {}, {}
        for cam in self.allcameras:
            self.campos[cam] = (-np.dot(camera_info[cam]['extrin'][:3, :3].T, camera_info[cam]['extrin'][:3, 3])).astype(np.float32)
            self.camrot[cam] = (camera_info[cam]['extrin'][:3, :3]).astype(np.float32)
            self.focal[cam] = (np.diag(camera_info[cam]['intrin'][:2, :2]) / downsample).astype(np.float32)
            self.princpt[cam] = (camera_info[cam]['intrin'][:2, 2] / downsample).astype(np.float32)
            size = img_size if img_size is not None else camera_info[cam]['size']
            self.size[cam] = np.floor(size.astype(np.float32) / downsample).astype(np.int32)

        # set up paths
        self.imagepath = os.path.join(capturepath, "{cam}", "images", "{frame:05d}.png")
        self.bg_image_path = os.path.join(capturepath, "{cam}", "alpha_map", "{frame:05d}.png")
        if geomdir is not None:
            self.vertpath = os.path.join(geomdir, "{frame:05d}.bin")
            self.transfpath = os.path.join(geomdir, "{frame:05d}_transform.txt")
            self.texpath = os.path.join(geomdir, "{frame:05d}.png")
        else:
            self.transfpath = None

        # build list of frames
        if framelist is None:
            list_of_frame_ids = sorted(os.listdir(os.path.join(capturepath, self.allcameras[0], "images")))
            self.framelist = [os.path.splitext(file)[0] for file in list_of_frame_ids if ".png" in file]
        else:
            self.framelist = framelist

        # truncate or extend frame list
        if maxframes <= len(self.framelist):
            if maxframes > -1:
                self.framelist = self.framelist[:maxframes]
        else:
            repeats = (maxframes + len(self.framelist) - 1) // len(self.framelist)
            self.framelist = (self.framelist * repeats)[:maxframes]

        # cartesian product with cameras
        self.framecamlist = [(x, cam)
                for x in self.framelist
                for cam in (self.cameras if len(self.cameras) > 0 else [None])]

        # set base pose
        if baseposepath is not None:
            self.basetransf = np.genfromtxt(baseposepath, max_rows=3).astype(np.float32)
        else:
            raise Exception("base transformation must be provided")

        # load normstats
        if "avgtex" in keyfilter:
            texmean = np.asarray(Image.open(os.path.join(geomdir, "tex_mean.png")), dtype=np.float32)
            self.texstd = float(np.genfromtxt(os.path.join(geomdir, "tex_var.txt")) ** 0.5)
            self.avgtexsize = avgtexsize
            avgtexmean = texmean
            if avgtexmean.shape[0] != self.avgtexsize:
                avgtexmean = cv2.resize(avgtexmean, dsize=(self.avgtexsize, self.avgtexsize),
                                        interpolation=cv2.INTER_LINEAR)
            self.avgtexmean = avgtexmean.transpose((2, 0, 1)).astype(np.float32).copy("C")

        if "verts" in keyfilter:
            self.vertmean = np.fromfile(os.path.join(geomdir, "vert_mean.bin"), dtype=np.float32).reshape((-1, 3))
            self.vertstd = float(np.genfromtxt(os.path.join(geomdir, "vert_var.txt")) ** 0.5)

        # load background images
        if bgpath is not None:
            readpool = multiprocessing.Pool(40)
            reader = ImageLoader(bgpath, blacklevel)
            self.bg = {cam: (image, bgmask)
                    for cam, image, bgmask
                    in readpool.starmap(reader, zip(self.cameras, [self.size[x] for x in self.cameras]))
                    if image is not None}
        else:
            self.bg = {}

    def get_allcameras(self):
        return self.allcameras

    def get_krt(self):
        return {k: {
                "pos": self.campos[k],
                "rot": self.camrot[k],
                "focal": self.focal[k],
                "princpt": self.princpt[k],
                "size": self.size[k]}
                for k in self.allcameras}

    def known_background(self):
        return "bg" in self.keyfilter

    def get_background(self, bg):
        if "bg" in self.keyfilter and not self.returnbg:
            for i, cam in enumerate(self.cameras):
                if cam in self.bg:
                    bg[cam].data[:] = torch.from_numpy(self.bg[cam][0]).to("cuda")

    def __len__(self):
        return len(self.framecamlist)

    def __getitem__(self, idx):
        frame, cam = self.framecamlist[idx]
        result = {}

        result["frameid"] = frame
        if cam is not None:
            result["cameraid"] = cam

        validinput = True

        # vertices
        for k in ["verts", "verts_next"]:
            if k in self.keyfilter:
                vertpath = self.vertpath.format(frame=int(frame) + (1 if k == "verts_next" else 0))
                verts = np.fromfile(vertpath, dtype=np.float32)
                if self.standardizeverts:
                    verts -= self.vertmean.ravel()
                    verts /= self.vertstd
                result[k] = verts.reshape((-1, 3))

        # texture averaged over all cameras for a single frame
        for k in ["avgtex", "avgtex_next"]:
            if k in self.keyfilter:
                texpath = self.texpath.format(frame=int(frame) + (1 if k == "avgtex_next" else 0))
                try:
                    tex = np.asarray(Image.open(texpath), dtype=np.uint8)
                    if tex.shape[0] != self.avgtexsize:
                        tex = cv2.resize(tex, dsize=(self.avgtexsize, self.avgtexsize), interpolation=cv2.INTER_LINEAR)
                    tex = tex.transpose((2, 0, 1)).astype(np.float32)
                except:
                    tex = np.zeros((3, self.avgtexsize, self.avgtexsize), dtype=np.float32)
                    validinput = False
                if np.sum(tex) == 0:
                    validinput = False
                texmask = np.sum(tex, axis=0) != 0
                if self.standardizeavgtex:
                    tex -= self.avgtexmean
                    tex /= self.texstd
                    tex[:, ~texmask] = 0.
                result[k] = tex

        # keep track of whether we fail to load any of the input
        result["validinput"] = np.float32(1.0 if validinput else 0.0)

        if "modelmatrix" in self.keyfilter or "modelmatrixinv" in self.keyfilter or "camera" in self.keyfilter:
            def to4x4(m):
                return np.r_[m, np.array([[0., 0., 0., 1.]], dtype=np.float32)]

            # per-frame rigid transformation of scene/object
            for k in ["modelmatrix", "modelmatrix_next"]:
                if k in self.keyfilter:
                    if self.transfpath is not None:
                        transfpath = self.transfpath.format(frame=int(frame) + (1 if k == "modelmatrix_next" else 0))
                        try:
                            frametransf = np.genfromtxt(os.path.join(transfpath), max_rows=3).astype(np.float32)
                        except:
                            frametransf = None

                        result[k] = to4x4(np.dot(
                            np.linalg.inv(to4x4(frametransf)),
                            to4x4(self.basetransf))[:3, :4])
                    else:
                        result[k] = np.eye(3, 4, dtype=np.float32)#np.linalg.inv(to4x4(self.basetransf))[:3, :4]

            # inverse of per-frame rigid transformation of scene/object
            for k in ["modelmatrixinv", "modelmatrixinv_next"]:
                if k in self.keyfilter:
                    if self.transfpath is not None:
                        transfpath = self.transfpath.format(frame=int(frame) + (1 if k == "modelmatrixinv_next" else 0))
                        try:
                            frametransf = np.genfromtxt(os.path.join(transfpath), max_rows=3).astype(np.float32)
                        except:
                            frametransf = None

                        result[k] = to4x4(np.dot(
                            np.linalg.inv(to4x4(self.basetransf)),
                            to4x4(frametransf))[:3, :4])
                    else:
                        result[k] = np.eye(3, 4, dtype=np.float32)#self.basetransf

        # camera-specific data
        if cam is not None:
            # camera pose
            if "camera" in self.keyfilter:
                result["campos"] = np.dot(self.basetransf[:3, :3].T, self.campos[cam] - self.basetransf[:3, 3])
                result["camrot"] = np.dot(self.basetransf[:3, :3].T, self.camrot[cam].T).T
                result["focal"] = self.focal[cam]
                result["princpt"] = self.princpt[cam]
                result["camindex"] = self.allcameras.index(cam)


            # camera images
            if "image" in self.keyfilter:
                # target image
                imagepath = self.imagepath.format(cam=cam, frame=int(frame))
                image = utils.downsample(
                        np.asarray(Image.open(imagepath), dtype=np.uint8),
                        self.downsample).transpose((2, 0, 1)).astype(np.float32)
                bg_image_path = self.bg_image_path.format(cam=cam, frame=int(frame))
                bg_img = np.asarray(Image.open(bg_image_path))
                bg_img = bg_img[:, :, None]
                bg_img = np.repeat(bg_img, 3, axis=2)
                background = utils.downsample(bg_img,
                                  self.downsample).transpose((2, 0, 1)).astype(np.float32)
                background /= 255.
                # resize background image to match target image
                background = background.transpose((1, 2, 0))
                background = cv2.resize(background, dsize=(image.shape[2], image.shape[1]))
                background = background.transpose((2, 0, 1))

                image = image * background  # alpha matting
                height, width = image.shape[1:3]
                valid = np.float32(1.0) if np.sum(image) != 0 else np.float32(0.)

                # remove black level
                result["image"] = np.clip(image - np.array(self.blacklevel, dtype=np.float32)[:, None, None], 0., None)
                result["imagevalid"] = valid

                # optionally mask pixels with bright background values
                if self.maskbrightbg and cam in self.bg:
                    result["imagemask"] = self.bg[cam][1]

                # optionally mask pixels with bright values
                if self.maskbright:
                    if "imagemask" in result:
                        result["imagemask"] *= np.where(
                                (image[0] > 245.) |
                                (image[1] > 245.) |
                                (image[2] > 245.), 0., 1.)[None, :, :]
                    else:
                        result["imagemask"] = np.where(
                                (image[0] > 245.) |
                                (image[1] > 245.) |
                                (image[2] > 245.), 0., 1.).astype(np.float32)[None, :, :]

            # background image
            if "bg" in self.keyfilter and self.returnbg:
                result["bg"] = self.bg[cam][0]

            # image pixel coordinates
            if "pixelcoords" in self.keyfilter:
                if self.subsampletype == "patch":
                    indx = torch.randint(0, width - self.subsamplesize + 1, size=(1,)).item()
                    indy = torch.randint(0, height - self.subsamplesize + 1, size=(1,)).item()

                    py, px = torch.meshgrid(
                            torch.arange(indy, indy + self.subsamplesize).float(),
                            torch.arange(indx, indx + self.subsamplesize).float())
                elif self.subsampletype == "random":
                    px = torch.randint(0, width, size=(self.subsamplesize, self.subsamplesize)).float()
                    py = torch.randint(0, height, size=(self.subsamplesize, self.subsamplesize)).float()
                elif self.subsampletype == "random2":
                    px = torch.random(size=(self.subsamplesize, self.subsamplesize)).float() * (width - 1)
                    py = torch.random(size=(self.subsamplesize, self.subsamplesize)).float() * (height - 1)
                elif self.subsampletype == "stratified":
                    ssy = self.subsamplesize
                    ssx = self.subsamplesize
                    bsizex = (width - 1.) / ssx
                    bsizey = (height - 1.) / ssy
                    px = (torch.arange(ssx)[None, :] + torch.rand(size=(ssy, ssx))) * bsizex
                    py = (torch.arange(ssy)[:, None] + torch.rand(size=(ssy, ssx))) * bsizey
                elif self.subsampletype == None:
                    py, px = torch.meshgrid(torch.arange(height).float(), torch.arange(width).float())
                else:
                    raise

                result["pixelcoords"] = torch.stack([px, py], dim=-1)

        return result
