import os
import numpy as np

import dataset.MvpDataset as datamodel

################ output path
outpath = "../output"

################ input path
capturepath = "../data/subject_24/"
objpath = "../data/subject_24_tracked_mesh/flame.obj"
geomdir = os.path.join("../data/subject_24_tracked_mesh")
bgpath = None
baseposepath = os.path.join(geomdir, "00000_transform.txt")

################ settings
holdoutcams = []
img_size = np.array([2200, 3208])
num_images_per_row = 3
batchsize = 6

def get_dataset(
        camerafilter=lambda x: x not in holdoutcams,
        keyfilter=[],
        maxframes=-1,
        subsampletype=None,
        downsample=2,
        **kwargs):
    """
    Parameters
    -----
    camerafilter : Callable[[str], bool]
        Function to determine cameras to use in dataset (camerafilter(cam) ->
        True to use cam, False to not use cam).
    segmentfilter : Callable[[str], bool]
        Function to determine segments to use in dataset (segmentfilter(seg) ->
        True to use seg, False to not use seg).
    keyfilter : list
        List of data to load from dataset. See Dataset class (e.g.,
        data.multiviewvideo) for a list of valid keys.
    maxframes : int
        Maximum number of frames to load.
    subsampletype : Optional[str]
        Type of subsampling to perform on camera images. Note the PyTorch
        module does the actual subsampling, the dataset class returns an array
        of pixel coordinates.
    downsample : int
        Downsampling factor of input images.
    """
    return datamodel.Dataset(
        geomdir=geomdir,
        capturepath=capturepath,
        bgpath=bgpath,
        returnbg=False,
        avgtexsize=256,
        baseposepath=baseposepath,
        camerafilter=camerafilter,
        keyfilter=["bg", "camera", "modelmatrix", "modelmatrixinv", "pixelcoords", "image", "verts"] + keyfilter,
        maxframes=maxframes,
        subsampletype=subsampletype,
        subsamplesize=384,
        downsample=downsample,
        blacklevel=[3.8, 2.5, 4.0],
        maskbright=True,
        maskbrightbg=True,
        img_size=img_size,
        convention="opengl",
        scale=1000.
        )

def get_renderoptions():
    """Return dict of rendering options"""
    return dict(
            algo=0, # raymarcher can support multiple types of rendering, 0 is default
            chlast=True, # whether voxel grid has channels (RGBA) in the last dimension
            dt=1.0) # stepsize

def get_autoencoder(dataset, renderoptions):
    print(f"Render options: {renderoptions}")
    """Return an autoencoder instance"""
    import torch
    import models.mvp.models.volumetric as aemodel
    import models.mvp.models.encoders.geotex as encoderlib
    import models.mvp.models.decoders.mvp as decoderlib
    import models.mvp.models.raymarchers.mvpraymarcher as raymarcherlib
    import models.mvp.models.colorcals.colorcal as colorcalib
    import models.mvp.models.bg.lap as bglib
    from models.mvp.utils import utils

    allcameras = dataset.get_allcameras()
    width, height = next(iter(dataset.get_krt().values()))["size"]

    # per-camera color calibration
    colorcal = colorcalib.Colorcal(dataset.get_allcameras())

    # mesh topology
    v, vt, vi, vti = utils.load_obj(objpath)
    vt = np.array(vt, dtype=np.float32)
    vi = np.array(vi, dtype=np.int32)
    vti = np.array(vti, dtype=np.int32)
    idxim, tidxim, barim = utils.gentritex(v, vt, vi, vti, 256)
    idxim = torch.tensor(idxim).long()
    tidxim = torch.tensor(tidxim).long()
    barim = torch.tensor(barim)

    vertmean = torch.from_numpy(dataset.vertmean)
    vertstd = dataset.vertstd

    encoder = encoderlib.Encoder(vertsize=len(v)*3, texin=False)
    print("encoder:", encoder)

    volradius = 256.
    decoder = decoderlib.Decoder(
        vt, vertmean, vertstd,
        idxim, tidxim, barim,
        volradius=256.,
        dectype="slab2d",
        nprims=256,
        primsize=(32, 32, 32),
        #nprims=4096,
        #primsize=(16, 16, 8),
        motiontype="deconv",
        sharedrgba=False,
        elr=True,
        postrainstart=100,
        renderoptions=renderoptions)

    print("decoder:", decoder)

    raymarcher = raymarcherlib.Raymarcher(volradius=volradius)
    print("raymarcher:", raymarcher)

    bgmodel = bglib.BGModel(width, height, allcameras, trainstart=0, startlevel=2, buftop=True)
    # initialize bg
    for i, cam in enumerate(dataset.cameras):
        if cam in dataset.bg:
            bgmodel.lap.pyr[-1].image[0, :, allcameras.index(cam), :,  :].data[:] = (
                    torch.from_numpy(dataset.bg[cam][0]).to("cuda"))
    print("bgmodel:", bgmodel)

    ae = aemodel.Autoencoder(
        dataset,
        encoder,
        decoder,
        raymarcher,
        colorcal,
        volradius,
        bgmodel,
        encoderinputs=["verts"],
        # encoderinputs=["verts", "avgtex"],
        topology={"vt": vt, "vi": vi, "vti": vti},
        imagemean=100.,
        imagestd=25.)

    print("encoder params:", sum(p.numel() for p in ae.encoder.parameters() if p.requires_grad))
    print("decoder params:", sum(p.numel() for p in ae.decoder.parameters() if p.requires_grad))
    print("colorcal params:", sum(p.numel() for p in ae.colorcal.parameters() if p.requires_grad))
    print("bgmodel params:", sum(p.numel() for p in ae.bgmodel.parameters() if p.requires_grad))
    print("total params:", sum(p.numel() for p in ae.parameters() if p.requires_grad))

    return ae

# profiles
class Train():
    """Profile for training models."""
    batchsize=batchsize
    def __init__(self, maxiter=20000, **kwargs):
        self.maxiter = maxiter
        self.otherargs = kwargs
    def get_autoencoder(self, dataset):
        """Returns a PyTorch Module that accepts inputs and produces a dict
        of output values. One of those output values should be 'losses', another
        dict with each of the separate losses. See models.volumetric for an example."""
        return get_autoencoder(dataset, **self.get_ae_args())
    def get_outputlist(self):
        """A dict that is passed to the autoencoder telling it what values
        to compute (e.g., irgbrec for the rgb image reconstruction)."""
        return []
    def get_ae_args(self):
        """Any non-data arguments passed to the autoencoder's forward method."""
        return dict(renderoptions={**get_renderoptions(), **self.otherargs})
    def get_dataset(self):
        """A Dataset class that returns data for the autoencoder"""
        return get_dataset(subsampletype="stratified")
    def get_optimizer(self, ae):
        """The optimizer used to train the autoencoder parameters."""
        import itertools
        import torch.optim
        lr = 2e-3
        aeparams = itertools.chain(
            [{"params": x} for k, x in ae.encoder.named_parameters()],
            [{"params": x} for k, x in ae.decoder.named_parameters()],
            [{"params": x} for x in ae.bgmodel.parameters()],
            )
        return torch.optim.Adam(aeparams, lr=lr, betas=(0.9, 0.999))
    def get_loss_weights(self):
        return {"irgbmse": 1.0, "vertmse": 0.1, "kldiv": 0.001, "primvolsum": 0.01}

class ProgressWriter():
    def batch(self, iternum, itemnum, **kwargs):
        import numpy as np
        from PIL import Image
        rows = []
        row = []
        for i in range(kwargs["image"].size(0)):
            row.append(
                np.concatenate((
                    kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                    kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1))
            if len(row) == num_images_per_row:
                rows.append(np.concatenate(row, axis=1))
                row = []
        if len(row) > 0:
            rows.append(np.concatenate([row[i] if i < len(row) else row[0]*0. for i in range(num_images_per_row)], axis=1))
        imgout = np.concatenate(rows, axis=0)
        Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(
                os.path.join(outpath, "prog_{:05}.jpg".format(iternum)))
    def finalize(self):
        pass

class Progress():
    """Profile for writing out progress images during training."""
    batchsize=batchsize
    def get_outputlist(self): return ["irgbrec"]
    def get_ae_args(self): return dict(renderoptions=get_renderoptions())
    def get_dataset(self): return get_dataset()
    def get_writer(self): return ProgressWriter()

class Eval():
    """Profile for evaluating models."""
    def __init__(self, outfilename=None, outfilesuffix=None,
            cam=None, camdist=768., camperiod=64, camrevs=0.2,
            segments=["S23_When_she_awoke_she_was_the_ship."],
            maxframes=-1,
            keyfilter=[],
            **kwargs):
        self.outfilename = outfilename
        self.outfilesuffix = outfilesuffix
        self.cam = cam
        self.camdist = camdist
        self.camperiod = camperiod
        self.camrevs = camrevs
        self.segments = segments
        self.segmentfilter = lambda x: True if segments == "all" else x in segments
        self.maxframes = maxframes
        self.keyfilter = keyfilter
        self.otherargs = kwargs
    def get_autoencoder(self, dataset): return get_autoencoder(dataset, **self.get_ae_args())
    def get_outputlist(self): return [k[0] if isinstance(k, tuple) else k for k in self.keyfilter]
    def get_ae_args(self): return dict(renderoptions={**get_renderoptions(),
        **self.otherargs})
    def get_dataset(self):
        import models.mvp.data.utils
        import models.mvp.data.camrotate as cameralib
        if self.cam == "all":
            camerafilter = lambda x: x.startswith("40")
        elif self.cam == "holdout":
            camerafilter = lambda x: x in holdoutcams
        else:
            camerafilter = lambda x: x == self.cam
        dataset = get_dataset(camerafilter=camerafilter,
                segmentfilter=self.segmentfilter,
                keyfilter=[k[0] if isinstance(k, tuple) else k for k in self.keyfilter],
                maxframes=self.maxframes,
                **{k: v for k, v in self.otherargs.items() if k in get_dataset.__code__.co_varnames},
                )
        if self.cam is None:
            camdataset = cameralib.Dataset(len(dataset), camdist=self.camdist,
                    period=self.camperiod, revs=self.camrevs,
                    **{k: v for k, v in self.otherargs.items() if k in cameralib.Dataset.__init__.__code__.co_varnames})
            return models.mvp.data.utils.ColCatDataset(camdataset, dataset)
        else:
            return dataset
    def get_writer(self):
        import models.mvp.utils.videowriter as writerlib
        if self.outfilename is None:
            outfilename = (
                    "render_{}_{}".format("-".join([x[:4].replace('_', '') for x in self.segments]), self.cam) +
                    (self.outfilesuffix if self.outfilesuffix is not None else "") +
                    ".mp4")
        else:
            outfilename = self.outfilename
        return writerlib.Writer(
            os.path.join(os.path.dirname(__file__), outfilename),
            keyfilter=self.keyfilter,
            colcorrect=[1.35, 1.16, 1.5],
            bgcolor=[255., 255., 255.],
            **{k: v for k, v in self.otherargs.items() if k in ["cmap", "cmapscale", "colorbar"]})
