import tensorrt as trt
from PIL import Image
import os
import numpy as np
import time
import pycuda.driver as cuda
from skimage import transform, io
import pycuda.autoinit
from io import BytesIO
import cv2

class ModelData(object):
    # Name of input node
    INPUT_NAME = "input"
    # CHW format of model input
    INPUT_SHAPE = (3, 320, 320)
    # Name of output node
    OUTPUT_NAME = "output"

    @staticmethod
    def get_input_channels():
        return ModelData.INPUT_SHAPE[0]

    @staticmethod
    def get_input_height():
        return ModelData.INPUT_SHAPE[1]

    @staticmethod
    def get_input_width():
        return ModelData.INPUT_SHAPE[2]

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    """Allocates host and device buffer for TRT engine inference.
    This function is similair to the one in common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.
    Args:
        engine (trt.ICudaEngine): TensorRT engine
    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()


    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]
    binding_to_type = {"input": np.float32, "output": np.float32}

    for binding in engine:
        size = abs(trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size)
        dtype = binding_to_type[str(binding)]
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings, stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference(object):
    """Manages TensorRT objects for model inference."""
    def __init__(self, trt_engine_path, trt_engine_datatype=trt.DataType.FLOAT, calib_dataset=None, batch_size=1):
        """Initializes TensorRT objects needed for model inference.
        Args:
            trt_engine_path (str): path where TensorRT engine should be stored
            uff_model_path (str): path of .uff model
            trt_engine_datatype (trt.DataType):
                requested precision of TensorRT engine used for inference
            batch_size (int): batch size for which engine
                should be optimized for
        """

        # We first load all custom plugins shipped with TensorRT,
        # some of them will be needed during inference
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TRT engine placeholder
        self.trt_engine = None

        # Display requested engine settings to stdout
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))

        # # If engine is not cached, we need to build it
        # if not os.path.exists(trt_engine_path):
        #     # This function uses supplied .uff file
        #     # alongside with UffParser to build TensorRT
        #     # engine. For more details, check implmentation
        #     self.trt_engine = engine_utils.build_engine(
        #         uff_model_path, TRT_LOGGER,
        #         trt_engine_datatype=trt_engine_datatype,
        #         calib_dataset=calib_dataset,
        #         batch_size=batch_size)
        #     # Save the engine to file
        #     engine_utils.save_engine(self.trt_engine, trt_engine_path)

        # If we get here, the file with engine exists, so we can load it
        if not self.trt_engine:
            print("Loading cached TensorRT engine from {}".format(
                trt_engine_path))
            self.trt_engine = load_engine(
                self.trt_runtime, trt_engine_path)

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.trt_engine)
        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()
        self.context.set_input_shape('input', (batch_size, 3, 320, 320))

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        input_volume = trt.volume(ModelData.INPUT_SHAPE)
        self.numpy_array = np.zeros((self.trt_engine.max_batch_size, input_volume))

    def infer(self, image_path):
        """Infers model on given image.
        Args:
            image_path (str): image to run object detection model on
        """
        
        # Load image into CPU
        img = self._load_img(image_path)
        shape = img.shape
        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, np.ravel(img))
        # When infering on single image, we measure inference
        # time to output it to the user
        inference_start_time = time.time()

        # Fetch output from the model
        [output] = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)

        # Output inference time
        print("TensorRT inference time: {} ms".format(
            int(round((time.time() - inference_start_time) * 1000))))
        # And return results
        return output.reshape(1, shape[1], shape[2])


    def infer_batch(self, image_list):
        """Infers model on batch of same sized images resized to fit the model.
        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        shape = self.context.get_tensor_shape('output')
        # Verify if the supplied batch size is not too big
        max_batch_size = self.trt_engine.max_batch_size
        actual_batch_size = len(image_list)
        if actual_batch_size > max_batch_size:
            raise ValueError(
                "image_paths list bigger ({}) than engine max batch size ({})".format(actual_batch_size, max_batch_size))

        # Load all images to CPU...
        imgs = self._load_imgs(image_list)
        # ...copy them into appropriate place into memory...
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, imgs.ravel())

        # ...fetch model outputs...
        [output] = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=max_batch_size)
        # ...and return results.
        return output.reshape(shape)

    def _load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image).reshape(
            (im_height, im_width, ModelData.get_input_channels())
        ).astype(np.uint8)

    def _rescale_img(self, image, output_size):
        image = np.array(image)
        h, w = image.shape[:2]
        if isinstance(output_size,int):
            if h > w:
                new_h, new_w = output_size*h/w,output_size
            else:
                new_h, new_w = output_size,output_size*w/h
        else:
            new_h, new_w = output_size
        
        new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)
        img = transform.resize(image,(output_size,output_size),mode='constant')
        return img

    def _to_image_lab(self, image):
        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
        image = image/np.max(image)
        if image.shape[2]==1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
        tmpImg = tmpImg.transpose((2, 0, 1))
        return tmpImg

    def _load_imgs(self, image_list):
        batch_size = self.trt_engine.max_batch_size
        for idx, img_np in enumerate(image_list):
            img_np = self.preprocess(img_np)
            self.numpy_array[idx] = img_np
        return self.numpy_array

    def preprocess(self, image):
        model_input_width = ModelData.get_input_width()
        model_input_height = ModelData.get_input_height()
        # Note: Bilinear interpolation used by Pillow is a little bit
        # different than the one used by Tensorflow, so if network receives
        # an image that is not 300x300, the network output may differ
        # from the one output by Tensorflow
        # image_resized = image.resize(
        #     size=(model_input_width, model_input_height),
        #     resample=Image.BILINEAR
        # )
        img_np = self._rescale_img(image, model_input_height)
        img_np = self._to_image_lab(img_np)
        # img_np = self._load_image_into_numpy_array(image_resized)
        # HWC -> CHW
        # img_np = img_np.transpose((2, 0, 1))
        # Normalize to [-1.0, 1.0] interval (expected by model)
        # img_np = (2.0 / 255.0) * img_np - 1.0
        img_np = img_np.ravel()
        return img_np
def normPRED(d):
    ma = np.max(d)
    mi = np.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def infer(img, engine_file='model_repository/u2net.plan', batch_size=1, ths=0.5):
    trt_inf = TRTInference(trt_engine_path=engine_file, trt_engine_datatype=trt.DataType.HALF, batch_size=batch_size)
    output = trt_inf.infer_batch([img])
    output = normPRED(output[:,0,:,:]).squeeze()
    image = Image.fromarray(output*255).convert('RGB')
    # image = image[:,:,::-1]
    mask_img = image.resize((img.shape[1],img.shape[0]),resample=Image.BILINEAR)
    mask_img = np.array(mask_img)
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    mask_img[mask_img > ths] = 1.0
    mask_img[mask_img <= ths] = 0.0
    mask_img = (mask_img * 255).astype(np.uint8)
    return mask_img

# if __name__ == '__main__':
#     trt_inf = TRTInference(trt_engine_path='backend/u2net.plan', trt_engine_datatype=trt.DataType.HALF, batch_size=1)
#     img = io.imread('backend/mcocr_private_145120aorof.jpg')
#     output = trt_inf.infer_batch(['backend/mcocr_private_145120aorof.jpg'])
#     output = normPRED(output[:,0,:,:]).squeeze()
    
#     image = Image.fromarray(output*255).convert('RGB')
#     # image = image[:,:,::-1]
#     image = image.resize((img.shape[1],img.shape[0]),resample=Image.BILINEAR)
#     image.save('backend/output.jpg')