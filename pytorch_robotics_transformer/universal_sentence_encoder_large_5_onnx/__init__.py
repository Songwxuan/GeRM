import onnxruntime as ort
from onnxruntime_extensions import get_library_path
from os import cpu_count

def load_onnx_model(model_filepath, use_cuda=True, device_id=0):
    _options = ort.SessionOptions()
    if not use_cuda:
        _options.inter_op_num_threads, _options.intra_op_num_threads = cpu_count(), cpu_count()
    _options.register_custom_ops_library(get_library_path())
    _providers = [("CUDAExecutionProvider", {'device_id': device_id}), 'CPUExecutionProvider']  # could use ort.get_available_providers()
    session = ort.InferenceSession(path_or_bytes=model_filepath, sess_options=_options, providers=_providers)
    return session