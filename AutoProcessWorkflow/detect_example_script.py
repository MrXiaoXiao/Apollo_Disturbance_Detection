import numpy as np
import matplotlib.pyplot as plt
import math
import os
import tensorflow as tf

from scipy.interpolate import interp1d
import obspy
from obspy.signal.trigger import trigger_onset

def extend_window(windows, extend_len=10):
    len_window = len(windows)
    for wdx in range(len(windows)):
        if wdx == 0 or wdx == len_window:
            continue
        windows[wdx][0] = windows[wdx][0] - extend_len
        if windows[wdx][0] < 0:
            windows[wdx][0] = 0
        windows[wdx][1] = windows[wdx][1] + extend_len
    return windows

def merge_window(windows):
    windows.sort(key=lambda x: x[0])

    merged_windows = [windows[0]]

    for current_start, current_end in windows[1:]:
        last_merged_start, last_merged_end = merged_windows[-1]

        # Check if the current window overlaps with the last merged window
        if current_start <= last_merged_end:
            # Merge the windows by extending the end time of the last merged window
            merged_windows[-1] = [last_merged_start, max(last_merged_end, current_end)]
        else:
            # No overlap, add the current window to the merged list
            merged_windows.append([current_start, current_end])

    # Update the windows list
    windows = merged_windows

    return windows

def apollo_linear_interp_gaps(trace):
    # use median filter to remove the high spikes
    trace = trace.copy()
    data = trace.data
    # remove the masks in the stream:
    indexes = np.where(data == -1)[0]
    mask_windows = []
    for i in range(len(indexes)):
        mask_windows.append([indexes[i]-1, indexes[i]+1])
    # merge the windows that has overlap or adjacent
    mask_windows = merge_window(mask_windows)
    # convert the masked array to normal array
    trace.data = np.ma.filled(trace.data, fill_value=-1)
    # if the last value is -1, then replace it with the latest value that is not -1
    if trace.data[-1] == -1:
        for i in range(len(trace.data) - 1, -1, -1):
            if trace.data[i] != -1:
                trace.data[-1] = trace.data[i]
                break
    # if the first value is -1, then replace it with the latest value that is not -1
    if trace.data[0] == -1:
        for i in range(len(trace.data)):
            if trace.data[i] != -1:
                trace.data[0] = trace.data[i]
                break
    x_ori = np.arange(0, len(trace.data))
    # not -1 location
    x = np.where(trace.data != -1)[0]
    # not -1 value
    data_not_minus_one = trace.data[x]
    try:
        f = interp1d(x, data_not_minus_one)
        trace.data = f(x_ori)
    except:
        pass
    return trace, mask_windows

def predict_on_trace(trace_path, model, input_length=8192, batch_size=1, det_th = 0.3):
    st = obspy.read(trace_path)
    st.merge()
    trace = st[0]
    trace, final_detections = apollo_linear_interp_gaps(trace)
    trace = trace.detrend('demean')
    trace_data = trace.data
    trace_data = trace_data / np.std(trace_data)
    total_steps = math.ceil(len(trace_data) / input_length) + 1
    current_step = 0
    results = np.zeros_like(trace_data)

    while current_step < total_steps:
        start = current_step * input_length
        end = min((current_step + batch_size) * input_length, len(trace_data))
        if end - start < input_length:
            break
        current_data_tmp = trace_data[start:end].copy()
        # reshape from [batch_size*input_length] to [batch_size, input_length, 1]
        current_data = np.zeros((batch_size, input_length, 1, 1))
        for bdx in range(batch_size):
            current_data[bdx, :, 0, 0] = current_data_tmp[bdx * input_length:(bdx + 1) * input_length]
        current_result = model.predict(current_data)[0]
        for bdx in range(batch_size):
            results[start + bdx * input_length: start + (bdx + 1) * input_length] = current_result[bdx, :,  0]
        current_step += batch_size

        detections = trigger_onset(current_result[bdx, :,  0], det_th, det_th)
        for t_det in detections:
            final_detections.append([t_det[0] + start, t_det[1] + start])
    
    current_data_tmp = trace_data[start:].copy()
    current_data = np.zeros((batch_size, input_length, 1, 1))
    current_data[0, :(end - start), 0, 0] = current_data_tmp[:]
    current_result = model.predict(current_data)[0]
    detections = trigger_onset(current_result[bdx, :(end - start),  0], det_th, det_th)
    for t_det in detections:
        final_detections.append([t_det[0] + start, t_det[1] + start])

    detections = trigger_onset(results[ :], det_th, det_th)
    for t_det in detections:
        final_detections.append([t_det[0], t_det[1]])

    return trace_data, results, final_detections, trace

from PickNet import PickNet_v2
import yaml

def process_and_save(model, trace_path, input_length, det_th, extend_sample_len, trace_save_path, csv_save_path):
    trace_data, results, final_detections, trace = predict_on_trace(trace_path, model, input_length, det_th = det_th)
    final_detections = extend_window(final_detections, extend_len=extend_sample_len)
    final_detections = merge_window(final_detections)
    final_detections_in_UTC_time = []
    
    starttime = trace.stats.starttime

    for t_window in final_detections:
        t_start = str(starttime + float(t_window[0])/trace.stats.sampling_rate)
        t_end = str(starttime + float(t_window[1])/trace.stats.sampling_rate)
        final_detections_in_UTC_time.append([t_start, t_end])

    with open(csv_save_path, 'w') as csv_file:
        for t_window in final_detections_in_UTC_time:
            csv_file.write('{},{}\n'.format(t_window[0], t_window[1]))

    trace.write(trace_save_path, format='MSEED')
    
    return trace_data, results, final_detections, trace

if __name__ == '__main__':
    cfgs = yaml.load(open('./disturbance_detection.yaml','r'), Loader=yaml.SafeLoader)
    model = PickNet_v2(cfgs)
    input_length = cfgs['PickNet']['length']
    model_path = 'disturbance_detection_used_in_paper.h5'
    model.load_weights(model_path)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfgs['Process']['gpu_id']
    try:
        tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    except:
        pass
    trace_path = cfgs['Process']['trace_path']
    trace_save_path = cfgs['Process']['trace_save_path']
    csv_save_path =  cfgs['Process']['csv_save_path']
    det_th = cfgs['Process']['det_th']
    extend_sample_len = cfgs['Process']['extend_sample_len']
    process_and_save(model, trace_path, input_length, det_th, extend_sample_len, trace_save_path, csv_save_path)
    print('Results saved to {} and {}'.format(trace_save_path, csv_save_path))

