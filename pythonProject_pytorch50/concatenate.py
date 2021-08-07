import scipy.io as sio
import numpy as np

translationgap_modify_bandmatrix_0628_inference_result_0_100 = sio.loadmat(f'translationgap_modify_bandmatrix_0628_inference_result_0_100.mat')
translationgap_modify_bandmatrix_0628_inference_result_101_197 = sio.loadmat(f'translationgap_modify_bandmatrix_0628_inference_result_101_197.mat')

label_10_31_translationgap_modify_bandmatrix_0628_inference_result_0_100 = translationgap_modify_bandmatrix_0628_inference_result_0_100['label_10_31']
label_10_31_translationgap_modify_bandmatrix_0628_inference_result_101_197 = translationgap_modify_bandmatrix_0628_inference_result_101_197['label_10_31']
label_10_31 = np.concatenate([label_10_31_translationgap_modify_bandmatrix_0628_inference_result_0_100, label_10_31_translationgap_modify_bandmatrix_0628_inference_result_101_197], axis=0)

label_50_50_translationgap_modify_bandmatrix_0628_inference_result_0_100 = translationgap_modify_bandmatrix_0628_inference_result_0_100['label_50_50']
label_50_50_translationgap_modify_bandmatrix_0628_inference_result_101_197 = translationgap_modify_bandmatrix_0628_inference_result_101_197['label_50_50']
label_50_50 = np.concatenate([label_50_50_translationgap_modify_bandmatrix_0628_inference_result_0_100, label_50_50_translationgap_modify_bandmatrix_0628_inference_result_101_197], axis=0)

pre_50_50_translationgap_modify_bandmatrix_0628_inference_result_0_100 = translationgap_modify_bandmatrix_0628_inference_result_0_100['pre_50_50']
pre_50_50_translationgap_modify_bandmatrix_0628_inference_result_101_197 = translationgap_modify_bandmatrix_0628_inference_result_101_197['pre_50_50']
pre_50_50 = np.concatenate([pre_50_50_translationgap_modify_bandmatrix_0628_inference_result_0_100, pre_50_50_translationgap_modify_bandmatrix_0628_inference_result_101_197], axis=0)

pre_10_31_translationgap_modify_bandmatrix_0628_inference_result_0_100 = translationgap_modify_bandmatrix_0628_inference_result_0_100['pre_10_31']
pre_10_31_translationgap_modify_bandmatrix_0628_inference_result_101_197 = translationgap_modify_bandmatrix_0628_inference_result_101_197['pre_10_31']
pre_10_31 = np.concatenate([pre_10_31_translationgap_modify_bandmatrix_0628_inference_result_0_100, pre_10_31_translationgap_modify_bandmatrix_0628_inference_result_101_197], axis=0)

noise_10_31_translationgap_modify_bandmatrix_0628_inference_result_0_100 = translationgap_modify_bandmatrix_0628_inference_result_0_100['noise_10_31']
noise_10_31_translationgap_modify_bandmatrix_0628_inference_result_101_197 = translationgap_modify_bandmatrix_0628_inference_result_101_197['noise_10_31']
noise_10_31 = np.concatenate([noise_10_31_translationgap_modify_bandmatrix_0628_inference_result_0_100, noise_10_31_translationgap_modify_bandmatrix_0628_inference_result_101_197], axis=0)

noise_label_10_31_translationgap_modify_bandmatrix_0628_inference_result_0_100 = translationgap_modify_bandmatrix_0628_inference_result_0_100['noise_label_10_31']
noise_label_10_31_translationgap_modify_bandmatrix_0628_inference_result_101_197 = translationgap_modify_bandmatrix_0628_inference_result_101_197['noise_label_10_31']
noise_label_10_31 = np.concatenate([noise_label_10_31_translationgap_modify_bandmatrix_0628_inference_result_0_100, noise_label_10_31_translationgap_modify_bandmatrix_0628_inference_result_101_197], axis=0)


sio.savemat('inference.mat', {'label_10_31': label_10_31,
                              'label_50_50': label_50_50,
                              'pre_50_50': pre_50_50,
                              'pre_10_31': pre_10_31,
                              'noise_10_31': noise_10_31,
                              'noise_label_10_31': noise_label_10_31})