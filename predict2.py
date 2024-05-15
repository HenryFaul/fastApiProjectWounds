from keras.models import load_model
from models.deeplab import Deeplabv3, relu6, BilinearUpsampling, DepthwiseConv2D
from utils.learning.metrics import dice_coef, precision, recall
from utils.io.data_custom import save_results_custom, save_rgb_results, save_history, load_test_images, DataGenCustom

# settings
input_dim_x = 224
input_dim_y = 224
color_space = 'rgb'
weight_file_name = '2019-12-19 01%3A53%3A15.480800.hdf5'
pred_save_path = './static/picture_uploads/generated_predictions/'


def make_prediction(current_path, base_url):
    data_gen = DataGenCustom(current_path+"/", split_ratio=0.0, x=input_dim_x, y=input_dim_y, color_space=color_space)
    x_test, test_label_filenames_list = load_test_images(current_path+"/")

    # ### get mobilenetv2 model
    model = Deeplabv3(input_shape=(input_dim_x, input_dim_y, 3), classes=1)
    model = load_model('./training_history/' + weight_file_name
                       , custom_objects={'recall': recall,
                                         'precision': precision,
                                         'dice_coef': dice_coef,
                                         'relu6': relu6,
                                         'DepthwiseConv2D': DepthwiseConv2D,
                                         'BilinearUpsampling': BilinearUpsampling})

    for image_batch, label_batch in data_gen.generate_data(batch_size=len(x_test), test=True):
        prediction = model.predict(image_batch, verbose=1)
        res = save_results_custom(prediction, 'rgb', pred_save_path, test_label_filenames_list, base_url)
        return res
