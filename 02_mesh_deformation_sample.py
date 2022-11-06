import time
import argparse

import cv2
import numpy as np


class CvWindow(object):
    def __init__(self, window_name='CvWindow'):
        self._window_name = window_name
        self._mouse_point = None

        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self._mouse_point = [x, y]

    def get_mouse_point(self):
        return self._mouse_point

    def imshow(self, image):
        cv2.imshow(self._window_name, image)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default='test.jpg')
    parser.add_argument(
        "--mesh_deformation_parameter",
        type=str,
        default='param_mesh_deformation.npz',
    )
    parser.add_argument(
        "--inverse_map",
        type=str,
        default='param_inverse_map.npz',
    )

    args = parser.parse_args()

    return args


def main():
    # 引数
    args = get_args()
    image_path = args.image
    mesh_deformation_parameter_path = args.mesh_deformation_parameter
    inverse_map_path = args.inverse_map

    # メッシュ変形パラメータ読み込み
    mesh_deformation_parameter = np.load(mesh_deformation_parameter_path)

    # input_grid_point_list = mesh_deformation_parameter['arr_0']
    # output_grid_point_list = mesh_deformation_parameter['arr_1']
    map_x = mesh_deformation_parameter['arr_2']
    map_y = mesh_deformation_parameter['arr_3']

    # 座標変換マップ読み込み
    inverse_map = np.load(inverse_map_path)

    inverse_map_x = inverse_map['arr_0']
    inverse_map_y = inverse_map['arr_1']

    # 画面生成
    input_window = CvWindow('Input')
    output_window = CvWindow('Output')

    while True:
        start_time = time.time()

        # 画像読み込み
        original_image = cv2.imread(image_path)

        # メッシュ変形実行
        deformation_image = cv2.remap(
            original_image,
            map_x,
            map_y,
            cv2.INTER_CUBIC,
        )

        # マウス座標変換
        input_window_mouse_point = input_window.get_mouse_point()
        if input_window_mouse_point is not None:
            mouse_x = input_window_mouse_point[0]
            mouse_y = input_window_mouse_point[1]
            inverse_x = inverse_map_x[mouse_y][mouse_x]
            inverse_y = inverse_map_y[mouse_y][mouse_x]
            if (not np.isnan(inverse_x)) and (not np.isnan(inverse_y)):
                cv2.circle(deformation_image, (int(inverse_x), int(inverse_y)),
                           4, (255, 0, 0), -1, cv2.LINE_AA)

        elapsed_time = time.time() - start_time

        # フレーム経過時間描画
        elapsed_time_text = "Elapsed time: "
        elapsed_time_text += str(round((elapsed_time * 1000), 1))
        elapsed_time_text += 'ms'
        cv2.putText(deformation_image, elapsed_time_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

        # 描画
        input_window.imshow(original_image)
        output_window.imshow(deformation_image)
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
