import copy
import argparse

import cv2
import numpy as np
from scipy.interpolate import griddata


class CvWindow(object):
    def __init__(self, window_name='CvWindow'):
        self._window_name = window_name
        self._mouse_point = None
        self._mouse_l_down_flag = False

        cv2.namedWindow(self._window_name)
        cv2.setMouseCallback(self._window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self._mouse_point = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_l_down_flag = True
        if event == cv2.EVENT_LBUTTONUP:
            self._mouse_l_down_flag = False

    def get_mouse_point(self):
        return self._mouse_point

    def get_mouse_l_down(self):
        return self._mouse_l_down_flag

    def imshow(self, image):
        cv2.imshow(self._window_name, image)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default='test.jpg')
    parser.add_argument("--grid_line_num", type=int, default=3)

    args = parser.parse_args()

    return args


def main():
    # 引数
    args = get_args()
    image_path = args.image
    grid_line_num = args.grid_line_num

    # 画像読み込み
    original_image = cv2.imread(image_path)

    # ROI選択
    xmin, ymin, width, height = cv2.selectROI(original_image)
    cv2.destroyAllWindows()

    # 入力・出力用のグリッド座標リスト
    input_grid_point_list = []
    for y_index in range(grid_line_num):
        for x_index in range(grid_line_num):
            x = int(xmin + ((width / (grid_line_num - 1)) * x_index))
            y = int(ymin + ((height / (grid_line_num - 1)) * y_index))
            input_grid_point_list.append([x, y])

    output_grid_point_list = copy.deepcopy(input_grid_point_list)

    # 画面生成
    input_window = CvWindow('Input')
    output_window = CvWindow('Output')

    # マウス座標に一番近いグリッドのインデックス
    input_grid_point_index = None
    output_grid_point_index = None

    while True:
        # 入力画面のマウス座標と左ボタン押下状態取得
        input_window_mouse_point = input_window.get_mouse_point()
        input_window_mouse_l_down = input_window.get_mouse_l_down()

        # 左ボタンを押していない時：マウス位置に一番近いグリッド交点を算出
        if not input_window_mouse_l_down:
            input_grid_point_index = calc_nearest_point(
                input_grid_point_list,
                input_window_mouse_point,
            )
        # 左ボタンを押している時：グリッド座標をマウス位置に更新
        elif input_window_mouse_l_down:
            input_grid_point_list[
                input_grid_point_index] = input_window_mouse_point

        # 入力画面のグリッド情報と出力画面のグリッド情報からメッシュ変形を実行
        map_x, map_y = calc_mesh_deformation(
            original_image,
            input_grid_point_list,
            output_grid_point_list,
        )
        deformation_image = cv2.remap(
            original_image,
            map_x,
            map_y,
            cv2.INTER_CUBIC,
        )

        # 出力画面のマウス座標と左ボタン押下状態取得
        output_window_mouse_point = output_window.get_mouse_point()
        output_window_mouse_l_down = output_window.get_mouse_l_down()

        # 左ボタンを押していない時：マウス位置に一番近いグリッド交点を算出
        if not output_window_mouse_l_down:
            output_grid_point_index = calc_nearest_point(
                output_grid_point_list,
                output_window_mouse_point,
            )
        # 左ボタンを押している時：グリッド座標をマウス位置に更新
        elif output_window_mouse_l_down:
            output_grid_point_list[
                output_grid_point_index] = output_window_mouse_point

        # デバッグ情報描画
        debug_image = copy.deepcopy(original_image)
        debug_image = draw_debug_info(
            debug_image,
            input_grid_point_list,
            grid_line_num,
            input_window_mouse_point,
            input_grid_point_index,
        )
        deformation_image = draw_debug_info(
            deformation_image,
            output_grid_point_list,
            grid_line_num,
            output_window_mouse_point,
            output_grid_point_index,
        )

        # 描画
        input_window.imshow(debug_image)
        output_window.imshow(deformation_image)
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

    # 変形パラメータを保存
    export_parameters(
        original_image,
        input_grid_point_list,
        output_grid_point_list,
    )


def calc_nearest_point(grid_point_list, mouse_point):
    min_distance = None
    min_distance_index = None

    # マウス座標に一番近いグリッド座標を探す
    for index, point in enumerate(grid_point_list):
        if mouse_point is not None:
            distance = np.linalg.norm(np.array(point) - np.array(mouse_point))
            if min_distance is None:
                min_distance = distance
                min_distance_index = index
            elif min_distance > distance:
                min_distance = distance
                min_distance_index = index

    return min_distance_index


def calc_mesh_deformation(
    image,
    input_grid_point_list,
    output_grid_point_list,
):
    destination_list = []
    for source_point in output_grid_point_list:
        destination_list.append([source_point[1], source_point[0]])

    width, height = image.shape[1], image.shape[0]

    grid_x, grid_y = np.mgrid[0:height - 1:height * 1j, 0:width - 1:width * 1j]
    destination = np.array(destination_list)
    source = np.array(input_grid_point_list)
    grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')

    map_x = np.append([], [ar[:, 0] for ar in grid_z]).reshape(height, width)
    map_y = np.append([], [ar[:, 1] for ar in grid_z]).reshape(height, width)
    map_x = map_x.astype('float32')
    map_y = map_y.astype('float32')

    return map_x, map_y


def export_parameters(
    image,
    input_grid_point_list,
    output_grid_point_list,
):
    # メッシュ変形パラメータ取得
    mesh_deformation_map_x, mesh_deformation_map_y = calc_mesh_deformation(
        image,
        input_grid_point_list,
        output_grid_point_list,
    )

    # メッシュ変形パラメータ保存
    np.savez(
        'param_mesh_deformation',
        input_grid_point_list,
        output_grid_point_list,
        mesh_deformation_map_x,
        mesh_deformation_map_y,
    )

    # 座標変換マップ生成
    inverse_source_list = []
    inverse_destination_list = []
    width, height = image.shape[1], image.shape[0]
    for y in range(height):
        for x in range(width):
            check_warp_x = np.isnan(mesh_deformation_map_x[y][x])
            check_warp_y = np.isnan(mesh_deformation_map_y[y][x])
            if (not check_warp_x) and (not check_warp_y):
                inverse_source_list.append([x, y])
                inverse_destination_list.append([
                    int(mesh_deformation_map_y[y][x]),
                    int(mesh_deformation_map_x[y][x])
                ])

    inverse_grid_x, inverse_grid_y = np.mgrid[0:height - 1:height * 1j,
                                              0:width - 1:width * 1j]
    inverse_destination = np.array(inverse_destination_list)
    inverse_source = np.array(inverse_source_list)
    inverse_grid_z = griddata(
        inverse_destination,
        inverse_source,
        (inverse_grid_x, inverse_grid_y),
        method='cubic',
    )
    inverse_map_x = np.append([], [ar[:, 0] for ar in inverse_grid_z]).reshape(
        height, width)
    inverse_map_y = np.append([], [ar[:, 1] for ar in inverse_grid_z]).reshape(
        height, width)

    # 座標変換マップ保存
    np.savez(
        'param_inverse_map',
        inverse_map_x,
        inverse_map_y,
    )


def draw_debug_info(
    image,
    grid_point_list,
    grid_line_num,
    mouse_point,
    mouse_point_index,
):
    # グリッド線・交点を描画
    for index, point in enumerate(grid_point_list):
        x_index = int(index % grid_line_num)
        y_index = int(index / grid_line_num)

        cv2.circle(image, point, 4, (0, 255, 0), -1, cv2.LINE_4)

        if index < len(grid_point_list) - 1:
            if x_index < grid_line_num - 1:
                next_point = grid_point_list[index + 1]
                cv2.line(image, point, next_point, (0, 255, 0), 1, cv2.LINE_AA)
            if y_index < grid_line_num - 1:
                next_point = grid_point_list[index + grid_line_num]
                cv2.line(image, point, next_point, (0, 255, 0), 1, cv2.LINE_AA)

    # グリッド交点に近いマウス位置を描画
    if mouse_point_index is not None:
        cv2.circle(image, grid_point_list[mouse_point_index], 4, (0, 0, 255),
                   -1, cv2.LINE_4)

    # マウス位置を描画
    if mouse_point is not None:
        cv2.circle(image, mouse_point, 6, (255, 0, 0), 2, cv2.LINE_AA)

    return image


if __name__ == '__main__':
    main()
