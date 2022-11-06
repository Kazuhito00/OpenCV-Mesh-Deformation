# OpenCV-Mesh-Deformation
OpenCVを用いてメッシュ変形を行うサンプルです。
<img src="https://user-images.githubusercontent.com/37477845/200170883-b6575825-53b4-40e9-be93-6f7fa04cdf77.gif" width="90%">

# Requirement
* OpenCV 4.5.3.56 or later
* scipy 1.6.2 or later ※01_calc_mesh_deformation_parameter.pyのみ

# Specification
2つのスクリプトで構成しています。<br>
* 01_calc_mesh_deformation_parameter.py<br>
GUI上で変形元と変形後のメッシュを操作して画像を確認するスクリプト<br>
Esc押下でスクリプトを終了して、<br>
メッシュ変形用パラメータ(param_mesh_deformation.npz)、座標変換用マップ(param_inverse_map.npz)を保存します。<br>

https://user-images.githubusercontent.com/37477845/200170512-9be80ed5-b6fb-4f6c-9fd7-4a75853cb0e5.mp4

* 02_mesh_deformation_sample.py<br>
メッシュ変形用パラメータ(param_mesh_deformation.npz)、座標変換用マップ(param_inverse_map.npz)を読み込み、<br>
画像を変形します。<br>
また、入力画像上のマウス位置を変換して出力画像上に青点を描画します。<br>

https://user-images.githubusercontent.com/37477845/200170527-0277cf76-f7f9-45c2-b02d-56574f8f0c63.mp4

# Usage
実行方法は以下です。
```bash
python 01_calc_mesh_deformation_parameter.py
```
* --image<br>
処理画像<br>
デフォルト：test.jpg
* --grid_line_num<br>
グリッド数<br>
デフォルト：3

```bash
python 02_mesh_deformation_sample.py
```
* --image<br>
処理画像<br>
デフォルト：test.jpg
* --mesh_deformation_parameter<br>
メッシュ変形用パラメータ読み込みパス<br>
デフォルト：'param_mesh_deformation.npz'
* --inverse_map<br>
座標変換用マップ用パラメータ読み込みパス<br>
デフォルト：'param_inverse_map.npz'

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
OpenCV-Mesh-Deformation is under [Apache-2.0 License](LICENSE).

# License(Image)
交差点の画像は[フリー素材ぱくたそ](https://www.pakutaso.com)様の写真を利用しています。
