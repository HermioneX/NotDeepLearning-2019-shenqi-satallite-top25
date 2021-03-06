# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:08:11 2019

@author: HermoineX_zhanling
"""
import datetime


train_start_time1 = datetime.datetime(2019,7,14,1)
train_end_time1 = datetime.datetime(2019,7,31,23)

test_start_time1 = datetime.datetime(2019,8,1,0)
test_end_time1 = datetime.datetime(2019,8,12,23)

#############  加上有卫星图像的时间（二者交集） ##############  （需要把数据移动过去）
train_start_time2 = datetime.datetime(2019,7,14,14)    # 相比全部的数据，从14日14点开始（BJT）
train_end_time2 = datetime.datetime(2019,7,31,23)

test_start_time2 = datetime.datetime(2019,8,1,0)
test_end_time2 = datetime.datetime(2019,8,12,23)   # 10.17 目前和test时间一致，可以全覆盖

picR = [[3, 53.5], [59, 138]]


src_path = './src_data/'

sate_path = './sate_data/'

# train_path = './train_data/'
# test_path = './test_data/'

# train_sate = './train_sate/'
# test_sate = './test_sate/'

dazi_test = [0,0,0,0,0,0,37,63,209,298,484,626,189,98,324,427,656,34,0,0,0,0,0,0] #直接取的前
dazi_train = [324,427,656,34,0,0,0,0,0,0,0,0,0,0,0,0,37,63,209,298,484,626,189,98]

img_train_path = src_path + '20190714-20190730/data/'
img_test_path = src_path + '20190731-20190813/data/'

clean_img_path = './sate_data/clean_sate/'

img_head = 'SEVP_NSMC_WXBL_FY4A_ETCC_ACHN_LNO_PY_'
img_tail = '00000.JPG'

zh_name = ['中卫', '乐亭', '佳木斯', '保定', '南昌', '吐鲁番', '大连', '安康', '宜昌', '张北', '攀枝花',
       '新乡', '昆明', '晋源', '淮安', '福州', '索伦', '荔湾', '西宁', '西青', '贵阳', '赣州', '达孜',
       '通辽', '酒泉', '银川', '长春', '青秀', '驻马店', '鼓楼']
py_name = ['zhongwei','laoting','jiamusi','baoding','nanchang','tulufan','dalian','ankang',\
        'yichang','zhangbei','panzhihua','xinxiang','kunming','jinyuanqu','huaian1','fuzhou',\
        'xinganmeng','liwan','xining','xiqing','guiyang1','ganzhou','dazi','tongliao','jiuquan','yinchuan',\
        'changchun', 'qingxiu','zhumadian','nanjinggulou']   # 索伦用兴安盟站信息代替


st_min_max_dic = dict({'TRI':[0,1500],
                      'hour':[0,23],
                      'maxT':[10,45],
                      'minT':[0,35],
                      'windSpeed':[0,8],
                      'cloudy':[0,1],
                      'dark':[0,1],
                      'light':[0,1],
                      'medium':[0,1],
                      'heavy':[0,1],
                      'other':[0,1],
                      'sun':[0,1],
                      'cloud_mean':[0,255],
                      'cloud_max':[0,255],
                      'cloud_min':[0,255],
                      'cloud_std':[0,100],  # 统计出来的
                      'cloud_25':[0,255],
                      'cloud_50':[0,255],
                      'cloud_75':[0,255],
                      'cloud_frac_mean':[0,1],
                      'cloud_frac1':[0,1],
                      'cloud_frac2':[0,1]})

latlon_list = [[37.5,105.2], #中卫
               [39.4,118.9],#乐亭
               [46.8,130.3],#佳木斯
               [38.9,115.4],#保定
               [28.6,115.8], #南昌
               [42.9,89.1],#吐鲁番
               [38.9,121.6],#大连
               [32.6,109.0],#安康
               [30.7,111.2],#宜昌
               [41.1,114.7],#张北
               [26.6,101.6],#攀枝花
               [35.3,113.8],#新乡
               [24.8,102.9],#昆明
               [37.7,112.4],#晋源
               [37.7,112.4],#淮安
               [26.0,119.3],#福州
               [46.6,121.2],#索伦,兴安盟
               [23.0,113.2],#荔湾
               [36.6,101.8],#西宁
               [39.1,117.0],#西青
               [26.6,106.6],#贵阳
               [25.8,114.9],#赣州
               [29.6,91.3],#达孜
               [43.6,122.2],#通辽
               [39.7,98.4],#酒泉
               [38.4,106.2],#银川
               [43.7,125.2],#长春
               [22.7,108.5],#青秀
               [33.0,114.0],#驻马店
               [32.0,118.7],#南京鼓楼
               ]

# latlon_list = [[37.5，105.2,], #中卫
#                [39.4,118.9],#乐亭
#                [46.8,130.3],#佳木斯
#                [38.9,115.4],#保定
#                [28.6,115.8], #南昌
#                [89.190113,42.981472],#吐鲁番
#                [121.631689,38.913563],#大连
#                [109.022093,32.681761],#安康
#                [111.246846,30.715331],#宜昌
#                [114.745639,41.178412],#张北
#                [101.66604,26.608192],#攀枝花
#                [113.89824,35.310582],#新乡
#                [102.930352,24.843573],#昆明
#                [112.448635,37.740348],#晋源
#                [112.448635,37.740348],#淮安
#                [119.338827,26.081987],#福州
#                [121.299119,46.616822],#索伦,兴安盟
#                [113.236125,23.040832],#荔湾
#                [101.802556,36.607163],#西宁
#                [117.029068,39.139773],#西青
#                [106.631118,26.662106],#贵阳
#                [114.90745,25.836217],#赣州
#                [91.352577,29.675408],#达孜
#                [122.25081,43.675092],#通辽
#                [98.476247,39.742463],#酒泉
#                [106.222111,38.450879],#银川
#                [125.28605,43.796973],#长春
#                [108.510347,22.788496],#青秀
#                [114.029904,33.006216],#驻马店
#                [118.776599,32.072632],#南京鼓楼
#                ]
