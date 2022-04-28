files = ["C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-10770759614217273359_1465_000_1485_000_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-10676267326664322837_311_180_331_180_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-11343624116265195592_5910_530_5930_530_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-1146261869236413282_1680_000_1700_000_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-11799592541704458019_9828_750_9848_750_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-10241508783381919015_2889_360_2909_360_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-10094743350625019937_3420_000_3440_000_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-11379226583756500423_6230_810_6250_810_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-11588853832866011756_2184_462_2204_462_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-11219370372259322863_5320_000_5340_000_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-11454085070345530663_1905_000_1925_000_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-10927752430968246422_4940_000_4960_000_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-10391312872392849784_4099_400_4119_400_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-10153695247769592104_787_000_807_000_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-11070802577416161387_740_000_760_000_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-11486225968269855324_92_000_112_000_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-11236550977973464715_3620_000_3640_000_with_camera_labels.tfrecord",
    "C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\training_and_validation\\segment-12027892938363296829_4086_280_4106_280_with_camera_labels.tfrecord"]
import os
import shutil

for f in files:
    shutil.move(f, os.path.join("C:\\Users\\arthe\\Desktop\\Object_Detection_in_Urban_Environment\\data\\val", os.path.basename(f)))