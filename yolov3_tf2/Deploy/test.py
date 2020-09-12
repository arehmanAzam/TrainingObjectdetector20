# import cv2
# # url = 'http://deployface.com/'
# url = 'http://127.0.0.1:5000'
# img = cv2.imread('55/3.jpg')
# # encode image as jpeg
# _, img_encoded = cv2.imencode('.jpg', img)
# files = {'media1': img_encoded.tostring()}
# print(requests.post(url, files=files).text)
import requests
import json
import base64
from timeit import default_timer as timer
import glob
import time


def base64ToString(b):
    return base64.b64decode(b).decode('utf-16')

def main():
    try:
        test_url = 'http://127.0.0.1:5000'
        #test_url = 'http://deployface.xavor.com'
        # test_url='http://47.242.134.221/bin'
        # prepare headers for http request


        content_type = 'image/jpeg'
        headers = {'content-type': content_type}
        # image = open('imgs/6.jpg', 'rb')
        # image_read=image.read()
        # image_encoded=base64.encodestring(image_read)
        #img = cv2.imread('55/3.jpg')
        my_imagelist = [f for f in glob.glob('./test_images' + "/*.*")]
        for image_path in my_imagelist:
            # with open("imgs/1.jpg", "rb") as image_file:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
            # encode image as jpeg
            print("Length of the string: ")
            print(len(encoded_string))
            #_, img_encoded = cv2.imencode('.jpg', img)
            # send http request with image and receive response
            # print(type(encoded_string))
            # while(1):
            start = timer()
            # response = requests.post(test_url, data=encoded_string, headers=headers)
            res = requests.post(test_url, json={"Image": encoded_string.decode()
                                                ,"time_sent":time.strftime('%Y-%m-%d %H:%M:%S')
                                                ,"Image_resolution": "1280x800"
                                                ,"bytes_sent": 2334})

            end = timer()
            # json_formatted_str = json.dumps(response, indent=2)
            if res.ok:
                print("image: "+ image_path)
                print("Response :" )
                print("Response Time: " + str(end-start))
                print(res.json())
            break
    except Exception as e:
        print('Exception : %s' %e)

if __name__ == '__main__':
    main()