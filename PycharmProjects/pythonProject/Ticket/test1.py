from selenium import webdriver
import logging
import requests





# logger = logging.getLogger('dmv_logs')
# logger.setLevel(logging.DEBUG)
# file_handler = logging.FileHandler('scrapes.log')
# formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
#
# # with open("config.json", "r") as jsonfile:
# #     config = json.load(jsonfile)
#
#
#
#
#
#
# # 大麦网主页
# damai_url = 'https://www.damai.cn/'
# # 登录
# login_url = 'https://passport.damai.cn/login?ru=https%3A%2F%2Fwww.damai.cn%2F'
# # 抢票目标页
# target_url = 'https://detail.damai.cn/item.htm?spm=a2oeg.search_category.searchtxt.ditem_1.51a156e0uWH3BW&id=731600651574'
#
#
import requests
import time

url = 'http://api.github.com/users/ssaunier'
max_retries = 3
retry_delay = 5

for _ in range(max_retries):
    try:
        response = requests.get(url).json()
        print(response['name'])
        break  # Break the loop if successful
    except requests.exceptions.SSLError as e:
        print("SSL Error:", e)
        print("Retrying in", retry_delay, "seconds...")
        time.sleep(retry_delay)
else:
    print("Max retries exceeded. Could not fetch data.")


# from cryptography import x509
# from cryptography.hazmat.backends import default_backend
# import ssl
# import socket
#
# def check_certificate(hostname, port=443):
#     context = ssl.create_default_context()
#     with context.wrap_socket(socket.socket(socket.AF_INET), server_hostname=hostname) as s:
#         cert = x509.load_der_x509_certificate(s.getpeercert(True), default_backend())
#         print("Certificate Subject:", cert.subject)
#         print("Issuer:", cert.issuer)
#         print("Valid From:", cert.not_valid_before)
#         print("Valid Until:", cert.not_valid_after)
#
# check_certificate("example.com")





# response = requests.get(damai_url).json()

# print(response)

# driver = webdriver.Chrome()  # 调用chrome浏览器
#
# driver.get(damai_url)
#
# print(driver.title)
#
# driver.quit()