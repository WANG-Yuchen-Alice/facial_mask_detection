import re
import sys
import urllib
import requests


def get_onepage_urls(onepageurl):
    """Gets the urls of all images of a single flip + the url of the next flip of the current flip"""
    if not onepageurl:
        print('It\'s the last page. It\'s over')
        return [], ''
    try:
        html = requests.get(onepageurl)
        html.encoding = 'utf-8'
        html = html.text
    except Exception as e:
        print(e)
        pic_urls = []
        fanye_url = ''
        return pic_urls, fanye_url
    pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)
    fanye_urls = re.findall(re.compile(r'<a href="(.*)" class="n">下一页</a>'), html, flags=0)
    fanye_url = 'http://image.baidu.com' + fanye_urls[0] if fanye_urls else ''
    return pic_urls, fanye_url


def down_pic(pic_urls):
    """Give a list of image links and download all images"""
    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=15)
            string = str(i + 1) + '.jpg'
            with open(string, 'wb') as f:
                f.write(pic.content)
                print('The %s image was successfully downloaded: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            print('Failed to download the %s image: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue


if __name__ == '__main__':
    keyword = '汽车'
    url_init_first = r'http://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1497491098685_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1497491098685%5E00_1519X735&word='
    url_init = url_init_first + urllib.parse.quote(keyword, safe='/')
    all_pic_urls = []
    onepage_urls, fanye_url = get_onepage_urls(url_init)
    all_pic_urls.extend(onepage_urls)

    fanye_count = 0  # Total number of page turns
    while 1:
        onepage_urls, fanye_url = get_onepage_urls(fanye_url)
        fanye_count += 1
        if fanye_url == '' and onepage_urls == []:
            break
        all_pic_urls.extend(onepage_urls)

    down_pic(list(set(all_pic_urls)))
