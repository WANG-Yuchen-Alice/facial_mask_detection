from selenium import webdriver
import time
import os
import requests

# How to use the proxy: You can use the proxy directly from windows
# browserOptions = webdriver.ChromeOptions()
# browserOptions.add_argument('--proxy-server=ip:port)
# browser = webdriver.Chrome(chrome_options=browserOptions)

#By modifying keyword, you can modify the search keyword
global keyword
keyword = 'mask+people'
url = 'https://www.google.com.hk/search?q='+keyword+'&tbm=isch'


class Crawler_google_images:
    def __init__(self):
        self.url = url

    # Get the Chrome driver and visit the url
    def init_browser(self):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--disable-infobars")
        browser = webdriver.Chrome(chrome_options=chrome_options)
        # visit the url
        browser.get(self.url)
        # Maximize the window, and then you need to climb all the images you see in the window
        browser.maximize_window()
        return browser

    # Download the photo
    def download_images(self, browser,round=2):
        picpath = './keyword'
        # Create one if the path does not exist
        if not os.path.exists(picpath): os.makedirs(picpath)
        # Record downloaded image addresses to avoid repeated downloads
        img_url_dic = []

        count = 0 # photo index
        pos = 0
        for i in range(round):
            pos += 500
            # Slide down
            js = 'var q=document.documentElement.scrollTop=' + str(pos)
            browser.execute_script(js)
            time.sleep(2)
            # Find the photo

            img_elements = browser.find_elements_by_tag_name('img')
            #Go through the web Element
            for img_element in img_elements:
                img_url = img_element.get_attribute('src')
                # The urls of the first few images are too long. They are not the urls of the images. Filter them out first and climb the ones behind them
                if isinstance(img_url, str):
                    if len(img_url) <= 200:
                        # Sift out the offending goole ICONS
                        if 'images' in img_url:
                            # Determine if you have already crawled, because each time you crawl to the current window, it may be repeated
                            # Actually, I could change this to save money by just storing the last url in the list, but I'm tired of writing it···
                            if img_url not in img_url_dic:
                                try:
                                    img_url_dic.append(img_url)
                                    # Download and save the image to the current directory
                                    filename = "./cat/" + str(count) + ".jpg"
                                    r = requests.get(img_url)
                                    with open(filename, 'wb') as f:
                                        f.write(r.content)
                                    f.close()
                                    count += 1
                                    print('this is '+str(count)+'st img')
                                    # Prevent anti-crawl mechanisms
                                    time.sleep(0.2)
                                except:
                                    print('failure')

    def run(self):
        self.__init__()
        browser = self.init_browser()
        self.download_images(browser,1000) # You can modify the number of crawling pages, the basic 10 pages is more than 100 pictures
        browser.close()
        print("Scratch finished")


if __name__ == '__main__':
    craw = Crawler_google_images()
    craw.run()
