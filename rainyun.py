import logging
import os
import random
import re
import time
import subprocess
import sys

import cv2
import ddddocr
import requests
from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


try:
    from webdriver_manager.chrome import ChromeDriverManager
    try:
        from webdriver_manager.core.utils import ChromeType
    except ImportError:
        try:
            from webdriver_manager.chrome import ChromeType
        except ImportError:
            ChromeType = None
except ImportError:
    print("webdriver_manager未安装，将使用备用方式")
    ChromeDriverManager = None
    ChromeType = None

try:
    from notify import send
    print("已加载通知模块 (notify.py)")
except ImportError:
    print("警告: 未找到 notify.py，将无法发送通知。")
    def send(*args, **kwargs):
        pass


def init_selenium(debug=False, headless=False):
    ops = webdriver.ChromeOptions()
    
    if headless or os.environ.get("GITHUB_ACTIONS", "false") == "true":
        for option in ['--headless', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']:
            ops.add_argument(option)
    
    ops.add_argument('--window-size=1920,1080')
    ops.add_argument('--disable-blink-features=AutomationControlled')
    ops.add_argument('--no-proxy-server')
    ops.add_argument('--lang=zh-CN')
    
    is_github_actions = os.environ.get("GITHUB_ACTIONS", "false") == "true"
    
    if debug and not is_github_actions:
        ops.add_experimental_option("detach", True)
    
    try:
        print("尝试直接使用系统ChromeDriver...")
        driver = webdriver.Chrome(options=ops)
        print("成功使用系统ChromeDriver")
        return driver
    except Exception as e:
        print(f"系统ChromeDriver失败: {e}")
    
    try:
        print("尝试使用webdriver-manager...")
        if ChromeDriverManager:
            if ChromeType and hasattr(ChromeType, 'GOOGLE'):
                manager = ChromeDriverManager(chrome_type=ChromeType.GOOGLE)
            else:
                manager = ChromeDriverManager()
            
            driver_path = manager.install()
            print(f"获取到ChromeDriver路径: {driver_path}")
            service = Service(driver_path)
            driver = webdriver.Chrome(service=service, options=ops)
            print("成功使用webdriver-manager")
            return driver
        else:
            raise ImportError("webdriver_manager未安装")
    except Exception as e:
        print(f"webdriver-manager失败: {e}")
    
    try:
        print("尝试使用备用ChromeDriver路径...")
        common_paths = ['/usr/local/bin/chromedriver', '/usr/bin/chromedriver', './chromedriver', 'chromedriver']
        for path in common_paths:
            try:
                service = Service(path)
                driver = webdriver.Chrome(service=service, options=ops)
                print(f"成功使用备用路径: {path}")
                return driver
            except:
                continue
    except Exception as e:
        print(f"备用路径失败: {e}")
    
    print("错误: 无法初始化ChromeDriver，请检查Chrome和ChromeDriver的安装")
    if is_github_actions:
        print("在GitHub Actions环境中，尝试安装ChromeDriver...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'chromedriver-binary-auto'])
            import chromedriver_binary
            driver = webdriver.Chrome(options=ops)
            print("成功使用chromedriver-binary-auto")
            return driver
        except Exception as e:
            print(f"备用安装失败: {e}")
    
    raise Exception("无法初始化Selenium WebDriver")

def download_image(url, filename):
    os.makedirs("temp", exist_ok=True)
    try:
        response = requests.get(url, timeout=10, proxies={"http": None, "https": None}, verify=False)
        if response.status_code == 200:
            path = os.path.join("temp", filename)
            with open(path, "wb") as f:
                f.write(response.content)
            return True
        else:
            logger.error(f"下载图片失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"下载图片异常: {str(e)}")
        return False


def get_url_from_style(style):
    return re.search(r'url\(["\']?(.*?)["\']?\)', style).group(1)


def get_width_from_style(style):
    return re.search(r'width:\s*([\d.]+)px', style).group(1)


def get_height_from_style(style):
    return re.search(r'height:\s*([\d.]+)px', style).group(1)


def download_captcha_img():
    """改进的验证码图片下载函数，添加更详细的日志"""
    if os.path.exists("temp"):
        for filename in os.listdir("temp"):
            file_path = os.path.join("temp", filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
    
    try:
        logger.info("[图片下载] 开始下载验证码图片(1)")
        slideBg = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="slideBg"]')), timeout=10)
        img1_style = slideBg.get_attribute("style")
        img1_url = get_url_from_style(img1_style)
        logger.info(f"[图片下载] 验证码URL: {img1_url}")
        
        if download_image(img1_url, "captcha.jpg"):
            logger.info("[图片下载] ✓ 验证码图片(1)下载成功")
        else:
            logger.error("[图片下载] 验证码图片(1)下载失败")
            raise Exception("验证码图片下载失败")
        
        logger.info("[图片下载] 开始下载验证码图片(2)")
        sprite = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="instruction"]/div/img')), timeout=10)
        img2_url = sprite.get_attribute("src")
        logger.info(f"[图片下载] 精灵图URL: {img2_url}")
        
        if download_image(img2_url, "sprite.jpg"):
            logger.info("[图片下载] ✓ 验证码图片(2)下载成功")
        else:
            logger.error("[图片下载] 验证码图片(2)下载失败")
            raise Exception("精灵图下载失败")
            
    except Exception as e:
        logger.error(f"[图片下载] 下载异常: {e}")
        raise


def check_captcha() -> bool:
    """改进的验证码检查函数"""
    try:
        raw = cv2.imread("temp/sprite.jpg")
        if raw is None:
            logger.error("无法读取验证码图片")
            return False
        
        h, w = raw.shape[:2]
        if h < 50 or w < 100:
            logger.warning(f"验证码图片尺寸过小: {w}x{h}")
            return False
        
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian < 50:
            logger.warning(f"验证码图片清晰度不足: {laplacian}")
            return False
            
        for i in range(3):
            w_segment = w // 3
            start_x = max(0, w_segment * i + 2)
            end_x = min(w, w_segment * (i + 1) - 2)
            temp = raw[:, start_x:end_x]
            cv2.imwrite(f"temp/sprite_{i + 1}.jpg", temp)
            
            with open(f"temp/sprite_{i + 1}.jpg", mode="rb") as f:
                temp_rb = f.read()
            try:
                result = ocr.classification(temp_rb)
                if result in ["0", "1"]:
                    logger.warning(f"发现无效验证码: sprite_{i + 1}.jpg = {result}")
                    return False
            except Exception as e:
                logger.error(f"OCR识别出错: {e}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"验证码检查失败: {e}")
        return False


def check_answer(d: dict) -> bool:
    """改进的答案检查函数，不仅检查重复标记，还检查相似度阈值"""
    flipped = dict()
    for key in d.keys():
        flipped[d[key]] = key
    
    if len(d.values()) != len(flipped.keys()):
        return False
    
    min_similarity_threshold = 0.3
    for i in range(3):
        similarity_key = f"sprite_{i + 1}.similarity"
        if similarity_key in d and float(d[similarity_key]) < min_similarity_threshold:
            logger.warning(f"相似度不足: {similarity_key} = {d[similarity_key]}")
            return False
    
    positions = []
    for i in range(3):
        position_key = f"sprite_{i + 1}.position"
        if position_key in d:
            x, y = map(int, d[position_key].split(","))
            positions.append((x, y))
    
    if len(positions) == 3:
        x_coords = [p[0] for p in positions]
        x_range = max(x_coords) - min(x_coords)
        
        if x_range < 50:
            logger.warning(f"位置分布过于集中: x范围 = {x_range}")
            return False
    
    return True


def preprocess_image(image):
    """图像预处理函数，提高特征匹配准确率"""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def compute_similarity(img1_path, img2_path):
    """优化的相似度计算函数"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1.shape[0] > 100 or img1.shape[1] > 100:
        scale = 100.0 / max(img1.shape)
        img1 = cv2.resize(img1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    if img2.shape[0] > 100 or img2.shape[1] > 100:
        scale = 100.0 / max(img2.shape)
        img2 = cv2.resize(img2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0.0, 0

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        if len(good) == 0:
            return 0.0, 0

        feature_factor = min(1.0, len(kp1) / 100.0, len(kp2) / 100.0)
        match_ratio = len(good) / min(len(des1), len(des2))
        
        similarity = match_ratio * 0.7 + feature_factor * 0.3
        
        return similarity, len(good)
    except Exception as e:
        logger.error(f"相似度计算出错: {e}")
        return 0.0, 0


def process_captcha():
    """改进的验证码处理函数"""
    try:
        logger.info("[验证码处理] 开始处理验证码")
        
        download_captcha_img()
        
        if check_captcha():
            logger.info("[验证码处理] 开始识别验证码")
            captcha = cv2.imread("temp/captcha.jpg")
            
            if captcha is None:
                logger.error("[验证码处理] 验证码图片为空")
                return False
            
            with open("temp/captcha.jpg", 'rb') as f:
                captcha_b = f.read()
            
            bboxes = det.detection(captcha_b)
            logger.info(f"[验证码处理] 检测到 {len(bboxes)} 个目标")
            
            result = dict()
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[i]
                spec = captcha[y1:y2, x1:x2]
                cv2.imwrite(f"temp/spec_{i + 1}.jpg", spec)
                for j in range(3):
                    similarity, matched = compute_similarity(f"temp/sprite_{j + 1}.jpg", f"temp/spec_{i + 1}.jpg")
                    similarity_key = f"sprite_{j + 1}.similarity"
                    position_key = f"sprite_{j + 1}.position"
                    if similarity_key in result.keys():
                        if float(result[similarity_key]) < similarity:
                            result[similarity_key] = similarity
                            result[position_key] = f"{int((x1 + x2) / 2)},{int((y1 + y2) / 2)}"
                    else:
                        result[similarity_key] = similarity
                        result[position_key] = f"{int((x1 + x2) / 2)},{int((y1 + y2) / 2)}"
            
            if check_answer(result):
                logger.info("[验证码处理] 识别结果验证通过")
                for i in range(3):
                    similarity_key = f"sprite_{i + 1}.similarity"
                    position_key = f"sprite_{i + 1}.position"
                    position = result[position_key]
                    logger.info(f"[验证码处理] 图案 {i + 1} 位于 ({position})，匹配率：{result[similarity_key]}")
                    
                    try:
                        slideBg = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="slideBg"]')))
                        style = slideBg.get_attribute("style")
                        x, y = int(position.split(",")[0]), int(position.split(",")[1])
                        width_raw, height_raw = captcha.shape[1], captcha.shape[0]
                        width, height = float(get_width_from_style(style)), float(get_height_from_style(style))
                        x_offset, y_offset = float(-width / 2), float(-height / 2)
                        final_x, final_y = int(x_offset + x / width_raw * width), int(y_offset + y / height_raw * height)
                        
                        logger.info(f"[验证码处理] 点击位置: ({final_x}, {final_y})")
                        ActionChains(driver).move_to_element_with_offset(slideBg, final_x, final_y).click().perform()
                        time.sleep(1)
                    except Exception as e:
                        logger.error(f"[验证码处理] 点击失败: {e}")
                
                try:
                    confirm = wait.until(
                        EC.element_to_be_clickable((By.XPATH, '//*[@id="tcStatus"]/div[2]/div[2]/div/div')))
                    logger.info("[验证码处理] 提交验证码")
                    confirm.click()
                    time.sleep(5)
                    
                    result_elem = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="tcOperation"]')))
                    if result_elem.get_attribute("class") == 'tc-opera pointer show-success':
                        logger.info("[验证码处理] ✓ 验证码通过")
                        return True
                    else:
                        logger.error("[验证码处理] 验证码未通过，正在重试")
                        return False
                except Exception as e:
                    logger.error(f"[验证码处理] 提交验证码失败: {e}")
                    return False
            else:
                logger.error("[验证码处理] 验证码识别失败，正在重试")
                return False
        else:
            logger.error("[验证码处理] 验证码质量检查失败，尝试刷新")
            return False
            
    except TimeoutException:
        logger.error("[验证码处理] 获取验证码元素超时")
        return False
    except Exception as e:
        logger.error(f"[验证码处理] 处理异常: {e}", exc_info=True)
        return False


def sign_in_account(user, pwd, debug=False, headless=False):
    """
    单个账户登录签到函数
    
    Args:
        user: 用户名
        pwd: 密码
        debug: 是否开启调试模式
        headless: 是否使用无头模式
        
    Returns:
        tuple: (成功状态, 用户名, 积分信息, 错误信息)
    """
    global driver, wait, ocr, det
    timeout = 15
    driver = None
    
    try:
        logger.info(f"开始处理账户: {user}")
        
        if not debug:
            delay_sec = random.randint(5, 10)
            logger.info(f"随机延时等待 {delay_sec} 秒")
            time.sleep(delay_sec)
        
        logger.info("初始化 ddddocr")
        ocr = ddddocr.DdddOcr(ocr=True, show_ad=False)
        det = ddddocr.DdddOcr(det=True, show_ad=False)
        
        logger.info("初始化 Selenium")
        driver = init_selenium(debug=debug, headless=headless)
        
        with open("stealth.min.js", mode="r") as f:
            js = f.read()
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": js
        })
        
        logger.info("发起登录请求")
        driver.get("https://app.rainyun.com/auth/login")
        wait = WebDriverWait(driver, timeout)
        
        max_retries = 3
        retry_count = 0
        login_success = False
        
        while retry_count < max_retries and not login_success:
            try:
                username = wait.until(EC.visibility_of_element_located((By.NAME, 'login-field')))
                password = wait.until(EC.visibility_of_element_located((By.NAME, 'login-password')))
                
                try:
                    login_button = wait.until(EC.element_to_be_clickable((By.XPATH,
                                                                        '//*[@id="app"]/div[1]/div[1]/div/div[2]/fade/div/div/span/form/button')))
                except:
                    try:
                        login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[type="submit"]')))
                    except:
                        login_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "登录")]')))
                
                username.clear()
                password.clear()
                
                username.send_keys(user)
                time.sleep(0.5)
                password.send_keys(pwd)
                time.sleep(0.5)
                
                driver.execute_script("arguments[0].click();", login_button)
                logger.info(f"登录尝试 {retry_count + 1}/{max_retries}")
                login_success = True
            except TimeoutException:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"登录失败，{retry_count}秒后重试...")
                    time.sleep(retry_count)
                    driver.refresh()
                else:
                    logger.error("页面加载超时，请尝试延长超时时间或切换到国内网络环境！")
                    raise Exception("登录页面加载超时或失败。")
        
        # 改进的验证码处理
        captcha_processed = False
        try:
            logger.info("[验证码] 等待验证码框架加载...")
            login_captcha = wait.until(
                EC.presence_of_element_located((By.ID, 'tcaptcha_iframe_dy')),
                timeout=20
            )
            logger.warning("[验证码] ✓ 找到验证码框架！")
            time.sleep(2)
            
            try:
                driver.switch_to.frame("tcaptcha_iframe_dy")
                logger.info("[验证码] ✓ 成功切换到验证码iframe")
                process_captcha()
                captcha_processed = True
                logger.info("[验证码] ✓ 验证码处理完成")
            except Exception as e:
                logger.error(f"[验证码] 切换iframe失败: {e}")
                driver.switch_to.default_content()
                
        except TimeoutException as e:
            logger.warning(f"[验证码] 未在规定时间内找到验证码框架: {e}")
        except Exception as e:
            logger.error(f"[验证码] 处理异常: {e}")
            try:
                driver.switch_to.default_content()
            except:
                pass
        
        time.sleep(5)
        try:
            driver.switch_to.default_content()
        except:
            pass
        
        # 验证登录状态并处理赚取积分
        if "dashboard" in driver.current_url:
            logger.info("✓ 登录成功！")
            logger.info("正在转到赚取积分页")
            
            for _ in range(3):
                try:
                    driver.get("https://app.rainyun.com/account/reward/earn")
                    logger.info("等待赚取积分页面加载...")
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                    time.sleep(3)
                    
                    earn = None
                    strategies = [
                        (By.XPATH, '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[2]/div/div/div/div[1]/div/div[1]/div/div[1]/div/span[2]/a'),
                        (By.XPATH, '//a[contains(@href, "earn") and contains(text(), "赚取")]'),
                        (By.CSS_SELECTOR, 'a[href*="earn"]'),
                        (By.XPATH, '//a[contains(@class, "earn")]')
                    ]
                    
                    for by, selector in strategies:
                        try:
                            earn = wait.until(EC.element_to_be_clickable((by, selector)))
                            logger.info(f"使用策略 {by}={selector} 找到赚取积分按钮")
                            break
                        except:
                            logger.debug(f"策略 {by}={selector} 未找到按钮，尝试下一种")
                            continue
                    
                    if earn:
                        driver.execute_script("arguments[0].scrollIntoView(true);", earn)
                        time.sleep(1)
                        logger.info("点击赚取积分")
                        driver.execute_script("arguments[0].click();", earn)
                        
                        try:
                            logger.info("检查是否需要验证码")
                            wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "tcaptcha_iframe_dy")))
                            logger.info("处理验证码")
                            process_captcha()
                            driver.switch_to.default_content()
                        except:
                            logger.info("未触发验证码或验证码框架加载失败")
                            driver.switch_to.default_content()
                        
                        logger.info("赚取积分操作完成")
                        break
                    else:
                        logger.warning("未找到赚取积分按钮，刷新页面重试...")
                        driver.refresh()
                        time.sleep(3)
                except Exception as e:
                    logger.error(f"访问赚取积分页面时出错: {e}")
                    time.sleep(3)
            else:
                logger.error("多次尝试后仍无法找到赚取积分按钮")
            
            driver.implicitly_wait(5)
            points_raw = driver.find_element(By.XPATH,
                                             '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[1]/div[1]/div/p/div/h3').get_attribute(
                "textContent")
            current_points = int(''.join(re.findall(r'\d+', points_raw)))
            logger.info(f"当前剩余积分: {current_points} | 约为 {current_points / 2000:.2f} 元")
            logger.info("任务执行成功！")
            
            return True, user, current_points, None
        else:
            logger.error("登录失败！")
            
            return False, user, 0, "登录失败，未能进入仪表盘页面，请检查账号密码或验证码处理逻辑。"

    except Exception as e:
        err_msg = f"脚本运行期间发生致命异常: {str(e)}"
        logger.error(err_msg, exc_info=True)
        
        return False, user, 0, err_msg

    finally:
        if driver:
            logger.info("正在关闭浏览器...")
            try:
                driver.quit()
            except:
                pass


if __name__ == "__main__":
    is_github_actions = os.environ.get("GITHUB_ACTIONS", "false") == "true"
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    headless = os.environ.get('HEADLESS', 'false').lower() == 'true'
    
    if is_github_actions:
        headless = True
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    ver = "2.3"
    logger.info("------------------------------------------------------------------")
    logger.info(f"雨云自动签到工作流 v{ver} by 皇帝二十 ~")
    logger.info("推广链接https://www.rainyun.com/bv_?s=rqd")
    logger.info("支持我https://rewards.qxzhan.cn/")
    logger.info("Github发布页: https://github.com/scfcn/Rainyun-Qiandao")
    logger.info("------------------------------------------------------------------")
    
    accounts = []
    
    users_env = os.environ.get("RAINYUN_USER", "")
    passwords_env = os.environ.get("RAINYUN_PASS", "")
    
    users = [user.strip() for user in users_env.split('\n') if user.strip()]
    passwords = [pwd.strip() for pwd in passwords_env.split('\n') if pwd.strip()]
    
    if len(users) == len(passwords):
        if len(users) > 0:
            logger.info(f"读取到 {len(users)} 个账户配置")
            for user,
