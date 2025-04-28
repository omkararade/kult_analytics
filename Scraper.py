from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import traceback
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import concurrent.futures
import os
import csv

def highlight_element(driver, element, color="red"):
    driver.execute_script(
        f"arguments[0].style.border='3px solid {color}'",
        element
    )

def setup_driver(headless=False):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def scroll_reviews_panel(driver, panel, max_attempts=20000, patience=10):
    last_height = driver.execute_script("return arguments[0].scrollHeight", panel)
    scroll_count = 0
    stagnant_scrolls = 0

    for attempt in range(max_attempts):
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", panel)
        time.sleep(1.5)
        new_height = driver.execute_script("return arguments[0].scrollHeight", panel)

        if new_height == last_height:
            stagnant_scrolls += 1
            if stagnant_scrolls >= patience:
                print(f"\n✅ Stopped scrolling early after {scroll_count} scrolls (no new content).")
                break
        else:
            stagnant_scrolls = 0

        last_height = new_height
        scroll_count += 1
        print(f"\rScrolled {scroll_count} times", end="", flush=True)

    print("\n✔ Finished scrolling. Extracting reviews now...")

def wait_for_element_to_be_clickable(driver, locator, timeout=30):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable(locator)
        )
        return element if element.is_displayed() and element.is_enabled() else None
    except:
        return None

def scrape_reviews(app_url, headless=True):
    driver = setup_driver(headless=headless)
    try:
        app_id = parse_qs(urlparse(app_url).query).get('id', ['unknown'])[0]
        app_name = app_id.split('.')[-1].capitalize() if app_id != 'unknown' else 'App'

        print(f"Opening URL: {app_url}")
        driver.get(app_url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))

        see_all_button = wait_for_element_to_be_clickable(driver, (By.XPATH, "//button[.//span[contains(text(), 'See all reviews')]]"))
        if see_all_button:
            driver.execute_script("arguments[0].click();", see_all_button)
            time.sleep(5)

        sort_button = wait_for_element_to_be_clickable(driver, (By.CSS_SELECTOR, "div[aria-label*='Sort reviews']"))
        if sort_button:
            driver.execute_script("arguments[0].click();", sort_button)
            time.sleep(2)
            

        reviews_panel = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.fysCi")))
        scroll_reviews_panel(driver, reviews_panel)

        reviews = driver.find_elements(By.CSS_SELECTOR, "div.RHo1pe")
        print(f"Found {len(reviews)} reviews")

        filename = f"{app_name} data_live.csv"
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Username", "Rating", "Date", "Review", "Reply", "Reply_Date", "Review_Helpful"])
            for review in reviews:
                try:
                    username = review.find_element(By.CSS_SELECTOR, "div.X5PpBb").text.strip()
                    rating = review.find_element(By.CSS_SELECTOR, "div.iXRFPc").get_attribute("aria-label").split()[1]
                    review_text = review.find_element(By.CSS_SELECTOR, "div.h3YV2d").text.strip() if review.find_elements(By.CSS_SELECTOR, "div.h3YV2d") else "N/A"
                    review_date = review.find_element(By.CSS_SELECTOR, "span.bp9Aid").text.strip() if review.find_elements(By.CSS_SELECTOR, "span.bp9Aid") else "N/A"
                    reply = review.find_element(By.CSS_SELECTOR, "div.I6j64d").text.strip() if review.find_elements(By.CSS_SELECTOR, "div.I6j64d") else "N/A"
                    reply_date = review.find_element(By.CSS_SELECTOR, "div.I9Jtec").text.strip() if review.find_elements(By.CSS_SELECTOR, "div.I9Jtec") else "N/A"
                    helpful = review.find_element(By.CSS_SELECTOR, "div.AJTPZc").text.strip() if review.find_elements(By.CSS_SELECTOR, "div.AJTPZc") else "N/A"
                    writer.writerow([username, rating, review_date, review_text, reply, reply_date, helpful])
                except Exception as e:
                    print(f"Review parse error: {e}")

        print(f"✔ Live CSV saved as {filename}")

    except Exception as e:
        print(f"\n✖ Fatal error: {str(e)}")
        traceback.print_exc()
    finally:
        driver.quit()

def run_parallel_scrapers(app_urls, max_workers=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(scrape_reviews, app_urls)

if __name__ == "__main__":
    app_urls = [
        "https://play.google.com/store/apps/details?id=mate.bluetoothprint&referrer=utm_source%3Dwebsite%26utm_medium%3Dhero",
    ]
    run_parallel_scrapers(app_urls)