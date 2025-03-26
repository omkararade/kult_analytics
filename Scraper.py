from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import traceback

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def scroll_reviews_panel(driver, panel, max_attempts=50):
    last_height = driver.execute_script("return arguments[0].scrollHeight", panel)
    
    for attempt in range(max_attempts):
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", panel)
        time.sleep(2)
        new_height = driver.execute_script("return arguments[0].scrollHeight", panel)
        
        if new_height == last_height:
            print("No more reviews to load.")
            break
        
        last_height = new_height
        print(f"Scrolled {attempt + 1}/{max_attempts}")

def scrape_reviews():
    driver = setup_driver()
    app_url = "https://play.google.com/store/apps/details?id=beauty.kult.app&hl=en_IN"
    
    try:
        print("Starting review scraping...")
        driver.get(app_url)
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))

        # Click 'See All Reviews' button
        try:
            see_all_button = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable((By.XPATH, "//button[.//span[contains(text(), 'See all reviews')]]"))
            )
            driver.execute_script("arguments[0].click();", see_all_button)
            print("Clicked 'See All Reviews'")
            time.sleep(3)
        except Exception as e:
            print(f"Could not find 'See All Reviews' button: {str(e)}")
            return

        # Wait for reviews panel
        try:
            reviews_panel = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.fysCi"))
            )
            print("Review panel opened")
        except Exception as e:
            print(f"Could not find reviews panel: {str(e)}")
            return

        # Scroll inside the reviews pop-up
        scroll_reviews_panel(driver, reviews_panel)

        # Extract reviews
        reviews = driver.find_elements(By.CSS_SELECTOR, "div.RHo1pe")
        review_data = []

        for review in reviews:
            try:
                username = review.find_element(By.CSS_SELECTOR, "div.X5PpBb").text.strip()
                rating = review.find_element(By.CSS_SELECTOR, "div.iXRFPc").get_attribute("aria-label").split()[1]
                
                try:
                    review_text = review.find_element(By.CSS_SELECTOR, "div.h3YV2d").text.strip()
                except:
                    review_text = "N/A"
                
                try:
                    review_date = review.find_element(By.CSS_SELECTOR, "span.bp9Aid").text.strip()
                except:
                    review_date = "N/A"
                
                try:
                    usefulness = review.find_element(By.CSS_SELECTOR, "div.AJTPZc").text.strip()
                except:
                    usefulness = "N/A"

                try:
                    reply = review.find_element(By.CSS_SELECTOR, "div.I6j64d").text.strip()
                except:
                    reply = "N/A"

                try:
                    reply_date = review.find_element(By.CSS_SELECTOR, "div.I9Jtec").text.strip()
                except:
                    reply_date = "N/A"
                
                review_data.append([username, rating, review_date, review_text, usefulness, reply, reply_date])
            except Exception as e:
                print(f"Skipping a review due to error: {str(e)}")

        print(f"Found {len(review_data)} reviews")

        # Save to CSV
        if review_data:
            df = pd.DataFrame(review_data, columns=["Username", "Rating", "Date", "Review", "Usefulness", "Reply", "Reply Date"])
            df.to_csv("google_play_reviews.csv", index=False, encoding='utf-8')
            print("Data saved to google_play_reviews.csv")
        else:
            print("No reviews found, skipping file save.")

    except Exception as e:
        print(f"Error during scraping: {str(e)}")
        traceback.print_exc()
    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_reviews()
