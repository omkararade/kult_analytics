from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode if UI is not needed
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--window-size=1920,1080")

# Provide the correct path to chromedriver.exe
chrome_driver_path = "D:/omnidatax/kult app/chromedriver.exe"

# Initialize WebDriver with correct path and options
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL of the app reviews page
app_url = "https://play.google.com/store/apps/details?id=beauty.kult.app&hl=en_IN"

def scroll_reviews_panel(panel, max_attempts=15):
    scroll_attempts = 0
    last_height = driver.execute_script("return arguments[0].scrollHeight", panel)
    while scroll_attempts < max_attempts:
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", panel)
        time.sleep(2)  # Allow time for new reviews to load
        new_height = driver.execute_script("return arguments[0].scrollHeight", panel)
        if new_height == last_height:
            break
        last_height = new_height
        scroll_attempts += 1

try:
    print("Starting review scraping...")
    driver.get(app_url)
    time.sleep(5)  # Allow time for the page to load

    # Click 'See All Reviews' button if available
    try:
        see_all_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[text()='See all reviews']/ancestor::button"))
        )
        driver.execute_script("arguments[0].click();", see_all_button)
        print("Clicked on 'See All Reviews'")
        time.sleep(5)
    except Exception as e:
        print(f"Could not find 'See All Reviews' button, continuing with visible reviews. Error: {e}")

    # Wait for reviews panel to appear
    reviews_panel = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div[jscontroller='H6eOGe']"))
    )
    print("Review panel opened")

    # Scroll inside the reviews pop-up to load more reviews
    scroll_reviews_panel(reviews_panel)

    # Extract reviews
    reviews = driver.find_elements(By.CSS_SELECTOR, "div.RHo1pe")
    review_data = []
    
    for review in reviews:
        try:
            username = review.find_element(By.CSS_SELECTOR, "span.X5PpBb").text.strip()
            rating_element = review.find_element(By.CSS_SELECTOR, "div.iXRFPc span")
            rating = rating_element.get_attribute("aria-label").split()[1] if rating_element else "N/A"
            review_text = review.find_element(By.CSS_SELECTOR, "span[jsname='bN97Pc']").text.strip()
            review_data.append([username, rating, review_text])
        except Exception as e:
            print(f"Skipping a review due to error: {e}")

    print(f"Found {len(review_data)} reviews")

    # Save to CSV
    if review_data:
        df = pd.DataFrame(review_data, columns=["Username", "Rating", "Review"])
        df.to_csv("google_play_reviews.csv", index=False, encoding='utf-8')
        print("Data saved to google_play_reviews.csv")
    else:
        print("No reviews found, skipping file save.")

except Exception as e:
    print(f"Error during scraping: {e}")

finally:
    driver.quit()
