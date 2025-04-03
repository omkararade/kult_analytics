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

def highlight_element(driver, element, color="red"):
    """Highlight an element with a colored border"""
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
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")  # Mimic real browser
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def scroll_reviews_panel(driver, panel, max_attempts=60):
    last_height = driver.execute_script("return arguments[0].scrollHeight", panel)
    scroll_count = 0
    
    for attempt in range(max_attempts):
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", panel)
        time.sleep(2.5)
        new_height = driver.execute_script("return arguments[0].scrollHeight", panel)
        
        if new_height == last_height:
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", panel)
            time.sleep(2)
            new_height = driver.execute_script("return arguments[0].scrollHeight", panel)
            if new_height == last_height:
                print(f"\nNo more reviews to load after {scroll_count} scrolls.")
                break
        
        last_height = new_height
        scroll_count += 1
        print(f"\rScrolled {scroll_count} times", end="", flush=True)

def take_screenshot(driver, name="screenshot"):
    """Take screenshot and save with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.png"
    driver.save_screenshot(filename)
    print(f"Screenshot saved as {filename}")
    return filename

def wait_for_javascript(driver, timeout=30):
    """Wait for JavaScript to finish loading"""
    try:
        WebDriverWait(driver, timeout).until(
            lambda driver: driver.execute_script('return document.readyState') == 'complete'
        )
    except Exception as e:
        print(f"⚠ JavaScript loading check failed: {str(e)}")

def wait_for_no_network_activity(driver, timeout=60):
    """Wait until there is no network activity"""
    end_time = time.time() + timeout
    while time.time() < end_time:
        network_idle = driver.execute_script("return window.performance && window.performance.getEntriesByType('resource').length === 0")
        if network_idle:
            return True
        time.sleep(2)
    print("⚠ Network activity still present after timeout")
    return False

def wait_for_reviews_to_load(driver, timeout=120, max_retries=5):
    """Wait for reviews to load after sorting change with retry logic"""
    print("Waiting for reviews to reload...")
    for attempt in range(max_retries):
        try:
            WebDriverWait(driver, timeout // max_retries).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='progressbar']"))
            )
            WebDriverWait(driver, timeout // max_retries).until_not(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='progressbar']"))
            )
            WebDriverWait(driver, timeout // max_retries).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.RHo1pe"))
            )
            print("✔ Reviews reloaded successfully")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(10)
            else:
                print("⚠ All retries failed, falling back to simple wait")
                time.sleep(30)
                return False
    return False

def wait_for_element_to_be_clickable(driver, locator, timeout=30):
    """Wait for an element to be clickable with additional checks"""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable(locator)
        )
        if element.is_displayed() and element.is_enabled():
            return element
        else:
            raise Exception("Element is not visible or enabled")
    except Exception as e:
        print(f"⚠ Failed to wait for element to be clickable: {str(e)}")
        return None

def scrape_reviews():
    driver = setup_driver(headless=False)
    
    try:
        app_url = "https://play.google.com/store/apps/details?id=com.prune&hl=en_IN"
        print(f"\nStarting review scraping at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Opening URL: {app_url}")
        
        driver.get(app_url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))
        print("✔ Page loaded successfully")

        # Step 1: Click 'See All Reviews'
        try:
            print("\nSTEP 1: Clicking 'See All Reviews' button")
            see_all_xpaths = [
                "//button[.//span[contains(text(), 'See all reviews')]]",
                "//button[contains(., 'See all reviews')]",
                "//button[contains(., 'See all')]"
            ]
            
            for xpath in see_all_xpaths:
                try:
                    see_all_button = wait_for_element_to_be_clickable(driver, (By.XPATH, xpath))
                    if see_all_button:
                        highlight_element(driver, see_all_button, "green")
                        driver.execute_script("arguments[0].scrollIntoView(true);", see_all_button)
                        time.sleep(1)
                        driver.execute_script("arguments[0].click();", see_all_button)
                        print(f"✔ Clicked 'See All Reviews' using XPath: {xpath}")
                        wait_for_javascript(driver)
                        break
                except:
                    continue
            else:
                raise Exception("Could not find 'See All Reviews' button with any XPath")
            
            time.sleep(5)
        except Exception as e:
            print(f"✖ Error in Step 1: {str(e)}")
            take_screenshot(driver, "see_all_error")
            return

        # Step 2: Open sorting dropdown
        try:
            print("\nSTEP 2: Opening sorting dropdown")
            dropdown_selectors = [
                ("XPATH", "//div[@role='button' and @aria-label='Most relevant']"),
                ("XPATH", "//div[contains(text(), 'Most relevant')]"),
                ("CSS", "div[aria-label*='Sort reviews']"),
                ("CSS", "div.D3Qfie")
            ]
            
            for selector_type, selector in dropdown_selectors:
                try:
                    locator = (By.XPATH, selector) if selector_type == "XPATH" else (By.CSS_SELECTOR, selector)
                    dropdown = wait_for_element_to_be_clickable(driver, locator)
                    if dropdown:
                        highlight_element(driver, dropdown, "blue")
                        driver.execute_script("arguments[0].scrollIntoView(true);", dropdown)
                        time.sleep(1)
                        driver.execute_script("arguments[0].click();", dropdown)
                        print(f"✔ Opened dropdown using {selector_type}: {selector}")
                        wait_for_javascript(driver)
                        break
                except:
                    continue
            else:
                raise Exception("Could not find sorting dropdown with any selector")
            
            time.sleep(5)  # Increased wait after opening dropdown
        except Exception as e:
            print(f"✖ Error in Step 2: {str(e)}")
            take_screenshot(driver, "dropdown_error")
            print("⚠ Continuing with default sorting...")

        # Step 3: Select 'Newest' option with aggressive handling
        try:
            print("\nSTEP 3: Selecting 'Newest' option")
            newest_selectors = [
                ("XPATH", "//span[@role='menuitemradio' and @aria-label='Newest']"),  # Unselected state
                ("XPATH", "//div[@class='jO7h3c' and contains(text(), 'Newest')]"),  # Previously working
                ("XPATH", "//div[@role='button' and @aria-label='Newest']"),  # Selected state
                ("XPATH", "//div[@id='sortBy_2']"),  # Using ID
                ("CSS", "span[aria-label='Newest']"),  # CSS for unselected
                ("CSS", "div[aria-label='Newest']"),  # CSS for selected
            ]
            
            newest_option = None
            for selector_type, selector in newest_selectors:
                try:
                    locator = (By.XPATH, selector) if selector_type == "XPATH" else (By.CSS_SELECTOR, selector)
                    newest_option = wait_for_element_to_be_clickable(driver, locator, timeout=40)
                    if newest_option is None:
                        raise Exception("Element not clickable after waiting")

                    # Debug: Print element attributes
                    attrs = driver.execute_script("var items = {}; for (index = 0; index < arguments[0].attributes.length; ++index) { items[arguments[0].attributes[index].name] = arguments[0].attributes[index].value }; return items;", newest_option)
                    print(f"Element attributes: {attrs}")

                    # Ensure visibility and interaction readiness
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", newest_option)
                    driver.execute_script("arguments[0].style.zIndex = '9999'; arguments[0].style.position = 'relative';", newest_option)
                    highlight_element(driver, newest_option, "purple")
                    time.sleep(3)

                    # Multiple click attempts with verification
                    clicked = False
                    for attempt in range(5):  # Increased attempts
                        try:
                            driver.execute_script("arguments[0].click();", newest_option)
                            time.sleep(2)
                            # Verify if selected
                            if driver.find_elements(By.XPATH, "//div[@aria-label='Newest' and @aria-pressed='true']") or \
                               driver.find_elements(By.XPATH, "//span[@aria-label='Newest' and @aria-checked='true']"):
                                clicked = True
                                break
                        except Exception as e:
                            print(f"JavaScript click attempt {attempt + 1}/5 failed: {str(e)}")

                    if not clicked:
                        actions = ActionChains(driver)
                        actions.move_to_element(newest_option).pause(2).click().perform()
                        time.sleep(2)
                        if driver.find_elements(By.XPATH, "//div[@aria-label='Newest' and @aria-pressed='true']") or \
                           driver.find_elements(By.XPATH, "//span[@aria-label='Newest' and @aria-checked='true']"):
                            clicked = True

                    # if not clicked:
                    #     actions.send_keys(Keys.TAB).send_keys(Keys.ENTER).perform()
                    #     time.sleep(2)
                    #     if driver.find_elements(By.XPATH, "//div[@aria-label='Newest' and @aria-pressed='true']") or \
                    #        driver.find_elements(By.XPATH, "//span[@aria-label='Newest' and @aria-checked='true']"):
                    #         clicked = True

                    # if clicked:
                    #     print(f"✔ Selected 'Newest' using {selector_type}: {selector}")
                    #     wait_for_javascript(driver)
                    #     if not wait_for_no_network_activity(driver):
                    #         print("⚠ Network activity still present, waiting longer...")
                    #         time.sleep(15)

                    #     # Wait for reviews to load with extra verification
                    #     if not wait_for_reviews_to_load(driver):
                    #         print("⚠ Reviews failed to load, attempting to re-click 'Newest'")
                    #         driver.execute_script("arguments[0].click();", newest_option)
                    #         time.sleep(5)
                    #         wait_for_reviews_to_load(driver)

                    #     # Final confirmation
                        WebDriverWait(driver, 40).until(
                            lambda driver: driver.find_elements(By.XPATH, "//div[@aria-label='Newest' and @aria-pressed='true']") or 
                                          driver.find_elements(By.XPATH, "//span[@aria-label='Newest' and @aria-checked='true']")
                        )
                        print("✔ Confirmed 'Newest' is selected")
                        break
                    else:
                        raise Exception("All click methods failed to select 'Newest'")

                except Exception as e:
                    print(f"Failed to interact with 'Newest' using {selector_type} '{selector}': {str(e)}")
                    continue
            else:
                raise Exception("Could not find or click 'Newest' option with any selector")

        except Exception as e:
            print(f"✖ Error in Step 3: {str(e)}")
            take_screenshot(driver, "newest_error")
            print("⚠ Continuing with default sorting...")

        # Step 4: Scrape reviews
        try:
            print("\nSTEP 4: Scraping reviews")
            reviews_panel = WebDriverWait(driver, 40).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.fysCi"))
            )
            highlight_element(driver, reviews_panel, "orange")
            print("✔ Found reviews panel")
            
            WebDriverWait(driver, 40).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.RHo1pe")))
            print("✔ Reviews are present in the panel")
            
            print("Scrolling to load all reviews...")
            scroll_reviews_panel(driver, reviews_panel)
            
            reviews = driver.find_elements(By.CSS_SELECTOR, "div.RHo1pe")
            review_data = []
            
            print(f"\nFound {len(reviews)} reviews. Extracting data...")
            
            for i, review in enumerate(reviews, 1):
                try:
                    if i % 10 == 0:
                        highlight_element(driver, review, "yellow")
                        time.sleep(0.3)
                    
                    username = review.find_element(By.CSS_SELECTOR, "div.X5PpBb").text.strip()
                    rating = review.find_element(By.CSS_SELECTOR, "div.iXRFPc").get_attribute("aria-label").split()[1]
                    
                    review_text = review.find_element(By.CSS_SELECTOR, "div.h3YV2d").text.strip() if review.find_elements(By.CSS_SELECTOR, "div.h3YV2d") else "N/A"
                    review_date = review.find_element(By.CSS_SELECTOR, "span.bp9Aid").text.strip() if review.find_elements(By.CSS_SELECTOR, "span.bp9Aid") else "N/A"
                    
                    try:
                        reply = review.find_element(By.CSS_SELECTOR, "div.I6j64d").text.strip()
                        reply_date = review.find_element(By.CSS_SELECTOR, "div.I9Jtec").text.strip()
                    except:
                        reply, reply_date = "N/A", "N/A"

                    try:
                        Review_Helpful = review.find_element(By.CSS_SELECTOR, "div.AJTPZc").text.strip()
                    except:
                        Review_Helpful = "N/A", "N/A"
                    
                    review_data.append([username, rating, review_date, review_text, reply, reply_date, Review_Helpful,])
                    
                    print(f"\rProcessed {i}/{len(reviews)} reviews", end="", flush=True)
                    
                except Exception as e:
                    print(f"\n⚠ Error processing review {i}: {str(e)}")
                    continue
            
            print(f"\n✔ Successfully extracted {len(review_data)} reviews")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"google_play_reviews_{timestamp}.csv"
            
            if review_data:
                df = pd.DataFrame(review_data, columns=[
                    "Username", "Rating", "Date", "Review", "Reply", "Reply_Date", "Review_Helpful"
                ])
                df.to_csv(filename, index=False, encoding='utf-8')
                print(f"✔ Data saved to {filename}")
            else:
                print("✖ No reviews found, skipping file save")
                
        except Exception as e:
            print(f"✖ Error in Step 4: {str(e)}")
            take_screenshot(driver, "scraping_error")
    
    except Exception as e:
        print(f"\n✖ Fatal error: {str(e)}")
        traceback.print_exc()
        take_screenshot(driver, "fatal_error")
    
    finally:
        print("\nScraping completed")
        input("Press Enter to close the browser...")
        driver.quit()

if __name__ == "__main__":
    scrape_reviews()