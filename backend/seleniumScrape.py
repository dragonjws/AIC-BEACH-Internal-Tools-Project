import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

options = Options()
options.add_argument("--headless")  # Runs without a window
options.add_argument("--no-sandbox") # Bypass OS security model (crucial for Linux)
options.add_argument("--disable-dev-shm-usage") # Overcomes limited resource problems
options.add_argument("--disable-gpu") # Recommended for headless mode

driver = webdriver.Chrome(options=options)

def scrapeLaws(url):
    driver.get(url)
    print(f"  --- Starting Link: {url} ---")

    while True:
        
        current_url = driver.current_url

        if "sectionNum=" not in driver.current_url:
            print("URL parameters lost. Forcing a reload...")
            # This physically reloads the page to try and get the server to re-send the data
            driver.refresh() 
            # Then we wait again for the specific law content to appear
            wait.until(EC.presence_of_element_located((By.ID, "single_law_section")))

        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.ID, "single_law_section")))
        sectionNum = driver.find_element(By.CSS_SELECTOR, "#codeLawSectionNoHead h6 b").text.strip()
        last_section = sectionNum
        

        # get the text we need
        # allHeaders = driver.find_elements(By.CSS_SELECTOR, "#codeLawSectionNoHead h4 b")
        # title = allHeaders[0].text
        # subtitle = allHeaders[1].text
        # sectionNum = driver.find_element(By.CSS_SELECTOR, "#codeLawSectionNoHead h6 b").text.strip()
        # text = driver.find_element(By.ID, "codeLawSectionNoHead").text 
        
        # f = open(f"{title}-{subtitle}-Section {sectionNum}.txt", "w")
        # f.write(f"{text}")
        # f.write(f"\n\nURL: {current_url}")
        # f.close()

        print("file read")

        try:
            next_button = driver.find_element(By.ID, "displayCodeSection:next")
            old_section_text = next_button.text
            
            

            # # Click the JS button (old)
            # next_button = wait.until(EC.element_to_be_clickable((By.ID, "displayCodeSection:next")))
            # next_button.click()
            
            clicked = False
            for attempt in range(3): # Try clicking up to 3 times
                try:
                    next_button = wait.until(EC.element_to_be_clickable((By.ID, "displayCodeSection:next")))
                    next_button.click()
                    
                    # Wait for the heading text to change
                    wait.until(lambda d: d.find_element(By.CSS_SELECTOR, "#codeLawSectionNoHead h6 b").text != last_section)
                    clicked = True
                    break 
                except Exception:
                    time.sleep(2) # Wait for loading overlays ???
            
            if not clicked:
                print("True ending")
                break

            # Wait for the URL to change (this "generates" the next link)
            wait.until(lambda d: d.find_element(By.CSS_SELECTOR, "#codeLawSectionNoHead h6 b").text != sectionNum)
            print(f"{sectionNum} != {driver.find_element(By.CSS_SELECTOR, '#codeLawSectionNoHead h6 b').text}")
            
            # Now driver.current_url IS the new link
            print(f"Successfully moved to: {driver.current_url}")
        
        except Exception as e:

            print("Reached the end of the code or button not found.")
            driver.save_screenshot(f"stop_point_debug{sectionNum}.png")
            break

def scrapeDivision(division_start_url):
    driver.get(division_start_url)
    wait = WebDriverWait(driver, 10)
    allLinks = []

    try:
        wait.until(EC.presence_of_element_located((By.ID, "expandedbranchcodesid")))
        chapterElements = driver.find_elements(By.CSS_SELECTOR, "#expandedbranchcodesid a")
        chapterLinks = [el.get_attribute("href") for el in chapterElements if el.get_attribute("href")]
        for i in range(1, len(chapterLinks)):
            try:
                driver.get(chapterLinks[i])
                selector = "#manylawsections h6 a, #showLawCodeSections h6 a, .manylawsections h6 a"
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, selector)))
                startElement = driver.find_element(By.CSS_SELECTOR, selector)
                startElement.click()
                wait.until(EC.url_changes(chapterLinks[i]))
                allLinks.append(driver.current_url)
                #print(chapterLinks[i])
            except TimeoutException:
                print(f"failed URL: {driver.current_url}")
                print("The page loaded, but is not the page we are looking for. (Has subsections)")
    except TimeoutException:
        try: 
            selector = "#manylawsections h6 a, #showLawCodeSections h6 a, .manylawsections h6 a"
            wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, selector)))
            startElement = driver.find_element(By.CSS_SELECTOR, selector)
            startElement.click()
            wait.until(EC.url_changes(division_start_url))
            allLinks.append(driver.current_url)
        except TimeoutException:
            print("here")
            print("The page loaded, but is not the page we are looking for.")

    print("-- Finished Scraping Division --")
    return allLinks

def scrapeCode(codeUrl):
    driver.get(codeUrl)
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "codes_toc_list")))
    division_start_urls = []

    codeElements = driver.find_elements(By.CSS_SELECTOR, ".codes_toc_list a")
    division_start_urls = [el.get_attribute("href") for el in codeElements if el.get_attribute("href")]



    return division_start_urls


codeUrl = "https://leginfo.legislature.ca.gov/faces/codesTOCSelected.xhtml?tocCode=BPC&tocTitle=+Business+and+Professions+Code+-+BPC"
# division_start_urls = scrapeCode(codeUrl)
#for division_start_url in division_start_urls:
    #startLinks = scrapeDivision(division_start_url)
    # for link in startLinks:
    #     scrapeLaws(link)


# testing
division_start_urls = ["https://leginfo.legislature.ca.gov/faces/codes_displayexpandedbranch.xhtml?tocCode=BPC&division=1.&title=&part=&chapter=&article="]
print(division_start_urls[0])
startLinks = scrapeDivision(division_start_urls[0])
print(startLinks)
for link in startLinks:
    scrapeLaws(link)




driver.quit()