from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# Set up the ChromeDriver service
service = Service(ChromeDriverManager().install())

# Initialize the Chrome WebDriver using the service
browser = webdriver.Chrome(service=service)

# Visit the webpage and take a screenshot
browser.get("https://www.drivebc.ca/mobile/pub/webcams/id/292.html")
browser.save_screenshot('screenshot.png')
browser.quit()