# plan is to scrape two days data into my computer (while it's running)
# then from this csv file, with the date, time, and number of cars
# i will run a predictive ai 
# need to know : 
#   what parameters does the predictive AI need to run?

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from PIL import Image
from io import BytesIO

# Set up the ChromeDriver service
service = Service(ChromeDriverManager().install())

# Initialize the Chrome WebDriver using the service
driver = webdriver.Chrome(service=service)

# Visit the webpage
driver.get("https://www.drivebc.ca/mobile/pub/webcams/id/684.html")

# Save screenshot of the entire page as a PNG
png = driver.get_screenshot_as_png()

# Open the image in memory with PIL library
im = Image.open(BytesIO(png))

# Define crop points 
left = 20
top = 300
width = 650
height = 650
im = im.crop((left, top, (left + width) , (top + height)))

# Save the cropped image
im.save('cropped_screenshot.png')

# Close the browser
driver.quit()