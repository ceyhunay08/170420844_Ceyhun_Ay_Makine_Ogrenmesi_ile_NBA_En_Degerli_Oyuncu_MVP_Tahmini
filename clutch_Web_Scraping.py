from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

driver_path = "./chromedriver.exe"

service = Service(driver_path)

seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(1996, 2023)]

all_data = []

desired_columns = ['PLAYER', 'GP', 'PTS', 'FGA', 'FTA', 'BLK', 'STL', 'OREB', 'AST', 'TOV']

for season in seasons:

    driver = webdriver.Chrome(service=service)

    url = f"https://www.nba.com/stats/players/clutch-traditional?Season={season}&SeasonType=Playoffs"
    driver.get(url)

    try:
        cookie_banner = driver.find_element(By.ID, "onetrust-accept-btn-handler")
        cookie_banner.click()
    except:
        pass

    wait = WebDriverWait(driver, 15)
    table = wait.until(
        EC.presence_of_element_located(
            (By.XPATH, '//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[3]'))
    )

    rows = table.find_elements(By.TAG_NAME, "tr")
    headers = [header.text for header in rows[0].find_elements(By.TAG_NAME, "th")]

    indices = [headers.index(col) for col in desired_columns]

    while True:
        rows = table.find_elements(By.TAG_NAME, "tr")

        for row in rows[1:]:
            cols = row.find_elements(By.TAG_NAME, "td")
            cols = [col.text for col in cols]

            selected_cols = [cols[i] for i in indices if i < len(cols)]

            all_data.append([season[:4]] + selected_cols)

        try:
            next_button = wait.until(EC.element_to_be_clickable((By.XPATH,
                                                                 '//*[@id="__next"]/div[2]/div[2]/div[3]/section[2]/div/div[2]/div[2]/div[1]/div[5]/button[2]')))
            if "disabled" in next_button.get_attribute("class"):
                print(f"Son sayfaya ulaşıldı, sezon: {season}.")
                break
            else:
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(5)
        except Exception as e:
            print(f"Son sayfa kontrolünde hata oluştu: {e}")
            break

    driver.quit()

columns = ['Season'] + desired_columns  # Sezon sütununu başa ekledik
df = pd.DataFrame(all_data, columns=columns)

df.to_csv("C:/Users/Lenovo/Documents/NBA_Clutch_Stats_With_Gp_1996_2023.csv", index=False)

print("Tüm sezonlar için veri seti kaydedildi.")
