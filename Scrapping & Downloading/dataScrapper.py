from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
import time

# Lists for storing downlaod links and descriptions of cases
links = []
data = []


def fetchData():
    _data = []
    _links = []
    # Access the table and get td's of tr's with the class 'odd' or 'even'
    table = driver.find_element(by=By.ID, value='resultsTable')
    for tr in table.find_elements(by=By.CSS_SELECTOR, value='tr.odd, tr.even'):
        td = tr.find_elements(by=By.TAG_NAME, value='td')
        if ("Tagline :" not in td[0].text.strip()):
            anchor = tr.find_element(by=By.TAG_NAME, value="a")
            href = anchor.get_attribute('href')
            _links.append(href)

            num = td[0].text.strip()
            desc = td[1].text.strip() + ", "+td[3].text.strip() + ", \t" + \
                td[4].text.strip() + ", \tDated: "+td[6].text.strip()
            line = num.ljust(8) + desc
            _data.append(line)
    print(len(_links), " links fetched!")
    return _data, _links


url = "https://www.supremecourt.gov.pk/judgement-search/#1573035933449-63bb4a39-ac81"
driver = webdriver.Chrome()
driver.get(url)

# Find the 'select' tag with id = 'case_type'
select_tag = Select(driver.find_element(by=By.ID, value='case_type'))

# Select one by one every option from the select tag
tagList = ["C.A.", "C.M.A.", "C.P.", "Const.P.", "Crl.A.", "Crl.P.", "S.M.C."]
for option_tag in select_tag.options:
    option_value = option_tag.get_attribute('value')
    if (option_value not in tagList):
        print("Skipping ", option_value, " due to less data...")
        continue

    select_tag.select_by_value(option_value)
    button = driver.find_element(by=By.XPATH, value="//input[@type='button' and @value='Search Result']")
    button.click()
    time.sleep(5)

    if (option_value in ["C.A.", "C.P."]):
        num = 10
    else:
        num = 3

    # Lists for fetching data from fetchData()
    dt, d, li, l = [], [], [], []
    for k in range(num):
        time.sleep(1)
        d, l = fetchData()
        dt += d
        li += l

        nextButton = driver.find_element(by=By.XPATH, value="//a[text()='Next']")
        nextButton.click()
        time.sleep(3)

    data.append(dt)
    links.append(li)
    print(len(dt), " records fetched!")


print("\nNow writing...\n")
with open('links.txt', 'w') as file:
    for i in range(len(links)):
        file.write("Category " + tagList[i] + "\n")
        for link in links[i]:
            file.write(link + '\n')

with open('descripton.txt', 'w') as wr:
    wr.write("Category".ljust(10) + "S.No.".ljust(8) + "Abstract of Judgements\n\n")
    for i in range(len(data)):
        for dt in data[i]:
            wr.write(tagList[i].ljust(10) + dt + "\n")
        wr.write('\n\n')

with open('cases_count.txt', 'w') as wr:
    wr.write("Category".ljust(10) + "Number of Judgements".ljust(20) + "\n")
    for i in range(len(data)):
        wr.write(tagList[i].ljust(10) + str(len(data[i])) + "\n")


print("Records written to file \"description.txt\"!")

# Close the web driver
driver.quit()
