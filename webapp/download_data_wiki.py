import os

import requests
from bs4 import BeautifulSoup

INDIAN_PLAYERS = [
    "https://en.wikipedia.org/wiki/Suryakumar_Yadav",
    "https://en.wikipedia.org/wiki/Yashasvi_Jaiswal",
    "https://en.wikipedia.org/wiki/Virat_Kohli",
    "https://en.wikipedia.org/wiki/Rohit_Sharma",
    "https://en.wikipedia.org/wiki/Hardik_Pandya",
    "https://en.wikipedia.org/wiki/Ravindra_Jadeja",
    "https://en.wikipedia.org/wiki/Axar_Patel",
    "https://en.wikipedia.org/wiki/Kuldeep_Yadav",
    "https://en.wikipedia.org/wiki/Jasprit_Bumrah",
]


NATIONAL_CRICKET_TEAMS = [
    "https://en.wikipedia.org/wiki/Australia_national_cricket_team",
    "https://en.wikipedia.org/wiki/England_cricket_team",
    "https://en.wikipedia.org/wiki/India_national_cricket_team",
    "https://en.wikipedia.org/wiki/New_Zealand_national_cricket_team",
    "https://en.wikipedia.org/wiki/Pakistan_national_cricket_team",
    "https://en.wikipedia.org/wiki/South_Africa_national_cricket_team",
    "https://en.wikipedia.org/wiki/Sri_Lanka_national_cricket_team",
    "https://en.wikipedia.org/wiki/West_Indies_cricket_team",
]


def scrape_wiki(url):
    # Make an API call to get the page content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    name = soup.find("h1", class_="firstHeading").text

    # Extract all the text from para HTML tags
    para_tags = soup.find_all("p")
    full_page_text = ""
    for para in para_tags:
        full_page_text += para.text

    return name, full_page_text


if __name__ == "__main__":
    # Store player bio taken from wikipedia in text files
    if not os.path.exists("/content/player_wiki"):
        os.mkdir("../data/player_wiki")

    player_data = []
    for player in INDIAN_PLAYERS:
        name, text = scrape_wiki(player)
        with open(f"../data/player_wiki/{name}.txt", "x") as f:
            f.write(text)

    # Store team bio taken from wikipedia in text files
    if not os.path.exists("/content/team_wiki"):
        os.mkdir("../data/team_wiki")

    for team in NATIONAL_CRICKET_TEAMS:
        name, text = scrape_wiki(team)
        with open(f"../data/team_wiki/{name}.txt", "x") as f:
            f.write(text)
