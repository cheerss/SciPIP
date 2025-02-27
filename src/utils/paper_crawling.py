import os
import re
import random
import requests
from requests.exceptions import RequestException
import re
from bs4 import BeautifulSoup
from .hash import generate_hash_id
from .header import get_dir
from loguru import logger

with open(get_dir("./assets/data/user_agents.txt"), "r", encoding="utf8") as f:
    user_agents = [l.rstrip() for l in f.readlines()]


def extract_title_from_index(index_url):
    try:
        headers = {"User-Agent": random.choice(user_agents)}
        response_title = requests.get(index_url, headers=headers)
        response_title.raise_for_status()
        soup = BeautifulSoup(response_title.content, "html.parser")
        papers = soup.find_all("h2", id="title")
        for paper in papers:
            title = paper.text.strip()
            return title
    except RequestException as e:
        logger.error(f"Failed to extract title from {index_url}: {e}")
        return None


def extract_year_from_index(index_url):
    try:
        headers = {"User-Agent": random.choice(user_agents)}
        response_year = requests.get(index_url, headers=headers)
        response_year.raise_for_status()
        soup = BeautifulSoup(response_year.content, "html.parser")

        year_tag = soup.find("dt", text="Year:")
        if year_tag:
            year_dd = year_tag.find_next_sibling("dd")
            if year_dd:
                year = year_dd.text.strip()
                return year
        else:
            print(f"Year not found in {index_url}")
            return None
    except requests.RequestException as e:
        print(f"Failed to extract year from {index_url}: {e}")
        return None


def extract_pdf_url_from_index(index_url, id):
    try:
        headers = {"User-Agent": random.choice(user_agents)}
        response = requests.get(index_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        pdf_link = soup.find("a", href=True, string=re.compile(r"\bPDF\b", re.I))
        if pdf_link:
            pdf_url = pdf_link["href"]
            return pdf_url
        else:
            logger.warning(f"No PDF link found on {index_url}")
            return None
    except RequestException as e:
        logger.error(f"Failed to extract PDF URL from {index_url}: {e}")
        return None


class PaperCrawling:
    def __init__(self, config, data_type="train") -> None:
        self.base_url = "https://aclanthology.org/"
        self.data_type = data_type
        self.paper_pdf_folder = config.DEFAULT.pdf_cached
        if not os.path.exists(self.paper_pdf_folder):
            os.makedirs(self.paper_pdf_folder)
            logger.info(f"Created directory '{self.paper_pdf_folder}'")

    def need_to_parse(self, paper: dict):
        if (
            paper["abstract"] is None
            or paper["introduction"] is None
            or paper["reference"] is None
        ):
            return True
        return False

    def get_title(self, paper):
        index_url = f"{self.base_url}{paper['id']}/"
        title = extract_title_from_index(index_url)
        return title

    def get_year(self, paper):
        index_url = f"{self.base_url}{paper['id']}/"
        year = extract_year_from_index(index_url)
        return year

    def get_pdf_url(self, paper):
        if "pdf_url" not in paper.keys() or paper["pdf_url"] is None:
            index_url = f"{self.base_url}{paper['id']}/"
            paper["pdf_url"] = extract_pdf_url_from_index(index_url, paper["id"])

    def download_paper(self, paper):
        headers = {"User-Agent": random.choice(user_agents)}
        pdf_folder = os.path.join(
            self.paper_pdf_folder, f"{paper['venue_name']}", f"{paper['year']}"
        )
        file_path = os.path.join(pdf_folder, f"{paper['hash_id']}.pdf")
        paper["pdf_path"] = file_path
        paper_url = paper["pdf_url"]
        if not os.path.exists(pdf_folder):
            os.makedirs(pdf_folder)
        if os.path.exists(file_path):
            # print("pdf file {} exist ...".format(file_path))
            return True
        try:
            response = requests.get(paper_url, headers=headers, timeout=10)
            response.raise_for_status()
        except Exception:
            print(f"download failed... {paper['pdf_url']}")
            return False

        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            logger.info("download success {}".format(paper_url))
            logger.info(f"save {file_path}")
            return True
        else:
            print("download failed, status code: {}".format(response.status_code))
            return False

    def get_page(self, url):
        headers = {"User-Agent": random.choice(user_agents)}
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                response.encoding = response.apparent_encoding
                return response.text
            return None
        except RequestException as e:
            print(e)

    def crawling(self, year, venue_name):
        """
        Args:
        Returns:
            paper_list (List of Dict):[
                {
                    "hash_id": hash_id, hash id of the paper
                    "year": year, published year
                    "venue_name": venue_name, venue name
                    "title": title, paper title
                    "pdf_url": pdf_url, paper url
                }
            ]
        """
        paper_list = []
        paper_html_list = []

        def append_paper_to_list(pdf_url, title):
            for paper in paper_html_list:
                if paper["title"] == title:
                    if paper["pdf_url"] != pdf_url:
                        logger.warning(
                            f"Different PDF URL found for the same title '{title}'."
                        )
                    return
            paper_html_list.append({"pdf_url": pdf_url, "title": title})

        if venue_name == "nips":
            if year == "2024":
                return []
            base_url = "https://papers.nips.cc/paper_files/paper/{}"
            target_url = base_url.format(year)
            target_html = self.get_page(target_url)
            soup = BeautifulSoup(target_html, "html.parser")
            ids = soup.find("div", {"class": "container-fluid"}).find_all("li")
            for id in ids:
                a = id.find("a")
                href = a.attrs.get("href")
                pdf_url = "https://papers.nips.cc{}".format(
                    href.replace("hash", "file")
                    .replace("Abstract", "Paper")
                    .replace("html", "pdf")
                )
                title = a.text
                append_paper_to_list(pdf_url, title)
            for paper_html in paper_html_list:
                title = paper_html["title"]
                pdf_url = paper_html["pdf_url"]
                hash_id = generate_hash_id(title)
                paper_list.append(
                    {
                        "hash_id": hash_id,
                        "year": year,
                        "venue_name": venue_name,
                        "title": title,
                        "pdf_url": pdf_url,
                    }
                )

        elif venue_name == "cvpr":
            base_url = "https://openaccess.thecvf.com/CVPR{}"
            dict_cvpr = {
                "2018": ["2018-06-19", "2018-06-20", "2018-06-21"],
                "2019": ["2019-06-18", "2019-06-28", "2019-06-20"],
                "2020": ["2020-06-16", "2020-06-17", "2020-06-18"],
                "2021": ["all"],
                "2022": ["all"],
                "2023": ["all"],
            }
            if year in dict_cvpr.keys():
                day_list = dict_cvpr[year]
                target_url = [
                    base_url.format(year) + "?day={}".format(day) for day in day_list
                ]
            else:
                target_url = [base_url.format(year)]
            print("paper list from {}".format(target_url))
            for url in target_url:
                target_html = self.get_page(url)
                soup = BeautifulSoup(target_html, "html.parser")
                dl_elements = soup.find("div", {"id": "content"}).find_all("dl")
                for dl in dl_elements:
                    dt_elements = dl.find_all("dt")
                    dd_elements = dl.find_all("dd")
                if year in dict_cvpr.keys():
                    dd_elements.pop(0)
                for idx in range(len(dt_elements)):
                    title = dt_elements[idx].text
                    href = dd_elements[idx * 2 + 1].find("a").attrs.get("href")
                    pdf_url = "https://openaccess.thecvf.com/{}".format(href)
                    hash_id = generate_hash_id(title)
                    paper_list.append(
                        {
                            "hash_id": hash_id,
                            "year": year,
                            "venue_name": venue_name,
                            "title": title,
                            "pdf_url": pdf_url,
                        }
                    )

        elif venue_name == "emnlp":
            if year == "2024":
                return []
            if year not in ["2020", "2021", "2022", "2023"]:
                dev_id = "main-container"
            else:
                dev_id = "{}emnlp-main".format(year)
            base_url = "https://aclanthology.org/events/emnlp-{}"
            target_url = base_url.format(year)
            target_html = self.get_page(target_url)
            soup = BeautifulSoup(target_html, "html.parser")
            ids = soup.find("div", {"id": dev_id}).find_all("p")
            for id in ids:
                a = id.find("a")
                pdf_url = a.attrs.get("href")
                title = id.find("strong").get_text()
                append_paper_to_list(pdf_url, title)
            for paper_html in paper_html_list:
                title = paper_html["title"]
                hash_id = generate_hash_id(title)
                pdf_url = paper_html["pdf_url"]
                if "http" not in pdf_url:
                    continue
                paper_list.append(
                    {
                        "hash_id": hash_id,
                        "year": year,
                        "venue_name": venue_name,
                        "title": title,
                        "pdf_url": pdf_url,
                    }
                )

        elif venue_name == "naacl":
            # https://aclanthology.org/
            if year in ["2023", "2020", "2017", "2014"]:
                return []
            dev_id = "main-container"
            base_url = "https://aclanthology.org/events/naacl-{}/"
            target_url = base_url.format(year)
            target_html = self.get_page(target_url)
            soup = BeautifulSoup(target_html, "html.parser")
            ids = soup.find("div", {"id": dev_id}).find_all("p")
            for id in ids:
                a = id.find("a")
                pdf_url = a.attrs.get("href")
                title = id.find("strong").get_text()
                append_paper_to_list(pdf_url, title)
            for paper_html in paper_html_list:
                title = paper_html["title"]
                hash_id = generate_hash_id(title)
                pdf_url = paper_html["pdf_url"]
                paper_list.append(
                    {
                        "hash_id": hash_id,
                        "year": year,
                        "venue_name": venue_name,
                        "title": title,
                        "pdf_url": pdf_url,
                    }
                )

        elif venue_name == "acl":
            dev_id = "main-container"
            base_url = "https://aclanthology.org/events/acl-{}/"
            target_url = base_url.format(year)
            target_html = self.get_page(target_url)
            soup = BeautifulSoup(target_html, "html.parser")
            ids = soup.find("div", {"id": dev_id}).find_all("p")
            for id in ids:
                a = id.find("a")
                pdf_url = a.attrs.get("href")
                title = id.find("strong").get_text()
                append_paper_to_list(pdf_url, title)

            for paper_html in paper_html_list:
                title = paper_html["title"]
                hash_id = generate_hash_id(title)
                pdf_url = paper_html["pdf_url"]
                if "http" not in pdf_url:
                    continue
                paper_list.append(
                    {
                        "hash_id": hash_id,
                        "year": year,
                        "venue_name": venue_name,
                        "title": title,
                        "pdf_url": pdf_url,
                    }
                )

        elif venue_name == "icml":
            hit = {
                "2024": "v235",
                "2023": "v202",
                "2022": "v162",
                "2021": "v139",
                "2020": "v119",
                "2019": "v97",
                "2018": "v80",
                "2017": "v70",
                "2016": "v48",
                "2015": "v37",
                "2014": "v32",
                "2013": "v28",
            }
            dev_id = "container"
            base_url = "https://proceedings.mlr.press/{}/"
            target_url = base_url.format(hit[year])
            target_html = self.get_page(target_url)
            soup = BeautifulSoup(target_html, "html.parser")
            ids = soup.find("main", {"class": "page-content"}).find_all(
                "div", {"class": "paper"}
            )
            for id in ids:
                title = id.find("p", class_="title").text
                pdf_url = id.find("a", text="Download PDF")["href"]
                append_paper_to_list(pdf_url, title)
            for paper_html in paper_html_list:
                title = paper_html["title"]
                hash_id = generate_hash_id(title)
                pdf_url = paper_html["pdf_url"]
                paper_list.append(
                    {
                        "hash_id": hash_id,
                        "year": year,
                        "venue_name": venue_name,
                        "title": title,
                        "pdf_url": pdf_url,
                    }
                )
        return paper_list
