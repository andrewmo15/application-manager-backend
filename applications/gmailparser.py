from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import concurrent.futures, base64, re
from bs4 import BeautifulSoup
from datetime import timedelta, datetime

class GmailParser:
    
    def __init__(self, access_token, last_refresh):
        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        self.creds = Credentials(access_token, scopes=SCOPES)
        self.MAX_RESULTS = 500
        self.last_refresh = self.getPrevDay(last_refresh)

    def getPrevDay(self, last_refresh):
        date_format = '%Y-%m-%d'
        date_obj = datetime.strptime(last_refresh, date_format)
        prev_day = date_obj - timedelta(1)
        return prev_day.strftime(date_format)

    def getEmailInfo(self, emailID):
        service = build('gmail', 'v1', credentials=self.creds)
        txt = service.users().messages().get(userId='me', id=emailID).execute()
        payload = txt['payload']
        headers = payload['headers']
        subject = ""
        date = ""
        for i in headers:
            if i['name'] == "Date":
                date = i["value"]
            if i['name'] == 'Subject':
                subject = i['value']
        body = ""
        try:
            data = None
            if payload['body']['size'] == 0:
                data = payload['parts'][0]['body']['data']
            else:
                data=payload['body']['data']
            data=data.replace("-", '+').replace('_', "/")
            body=base64.b64decode(data)
            body=self.cleanText(body)
        except:
            pass
        return {"subject": subject, "date": date, "body": body}

    def getEmails(self):
        service = build('gmail', 'v1', credentials=self.creds)
        results = service.users().messages().list(userId='me', maxResults=self.MAX_RESULTS, q=f"after:{self.last_refresh}").execute()
        if results['resultSizeEstimate'] == 0:
            return []        
        ids = [r["id"] for r in results["messages"]]
        nextPageToken = None if not "nextPageToken" in results else results["nextPageToken"]
        while nextPageToken:
            results = service.users().messages().list(userId='me', pageToken=nextPageToken).execute()
            nextPageToken = None if not "nextPageToken" in results else results["nextPageToken"]
            ids += [r["id"] for r in results["messages"]]
        rtn = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            emails = executor.map(self.getEmailInfo, ids)
            for email in emails:
                rtn.append(email)
        return rtn
    
    def getTextFromHTML(self, html):
        soup = BeautifulSoup(html, features="html.parser")
        # rip out all scripts and style elements
        for script in soup(["script", "style"]):
            script.extract()    
        text = soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = " ".join(chunk for chunk in chunks if chunk)
        return text

    def cleanText(self, text):
        text = str(text, 'utf-8')
        # remove whitespace
        text = " ".join(text.split())
        # remove links
        text = re.sub(r'http\S+', '', text)
        # convert HTML to plain text
        text = self.getTextFromHTML(text)
        # remove characters to prevent cell overflow
        if len(text) > 32000:
            return text[:32000]
        return text